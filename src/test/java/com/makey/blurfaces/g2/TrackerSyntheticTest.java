package com.makey.blurfaces.g2;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Synthetic trajectory test harness for the face tracker:
 * SourceTracks, FaceTrack, One Euro filter, hysteresis, prediction,
 * coasting, and directional motion expansion.
 *
 * Metrics verified:
 * - «лицо открыто» (face exposure prevention / ground-truth containment)
 * - «маска» (directional gain expansion along velocity vector, gain bounding)
 * - «дыхание» (suppression of gain pulsation / breathing on detector noise)
 */
public class TrackerSyntheticTest {

    private static final long DT_30FPS_NS = 33_333_333L; // ~33.3ms per frame
    private static final String TEST_SOURCE = "camera-preview-0";

    @Before
    public void setUp() {
        Main.configuredConfidence = Main.TRACK_NEW_MIN_CONFIDENCE;
    }

    @After
    public void tearDown() {
        Main.configuredConfidence = Main.TRACK_NEW_MIN_CONFIDENCE;
    }

    /**
     * a) Still Face & Noise / Breathing Test:
     * A static face with small detector jitter (+/- 2px noise in normalized coords ~ +/- 0.002).
     * Verify that:
     * - One Euro filter smooths jitter (position variance reduced).
     * - followGain deadband prevents breathing/pulsing (gain does not oscillate).
     * - Gain stays bounded within safe limits.
     * - Metric «дыхание»: breathing amplitude (max(gain) - min(gain)) <= TRACK_GAIN_HYSTERESIS.
     * - Metric «лицо открыто»: 100% coverage, face is never exposed.
     */
    @Test
    public void testStillFaceNoiseAndBreathing() {
        Main.SourceTracks tracker = new Main.SourceTracks();
        float gtX = 0.50f;
        float gtY = 0.50f;
        float rx = 0.08f;
        float ry = 0.10f;

        long baseTimeNs = 1_000_000_000L;
        int totalFrames = 60; // 2 seconds of 30fps video

        List<Float> rawNoiseX = new ArrayList<>();
        List<Float> oneEuroX = new ArrayList<>();
        List<Float> gainXHistory = new ArrayList<>();
        int exposedFrames = 0;

        for (int frame = 0; frame < totalFrames; frame++) {
            long now = baseTimeNs + frame * DT_30FPS_NS;

            // Small detector jitter (+/- 2 pixels in 1080p ~ 0.0018 - 0.002)
            float noiseX = (float) (Math.sin(frame * 1.7) * 0.002);
            float noiseY = (float) (Math.cos(frame * 2.3) * 0.002);
            float measuredX = gtX + noiseX;
            float measuredY = gtY + noiseY;
            rawNoiseX.add(measuredX);

            Main.FaceGeometry detection = makeSingleDetection(measuredX, measuredY, rx, ry, 0.90f, now);
            tracker.update(detection, now);

            Main.FaceTrack track = tracker.tracks[0];
            assertNotNull("Track 0 must be active", track);
            oneEuroX.add(track.values[0]);
            gainXHistory.add(track.drawGainX);

            // Render prediction query (e.g. 16ms render horizon)
            long renderTime = now + 16_000_000L;
            Main.FaceGeometry rendered = tracker.geometryAt(renderTime, TEST_SOURCE);

            assertNotNull("Rendered geometry must not be null for active face", rendered);
            assertEquals("Exactly one face tracked", 1, rendered.count);

            float outX = rendered.faces[0];
            float outY = rendered.faces[1];
            float outRx = rendered.faces[2];
            float outRy = rendered.faces[5];

            // Metric «лицо открыто»: verify face center and contour are inside the blur ellipse
            boolean covered = isPointInsideEllipse(gtX, gtY, outX, outY, outRx, outRy);
            if (!covered) {
                exposedFrames++;
            }
        }

        // 1. Verify One Euro filter smooths jitter (variance of smoothed < raw)
        // Skip first 10 frames of filter warm-up
        double rawVariance = computeVariance(rawNoiseX.subList(10, totalFrames));
        double filteredVariance = computeVariance(oneEuroX.subList(10, totalFrames));
        assertTrue("One Euro filtered variance (" + filteredVariance + ") should be less than raw noise variance (" + rawVariance + ")",
                filteredVariance < rawVariance);

        // 2. Metric «дыхание» (Breathing):
        // Verify followGain deadband prevents pulsing under detector noise
        // Direct deadband verification on followGain:
        float baseGain = 1.10f;
        float smallDropTarget = baseGain - (Main.TRACK_GAIN_HYSTERESIS * 0.8f);
        float preservedGain = Main.followGain(baseGain, smallDropTarget, 0.033f);
        assertEquals("followGain deadband must completely absorb drops <= TRACK_GAIN_HYSTERESIS",
                baseGain, preservedGain, 1e-6f);

        // Steady-state breathing amplitude across settled frames (frames 30 to 60)
        float minGain = Float.MAX_VALUE;
        float maxGain = -Float.MAX_VALUE;
        for (int i = 30; i < gainXHistory.size(); i++) {
            float g = gainXHistory.get(i);
            if (g < minGain) minGain = g;
            if (g > maxGain) maxGain = g;
        }
        float breathingAmplitude = maxGain - minGain;
        assertTrue("Steady-state breathing amplitude (" + breathingAmplitude + ") must remain tightly bounded (< 0.08)",
                breathingAmplitude <= 0.08f);

        // 3. Gain stays bounded tightly
        for (float g : gainXHistory) {
            assertTrue("Gain (" + g + ") must be >= 1.0", g >= 1.0f);
            assertTrue("Gain (" + g + ") must be <= TRACK_MAX_GAIN (" + Main.TRACK_MAX_GAIN + ")",
                    g <= Main.TRACK_MAX_GAIN);
            assertTrue("Still face gain (" + g + ") must stay tightly bounded <= 1.15", g <= 1.15f);
        }

        // 4. Metric «лицо открыто»: zero unblurred exposures
        assertEquals("Face must remain 100% covered across all frames", 0, exposedFrames);
    }

    /**
     * b) Rapid Motion / Jerk Acceleration Test:
     * A face moving at high speed with acceleration along the X axis.
     * Verify that:
     * - Velocity and acceleration estimates track the moving face.
     * - Predicted position anticipates the moving face forward along trajectory.
     * - Mask gain expands along the velocity vector (gainX > gainY for X-motion).
     * - Metric «маска»: anisotropic dilation covers motion direction.
     * - Metric «лицо открыто»: moving face stays covered at render horizon.
     */
    @Test
    public void testRapidMotionAndJerkAcceleration() {
        Main.SourceTracks tracker = new Main.SourceTracks();
        float x0 = 0.20f;
        float y0 = 0.50f;
        float rx = 0.08f;
        float ry = 0.10f;
        float v0 = 0.35f; // high initial velocity (0.35 screen widths per second)
        float accel = 0.80f; // jerk acceleration (0.80 units/sec^2)

        long baseTimeNs = 1_000_000_000L;
        int frames = 25;
        int exposedFrames = 0;

        for (int frame = 0; frame < frames; frame++) {
            long now = baseTimeNs + frame * DT_30FPS_NS;
            float t = (float) (frame * DT_30FPS_NS) / 1_000_000_000f;

            // Ground truth kinematics: x(t) = x0 + v0*t + 0.5*a*t^2
            float gtX = x0 + v0 * t + 0.5f * accel * t * t;
            float gtY = y0;

            Main.FaceGeometry detection = makeSingleDetection(gtX, gtY, rx, ry, 0.95f, now);
            tracker.update(detection, now);

            Main.FaceTrack track = tracker.tracks[0];
            assertNotNull("Track must be present", track);

            if (frame > 5) {
                // Verify velocity estimate is positive and significant along X
                assertTrue("Estimated X velocity (" + track.velocity[0] + ") must be > 0.20",
                        track.velocity[0] > 0.20f);
                // Velocity along Y should remain near 0
                assertTrue("Estimated Y velocity (" + track.velocity[1] + ") should be near 0",
                        Math.abs(track.velocity[1]) < 0.05f);
                // Acceleration along X should be positive
                assertTrue("Estimated X acceleration (" + track.accel[0] + ") must be > 0",
                        track.accel[0] > 0.0f);

                // Render query at horizon (33ms ahead)
                long renderTime = now + DT_30FPS_NS;
                Main.FaceGeometry predicted = tracker.geometryAt(renderTime, TEST_SOURCE);
                assertNotNull("Predicted geometry should not be null", predicted);

                float renderGtX = x0 + v0 * (t + 0.0333f) + 0.5f * accel * (t + 0.0333f) * (t + 0.0333f);

                // Check forward prediction: predicted center anticipates motion forward
                assertTrue("Predicted center (" + predicted.faces[0] + ") must anticipate motion ahead of current measurement (" + gtX + ")",
                        predicted.faces[0] >= track.values[0]);

                // Metric «маска»: directional gain expansion along velocity vector
                // Motion is horizontal, so drawGainX should be greater than drawGainY
                assertTrue("Mask must expand along velocity vector: gainX (" + track.drawGainX
                                + ") should be >= gainY (" + track.drawGainY + ")",
                        track.drawGainX >= track.drawGainY);
                assertTrue("gainX must never exceed TRACK_MAX_GAIN (" + Main.TRACK_MAX_GAIN + ")",
                        track.drawGainX <= Main.TRACK_MAX_GAIN);
                assertTrue("Perpendicular gainY (" + track.drawGainY + ") must stay tightly bounded <= 1.25",
                        track.drawGainY <= 1.25f);
                assertTrue("Mask area expansion factor (" + (track.drawGainX * track.drawGainY)
                                + ") must never exceed 2.0x base area",
                        (track.drawGainX * track.drawGainY) <= 2.0f);

                // Metric «лицо открыто»: predicted ellipse covers ground truth position at render time
                boolean covered = isPointInsideEllipse(renderGtX, gtY,
                        predicted.faces[0], predicted.faces[1],
                        predicted.faces[2], predicted.faces[5]);
                if (!covered) {
                    exposedFrames++;
                }
            }
        }

        assertEquals("Face must never be exposed during high-speed accelerated motion", 0, exposedFrames);
    }

    /**
     * c) Occlusion & Coasting Test:
     * Face detected for 500ms, then detector drops it for 400ms.
     * Verify that:
     * - During coasting (TRACK_COAST_NS), geometryAt still returns non-null geometry.
     * - Coordinates are preserved (no NaN, center stays near last known center).
     * - Ellipse grows to prevent unblurred exposure (TRACK_LOST_MARGIN increases gain).
     * - After coast limit expires (hold + coast), track cleanly expires and returns null.
     */
    @Test
    public void testOcclusionAndCoasting() {
        Main.SourceTracks tracker = new Main.SourceTracks();
        float faceX = 0.40f;
        float faceY = 0.45f;
        float rx = 0.08f;
        float ry = 0.10f;

        long baseTimeNs = 1_000_000_000L;

        // 1. Detect face steadily for 500ms (15 frames)
        int initialFrames = 15;
        for (int i = 0; i < initialFrames; i++) {
            long now = baseTimeNs + i * DT_30FPS_NS;
            Main.FaceGeometry det = makeSingleDetection(faceX, faceY, rx, ry, 0.90f, now);
            tracker.update(det, now);
        }

        long lastDetectionTime = baseTimeNs + (initialFrames - 1) * DT_30FPS_NS;
        Main.FaceGeometry preOcclusion = tracker.geometryAt(lastDetectionTime, TEST_SOURCE);
        assertNotNull(preOcclusion);
        float baselineRadiusX = preOcclusion.faces[2];

        // 2. Detector drops face for 400ms (no detections)
        // Note: 400ms is longer than adaptiveHoldNanos (350ms) but well within TRACK_COAST_NS (700ms).
        int occlusionSteps = 12; // 12 * 33.3ms = 400ms
        float lastObservedRadiusX = baselineRadiusX;

        for (int i = 1; i <= occlusionSteps; i++) {
            long coastNow = lastDetectionTime + i * DT_30FPS_NS;

            // Update with empty detections (detector dropped the face)
            Main.FaceGeometry emptyDet = new Main.FaceGeometry(new float[0], 0, coastNow, TEST_SOURCE);
            tracker.update(emptyDet, coastNow);

            Main.FaceGeometry coastGeometry = tracker.geometryAt(coastNow, TEST_SOURCE);

            // Verify geometryAt still returns non-null geometry during coasting
            assertNotNull("Geometry must not be null during coasting at dt=" + (i * 33.3f) + "ms", coastGeometry);
            assertEquals("Must report 1 face during coasting", 1, coastGeometry.count);

            // Coordinates preserved
            float cx = coastGeometry.faces[0];
            float cy = coastGeometry.faces[1];
            assertEquals("Center X must be preserved during coasting", faceX, cx, 0.02f);
            assertEquals("Center Y must be preserved during coasting", faceY, cy, 0.02f);

            // Verify ellipse grows to prevent unblurred exposure
            float currentRadiusX = coastGeometry.faces[2];
            assertTrue("Coasting ellipse radius X (" + currentRadiusX + ") must be >= baseline (" + baselineRadiusX + ")",
                    currentRadiusX >= baselineRadiusX * 0.99f);
            lastObservedRadiusX = currentRadiusX;
        }

        // Verify that by the end of 400ms occlusion, the ellipse expanded significantly
        assertTrue("Ellipse must have expanded due to TRACK_LOST_MARGIN (last=" + lastObservedRadiusX
                        + ", base=" + baselineRadiusX + ")",
                lastObservedRadiusX > baselineRadiusX);
        assertTrue("Coasting radius must not balloon excessively (<= 1.40 * baseline)",
                lastObservedRadiusX <= baselineRadiusX * 1.40f);

        // 3. After coast limit expires (hold + coast = 350ms + 700ms = 1050ms):
        // Continue advancing empty frames at 30fps past the 1050ms holdLimit (e.g. up to 1200ms)
        int totalOcclusionSteps = 36; // 36 * 33.3ms = 1200ms
        for (int i = occlusionSteps + 1; i <= totalOcclusionSteps; i++) {
            long coastNow = lastDetectionTime + i * DT_30FPS_NS;
            Main.FaceGeometry emptyDet = new Main.FaceGeometry(new float[0], 0, coastNow, TEST_SOURCE);
            tracker.update(emptyDet, coastNow);
        }

        long expiredTime = lastDetectionTime + totalOcclusionSteps * DT_30FPS_NS;
        Main.FaceGeometry expiredGeometry = tracker.geometryAt(expiredTime, TEST_SOURCE);
        assertNull("After coast expiration (> 1050ms), geometryAt must cleanly return null", expiredGeometry);
        assertNull("Track slot must be cleanly cleared", tracker.tracks[0]);
    }

    /** Confidence hysteresis belongs to native ByteTrack, not this renderer. */
    @Test
    public void testNativeConfirmedLowLightHeadsAreNotFilteredAgain() {
        for (float configured : new float[]{.35f, .25f, .18f}) {
            Main.configuredConfidence = configured;
            Main.SourceTracks tracker = new Main.SourceTracks();
            long t = 1_000_000_000L;
            // Native confirmed a head below the configured daytime threshold.
            tracker.update(makeSingleDetection(.50f, .50f, .08f, .10f, .15f, t), t);
            assertNotNull("Native confirmation must create a mask", tracker.tracks[0]);
            assertEquals(t, tracker.tracks[0].lastSeenNanos);
            t += DT_30FPS_NS;
            tracker.update(makeSingleDetection(.51f, .51f, .08f, .10f, .10f, t), t);
            assertEquals("Native low-score continuation must update the mask", t,
                    tracker.tracks[0].lastSeenNanos);
            assertNull(tracker.tracks[1]);
            t += DT_30FPS_NS;
            tracker.update(makeSingleDetection(.90f, .90f, .08f, .10f, .15f, t), t);
            assertNotNull("A second confirmed head also needs protection", tracker.tracks[1]);
            assertEquals(.90f, tracker.tracks[1].values[0], .001f);
            assertEquals(2, tracker.geometryAt(t, TEST_SOURCE).count);
        }
    }

    /**
     * e) Multi-Face Identity Independence:
     * Two faces crossing or passing near each other.
     * Verify global-nearest matching maintains independent tracking slots without
     * swapping or discarding tracks, even when detection order is reversed on alternating frames.
     */
    @Test
    public void testMultiFaceIdentityIndependence() {
        Main.SourceTracks tracker = new Main.SourceTracks();
        long baseTimeNs = 1_000_000_000L;

        // Face 1 moves right: x from 0.25 to 0.45, y = 0.40
        // Face 2 moves left:  x from 0.75 to 0.55, y = 0.60
        int frames = 20;

        for (int frame = 0; frame < frames; frame++) {
            long now = baseTimeNs + frame * DT_30FPS_NS;
            float progress = (float) frame / (float) (frames - 1);

            float f1X = 0.25f + 0.20f * progress;
            float f1Y = 0.40f;
            float f2X = 0.75f - 0.20f * progress;
            float f2Y = 0.60f;

            // Reverse order on odd frames to stress test order independence: [F1, F2] vs [F2, F1]
            Main.FaceGeometry multiDet;
            if (frame % 2 == 0) {
                multiDet = makeMultiDetection(
                        new float[][] {
                                { f1X, f1Y, 0.07f, 0.09f },
                                { f2X, f2Y, 0.07f, 0.09f }
                        },
                        new float[] { 0.90f, 0.90f },
                        now
                );
            } else {
                multiDet = makeMultiDetection(
                        new float[][] {
                                { f2X, f2Y, 0.07f, 0.09f },
                                { f1X, f1Y, 0.07f, 0.09f }
                        },
                        new float[] { 0.90f, 0.90f },
                        now
                );
            }

            tracker.update(multiDet, now);

            // Verify both tracks are present
            assertNotNull("Track 0 must be active at frame " + frame, tracker.tracks[0]);
            assertNotNull("Track 1 must be active at frame " + frame, tracker.tracks[1]);

            // Verify slot 0 persistently tracks Face 1 (y near 0.40) and slot 1 tracks Face 2 (y near 0.60)
            float t0Y = tracker.tracks[0].values[1];
            float t1Y = tracker.tracks[1].values[1];
            assertEquals("Track 0 must maintain Face 1 identity (y ~ 0.40)", 0.40f, t0Y, 0.05f);
            assertEquals("Track 1 must maintain Face 2 identity (y ~ 0.60)", 0.60f, t1Y, 0.05f);

            // Verify X coordinates track respective faces
            float t0X = tracker.tracks[0].values[0];
            float t1X = tracker.tracks[1].values[0];
            assertEquals("Track 0 X must follow Face 1", f1X, t0X, 0.05f);
            assertEquals("Track 1 X must follow Face 2", f2X, t1X, 0.05f);

            // Query geometryAt to ensure both faces are returned
            Main.FaceGeometry outGeo = tracker.geometryAt(now + 16_000_000L, TEST_SOURCE);
            assertNotNull(outGeo);
            assertEquals("Both faces must be rendered in geometry output", 2, outGeo.count);
        }
    }

    /** The native head detector replaced MediaPipe landmarks and does not infer yaw. */
    @Test
    public void testNativeErrorsInvalidateGeometryAndRecover() {
        String source = "native-error";
        long t = System.nanoTime();
        Main.SOURCE_ACTIVE_SINCE.put(source, t);
        Main.SourceTracks tracker = new Main.SourceTracks();
        Main.SOURCE_TRACKS.put(source, tracker);
        for (int status : new int[]{-1, -2, -3, -4, -5, Main.MAX_FACES + 1}) {
            tracker.update(makeSingleDetection(.5f, .5f, .08f, .10f, .9f, t), t);
            assertNotNull(tracker.geometryAt(t, source));
            try {
                tracker.update(new Main.FaceGeometry(new float[24], new float[4],
                        new float[4], status, t + 1, source), t + 1);
                org.junit.Assert.fail("Invalid status must fail: " + status);
            } catch (IllegalArgumentException expected) { }
            assertNull(tracker.geometryAt(t + 1, source));
            assertFalse(Main.hasFreshResult(source, t + 1));
            assertEquals(-1, Main.resolveFaceCount(source, t + 1));
            assertEquals(0L, tracker.lastPublishedNanos);
            // A real successful empty observation, unlike an error, may clear the frame.
            tracker.update(new Main.FaceGeometry(new float[0], 0, t + 2, source), t + 2);
            assertTrue(Main.hasFreshResult(source, t + 2));
            assertEquals(0, Main.resolveFaceCount(source, t + 2));
            t += 3;
        }
        // Java exceptions invalidate a previously successful publication as well.
        Main.invalidateResult(source);
        assertEquals(-1, Main.resolveFaceCount(source, t));
        Main.SOURCE_TRACKS.remove(source);
        Main.SOURCE_ACTIVE_SINCE.remove(source);
    }

    /**
     * g) FaceTrack Yaw Hold Extension and Profile Coasting:
     * - FaceTrack holdLimit increases proactively when |yaw| > 0.25.
     * - significantSpeed margin is kept alive in profile view.
     * - Under occlusion, a face track with profile yaw survives longer than base hold limit.
     */
    @Test
    public void testFaceTrackYawProfileHoldAndMargin() {
        float[] det = new float[Main.FACE_STRIDE];
        det[0] = 0.50f; det[1] = 0.50f;
        det[2] = 0.08f; det[3] = 0.0f;
        det[4] = 0.0f;  det[5] = 0.10f;

        long t0 = 1_000_000_000L;
        Main.FaceTrack frontalTrack = new Main.FaceTrack(det, 0, t0, t0, 0.0f);
        Main.FaceTrack profileTrack = new Main.FaceTrack(det, 0, t0, t0, 0.75f);

        // 1. Verify holdLimit extension
        long baseHold = 1_050_000_000L;
        long frontalHold = frontalTrack.holdLimit(baseHold);
        long profileHold = profileTrack.holdLimit(baseHold);

        assertEquals("Frontal track holdLimit equals baseHold", baseHold, frontalHold);
        assertEquals("Profile track holdLimit extended by 1.0 + 0.40 * 0.75 = 1.30x",
                (long) (baseHold * 1.30f), profileHold);
        assertTrue("Profile holdLimit must be strictly greater than frontal", profileHold > frontalHold);

        // 2. Verify significantSpeed margin boost in predict()
        float[] outFrontal = new float[Main.FACE_STRIDE];
        float[] outProfile = new float[Main.FACE_STRIDE];
        long predictTime = t0 + 33_333_333L;

        frontalTrack.predict(outFrontal, 0, predictTime, frontalHold);
        profileTrack.predict(outProfile, 0, predictTime, profileHold);

        // Profile track must produce wider gain / radius than frontal track
        assertTrue("Profile track gainX (" + profileTrack.drawGainX + ") should be >= frontal (" + frontalTrack.drawGainX + ")",
                profileTrack.drawGainX >= frontalTrack.drawGainX);

        // 3. Occlusion coasting in SourceTracks:
        Main.SourceTracks tracker = new Main.SourceTracks();
        float[] yaws = new float[] { 0.75f };
        Main.FaceGeometry profileDet = new Main.FaceGeometry(det, new float[] { 0.90f }, yaws, 1, t0, TEST_SOURCE);
        tracker.update(profileDet, t0);

        // Normal base hold limit is adaptiveHoldNanos (350ms) + coast (700ms) = 1050ms.
        // For profileTrack (yaw=0.75), holdLimit is extended to 1050ms * 1.30 = 1365ms.
        // Query at 1150ms elapsed: normal track would expire (> 1050ms), but profile track survives!
        long occludedTime = t0 + 1_150_000_000L;
        Main.FaceGeometry emptyDet = new Main.FaceGeometry(new float[0], 0, occludedTime, TEST_SOURCE);
        tracker.update(emptyDet, occludedTime);

        Main.FaceGeometry surviving = tracker.geometryAt(occludedTime, TEST_SOURCE);
        assertNotNull("Profile track must survive at 1150ms occlusion due to yaw hold extension", surviving);
        assertEquals(1, surviving.count);
    }

    /**
     * Edge case verification for yaw calculations, bounds checking, and NaN safety.
     */
    @Test
    public void testYawRobustnessAndEdgeCases() {
        // The JNI boundary replaces the retired landmark decoder. Malformed
        // geometry invalidates the entire result rather than exposing one face.
        assertFalse(Main.validNativeResult(null));
        Main.FaceGeometry valid = makeSingleDetection(.5f, .5f, .1f, .1f, .9f, 1L);
        assertTrue(Main.validNativeResult(valid));
        for (int i = 0; i < Main.FACE_STRIDE; i++) {
            float saved = valid.faces[i];
            for (float bad : new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
                valid.faces[i] = bad;
                assertFalse(Main.validNativeResult(valid));
            }
            valid.faces[i] = saved;
        }
        for (float score : new float[]{Float.NaN, Float.POSITIVE_INFINITY, -.1f, 1.1f}) {
            valid.scores[0] = score;
            assertFalse(Main.validNativeResult(valid));
        }
        valid.scores[0] = .9f;
        valid.faces[2] = 0f;
        assertFalse("Degenerate ellipse cannot cover a head", Main.validNativeResult(valid));
        assertFalse(Main.validNativeResult(new Main.FaceGeometry(new float[5], new float[1], 1, 1L, TEST_SOURCE)));
        assertFalse(Main.validNativeResult(new Main.FaceGeometry(new float[6], new float[0], 1, 1L, TEST_SOURCE)));

        // 4. FaceTrack NaN yaw resilience
        float[] det = new float[Main.FACE_STRIDE];
        det[0] = 0.5f; det[1] = 0.5f; det[2] = 0.1f; det[3] = 0f; det[4] = 0f; det[5] = 0.1f;
        Main.FaceTrack track = new Main.FaceTrack(det, 0, 1_000_000_000L, 1_000_000_000L, Float.NaN);
        assertEquals("NaN yaw in constructor must default to 0.0", 0.0f, track.yaw, 1e-6f);
        track.observe(det, 0, 1_033_333_333L, 1_033_333_333L, Float.NaN);
        assertEquals("NaN yaw in observe must be ignored/defaulted to 0.0", 0.0f, track.yaw, 1e-6f);

        // 5. SourceTracks isExpired consistency
        Main.SourceTracks tracks = new Main.SourceTracks();
        Main.FaceTrack profileFace = new Main.FaceTrack(det, 0, 1_000_000_000L, 1_000_000_000L, 0.75f);
        long baseHold = tracks.holdLimit(); // ~1050ms
        long profileHold = tracks.holdLimit(profileFace); // ~1365ms
        assertTrue("Profile hold must be greater than base hold", profileHold > baseHold);
        long tQuery = 1_000_000_000L + baseHold + 50_000_000L; // 50ms past base hold, but well before profile hold
        assertFalse("Profile track must not be expired past base hold within profile hold",
                tracks.isExpired(profileFace, tQuery));
    }

    /**
     * Metric: «неузнаваемо на итоговых 320-512px независимо от удаления/размера лица».
     * Verifies adaptive blur radius scale and adaptive pixel grid based on face size:
     * - Large face (radius >= 0.18): scale is 1.0f.
     * - Smaller face (down to 0.04): scale smoothly increases up to 2.5f.
     * - Distant / tiny face (radius <= 0.04): scale capped at 2.5f.
     * - Multi-face: minimum face radius governs blur scale and pixel grid.
     * - Pixelation grid: face diameter is broken into coarse blocks (~3-4 blocks across face)
     *   preventing feature recognition across all zoom/distance levels on 320-512px output.
     * - Zero faces / fail-closed: defaults to scale 1.0f and grid 32.0f.
     */
    @Test
    public void testAdaptiveBlurRadiusAndPixelGridFromFaceSize() {
        // 1. Large face (radius >= 0.18)
        float[] largeFaces = new float[] { 0.5f, 0.5f, 0.20f, 0.0f, 0.0f, 0.25f };
        float minRadiusLarge = Main.computeMinFaceRadius(1, largeFaces);
        assertEquals(0.20f, minRadiusLarge, 1e-5f);
        float blurScaleLarge = Main.computeBlurRadiusScale(minRadiusLarge);
        assertEquals(1.0f, blurScaleLarge, 1e-5f);
        float gridLarge = Main.computePixelGrid(minRadiusLarge, 1);
        assertEquals(12.0f, gridLarge, 1e-5f);

        // 2. Medium face (radius 0.09) -> scale should double (2.0f)
        float[] medFaces = new float[] { 0.5f, 0.5f, 0.09f, 0.0f, 0.0f, 0.12f };
        float minRadiusMed = Main.computeMinFaceRadius(1, medFaces);
        assertEquals(0.09f, minRadiusMed, 1e-5f);
        float blurScaleMed = Main.computeBlurRadiusScale(minRadiusMed);
        assertEquals(2.0f, blurScaleMed, 1e-5f);
        float gridMed = Main.computePixelGrid(minRadiusMed, 1);
        assertEquals(3.0f / (0.09f * 2.0f), gridMed, 1e-5f); // ~16.67f

        // 3. Small face (radius 0.04) -> scale reaches maximum (2.5f)
        float[] smallFaces = new float[] { 0.5f, 0.5f, 0.04f, 0.0f, 0.0f, 0.05f };
        float minRadiusSmall = Main.computeMinFaceRadius(1, smallFaces);
        assertEquals(0.04f, minRadiusSmall, 1e-5f);
        float blurScaleSmall = Main.computeBlurRadiusScale(minRadiusSmall);
        assertEquals(2.5f, blurScaleSmall, 1e-5f);
        float gridSmall = Main.computePixelGrid(minRadiusSmall, 1);
        assertEquals(3.0f / (0.04f * 2.0f), gridSmall, 1e-5f); // 37.5f

        // 4. Tiny / distant face below 0.04 -> clamped to 2.5f and 48.0f grid
        float[] tinyFaces = new float[] { 0.5f, 0.5f, 0.02f, 0.0f, 0.0f, 0.03f };
        float minRadiusTiny = Main.computeMinFaceRadius(1, tinyFaces);
        assertEquals(0.02f, minRadiusTiny, 1e-5f);
        float blurScaleTiny = Main.computeBlurRadiusScale(minRadiusTiny);
        assertEquals(2.5f, blurScaleTiny, 1e-5f);
        float gridTiny = Main.computePixelGrid(minRadiusTiny, 1);
        assertEquals(48.0f, gridTiny, 1e-5f); // clamped to max 48.0f

        // 5. Multi-face: 1 large face (radius 0.22) and 1 small face (radius 0.05)
        float[] multiFaces = new float[2 * Main.FACE_STRIDE];
        // face 0: large
        multiFaces[0] = 0.3f; multiFaces[1] = 0.3f;
        multiFaces[2] = 0.22f; multiFaces[3] = 0.0f;
        multiFaces[4] = 0.0f; multiFaces[5] = 0.28f;
        // face 1: small
        int off1 = Main.FACE_STRIDE;
        multiFaces[off1] = 0.7f; multiFaces[off1 + 1] = 0.7f;
        multiFaces[off1 + 2] = 0.05f; multiFaces[off1 + 3] = 0.0f;
        multiFaces[off1 + 4] = 0.0f; multiFaces[off1 + 5] = 0.07f;
        float minRadiusMulti = Main.computeMinFaceRadius(2, multiFaces);
        assertEquals("Minimum face radius must pick the smaller face", 0.05f, minRadiusMulti, 1e-5f);
        float blurScaleMulti = Main.computeBlurRadiusScale(minRadiusMulti);
        assertTrue("Blur scale must be increased for smaller face in scene", blurScaleMulti > 1.0f);
        assertEquals(2.5f, blurScaleMulti, 1e-5f); // 0.18 / 0.05 = 3.6 -> clamped to 2.5f
        float gridMulti = Main.computePixelGrid(minRadiusMulti, 2);
        assertEquals(30.0f, gridMulti, 1e-5f); // 3.0 / (0.05 * 2) = 30.0f

        // 6. Zero faces / fail-closed
        float minRadiusZero = Main.computeMinFaceRadius(0, null);
        assertEquals(0.20f, minRadiusZero, 1e-5f);
        assertEquals(1.0f, Main.computeBlurRadiusScale(minRadiusZero), 1e-5f);
        assertEquals(32.0f, Main.computePixelGrid(minRadiusZero, 0), 1e-5f);

        // 7. Metric unrecognizability verification on 320-512px output:
        // Across face diameters ranging from 15px to 300px:
        // - Pixelation blocks across face diameter = diameter * uPixelGrid
        // Must stay in coarse range [2.4, 7.5], with typical range being ~3.0 blocks,
        // so features like eyes, nose, mouth cannot be resolved.
        int[] testResolutions = { 320, 512 };
        for (int res : testResolutions) {
            for (float r = 0.02f; r <= 0.30f; r += 0.02f) {
                float diameterNorm = r * 2.0f;
                float grid = Main.computePixelGrid(r, 1);
                float blocksAcrossFace = diameterNorm * grid;
                assertTrue("Blocks across face must be >= 1.8 to avoid single-block collapse (res=" + res + ", r=" + r + ", blocks=" + blocksAcrossFace + ")",
                        blocksAcrossFace >= 1.8f);
                assertTrue("Blocks across face must be <= 7.5 to eliminate feature recognition (res=" + res + ", r=" + r + ", blocks=" + blocksAcrossFace + ")",
                        blocksAcrossFace <= 7.5f);

                // Small faces (< 0.18 radius) must receive increased blur scale
                float scale = Main.computeBlurRadiusScale(r);
                if (r < 0.18f) {
                    assertTrue("Small face r=" + r + " must have increased blur scale > 1.0", scale > 1.0f);
                } else {
                    assertEquals("Large face r=" + r + " must have blur scale 1.0", 1.0f, scale, 1e-5f);
                }
            }
        }
    }

    // --- Helper & Metric Utility Methods ---

    private static Main.FaceGeometry makeSingleDetection(float x, float y, float rx, float ry, float score, long timeNs) {
        float[] faces = new float[Main.FACE_STRIDE];
        faces[0] = x;
        faces[1] = y;
        faces[2] = rx;
        faces[3] = 0.0f;
        faces[4] = 0.0f;
        faces[5] = ry;
        float[] scores = new float[] { score };
        return new Main.FaceGeometry(faces, scores, 1, timeNs, TEST_SOURCE);
    }

    private static Main.FaceGeometry makeMultiDetection(float[][] faceParams, float[] scores, long timeNs) {
        int count = faceParams.length;
        float[] faces = new float[count * Main.FACE_STRIDE];
        for (int i = 0; i < count; i++) {
            int off = i * Main.FACE_STRIDE;
            faces[off] = faceParams[i][0];     // cx
            faces[off + 1] = faceParams[i][1]; // cy
            faces[off + 2] = faceParams[i][2]; // rx (axisX.x)
            faces[off + 3] = 0.0f;             // axisX.y
            faces[off + 4] = 0.0f;             // axisY.x
            faces[off + 5] = faceParams[i][3]; // ry (axisY.y)
        }
        return new Main.FaceGeometry(faces, scores, count, timeNs, TEST_SOURCE);
    }

    @Test
    public void testCleanFrameTapBlurRadiusScaleClampingAndOverloads() {
        // Clamping range [1.0f, 2.5f] and NaN / Infinity safety
        assertEquals(1.0f, CleanFrameTap.clampBlurRadiusScale(Float.NaN), 1e-5f);
        assertEquals(1.0f, CleanFrameTap.clampBlurRadiusScale(Float.POSITIVE_INFINITY), 1e-5f);
        assertEquals(1.0f, CleanFrameTap.clampBlurRadiusScale(Float.NEGATIVE_INFINITY), 1e-5f);
        assertEquals(1.0f, CleanFrameTap.clampBlurRadiusScale(-5.0f), 1e-5f);
        assertEquals(1.0f, CleanFrameTap.clampBlurRadiusScale(0.0f), 1e-5f);
        assertEquals(1.0f, CleanFrameTap.clampBlurRadiusScale(0.5f), 1e-5f);
        assertEquals(1.0f, CleanFrameTap.clampBlurRadiusScale(1.0f), 1e-5f);
        assertEquals(1.8f, CleanFrameTap.clampBlurRadiusScale(1.8f), 1e-5f);
        assertEquals(2.5f, CleanFrameTap.clampBlurRadiusScale(2.5f), 1e-5f);
        assertEquals(2.5f, CleanFrameTap.clampBlurRadiusScale(3.0f), 1e-5f);
        assertEquals(2.5f, CleanFrameTap.clampBlurRadiusScale(100.0f), 1e-5f);

        // Ensure step calculation never divides by zero or produces non-finite values
        float[] scales = { Float.NaN, -1.0f, 0f, 0.5f, 1.0f, 1.5f, 2.5f, 5.0f, Float.POSITIVE_INFINITY };
        for (float s : scales) {
            float clamped = CleanFrameTap.clampBlurRadiusScale(s);
            float step = (4.5f * clamped) / CleanFrameTap.SIZE;
            assertTrue(Float.isFinite(step));
            assertTrue(step > 0f);
            assertTrue(step >= (4.5f * 1.0f) / CleanFrameTap.SIZE - 1e-6f);
            assertTrue(step <= (4.5f * 2.5f) / CleanFrameTap.SIZE + 1e-6f);
        }

        // Both 5-param and 6-param overloads validate frame input and fail gracefully
        CleanFrameTap tap = new CleanFrameTap();
        int res5 = tap.renderBlur(0, null, null, null, null);
        assertEquals(0, res5);
        assertEquals("invalid frame input", tap.lastError());

        int res6 = tap.renderBlur(0, null, null, null, null, 2.0f);
        assertEquals(0, res6);
        assertEquals("invalid frame input", tap.lastError());
    }

    @Test
    public void testPixelGridUniformCachingFields() throws Exception {
        Class<?> csClass = Class.forName("com.makey.blurfaces.g2.Main$CameraState");
        java.lang.reflect.Field csPixelGrid = csClass.getDeclaredField("pixelGrid");
        assertEquals(int.class, csPixelGrid.getType());

        Class<?> esClass = Class.forName("com.makey.blurfaces.g2.Main$EncoderState");
        java.lang.reflect.Field esPixelGrid = esClass.getDeclaredField("pixelGrid");
        assertEquals(int.class, esPixelGrid.getType());
    }

    /**
     * Item 8 requirement a:
     * LK correctly tracks synthetic translation shifts (e.g. shift image patch by 4 pixels, LK returns ~4 pixels).
     * Also verifies multi-direction shifts, negative shifts, and rejection of flat/uniform patches.
     */
    @Test
    public void testLucasKanadeSyntheticTranslationShift4Pixels() {
        SparseLucasKanadeTracker tracker = new SparseLucasKanadeTracker();
        int width = 192, height = 192;
        float[] ptsX = new float[8];
        float[] ptsY = new float[8];
        int numPoints = SparseLucasKanadeTracker.selectTrackPoints(
                96f / width, 96f / height, 20f / width, 24f / height, width, height, ptsX, ptsY);
        assertEquals(8, numPoints);

        // 1. Shift by +4.0 pixels along X, 0.0 along Y
        float[] prev = generateSyntheticFacePattern(96f, 96f, 16f);
        float[] currShiftX4 = generateSyntheticFacePattern(100f, 96f, 16f);

        SparseLucasKanadeTracker.FlowResult resX4 = tracker.track(
                prev, currShiftX4, width, height, ptsX, ptsY, numPoints);
        assertTrue("Flow result must be valid", resX4.valid);
        assertTrue("At least 2 points must be valid (was " + resX4.validPoints + ")",
                resX4.validPoints >= SparseLucasKanadeTracker.MIN_VALID_POINTS);
        assertEquals("LK must track ~4.0 pixels X shift", 4.0f, resX4.flowX, 0.4f);
        assertEquals("LK must track ~0.0 pixels Y shift", 0.0f, resX4.flowY, 0.4f);

        // 2. Diagonal shift (+4.0 pixels X, -3.0 pixels Y)
        float[] currDiag = generateSyntheticFacePattern(100f, 93f, 16f);
        SparseLucasKanadeTracker.FlowResult resDiag = tracker.track(
                prev, currDiag, width, height, ptsX, ptsY, numPoints);
        assertTrue("Diagonal flow result must be valid", resDiag.valid);
        assertEquals("LK must track ~4.0 pixels X shift", 4.0f, resDiag.flowX, 0.4f);
        assertEquals("LK must track ~-3.0 pixels Y shift", -3.0f, resDiag.flowY, 0.4f);

        // 3. Negative shift (-4.0 pixels X, +2.0 pixels Y)
        float[] currNeg = generateSyntheticFacePattern(92f, 98f, 16f);
        SparseLucasKanadeTracker.FlowResult resNeg = tracker.track(
                prev, currNeg, width, height, ptsX, ptsY, numPoints);
        assertTrue("Negative flow result must be valid", resNeg.valid);
        assertEquals("LK must track ~-4.0 pixels X shift", -4.0f, resNeg.flowX, 0.4f);
        assertEquals("LK must track ~2.0 pixels Y shift", 2.0f, resNeg.flowY, 0.4f);

        // 4. Flat/uniform image: should be rejected due to low gradient determinant
        float[] flat = new float[width * height];
        java.util.Arrays.fill(flat, 128f);
        SparseLucasKanadeTracker.FlowResult resFlat = tracker.track(
                flat, flat, width, height, ptsX, ptsY, numPoints);
        assertFalse("Flat uniform patch must yield invalid flow", resFlat.valid);
    }

    /**
     * Item 8 requirement b:
     * Track position and velocity update with flow during camera jerk.
     */
    @Test
    public void testTrackPositionAndVelocityUpdateWithFlowDuringCameraJerk() {
        Main.SourceTracks tracker = new Main.SourceTracks();
        float initialX = 0.50f;
        float initialY = 0.50f;
        float rx = 0.08f;
        float ry = 0.10f;
        long baseTimeNs = 1_000_000_000L;

        // Establish initial stationary track
        for (int i = 0; i < 5; i++) {
            long now = baseTimeNs + i * DT_30FPS_NS;
            Main.FaceGeometry det = makeSingleDetection(initialX, initialY, rx, ry, 0.95f, now);
            tracker.update(det, now);
        }

        Main.FaceTrack track = tracker.tracks[0];
        assertNotNull("Track 0 must be present", track);
        assertEquals("Initial X velocity should be ~0", 0.0f, track.velocity[0], 0.05f);
        assertEquals("Initial Y velocity should be ~0", 0.0f, track.velocity[1], 0.05f);

        // Simulate camera jerk: 4 pixels right shift on 192x192 frame (~0.0208 normalized)
        float shiftPixelsX = 4.0f;
        float flowXNorm = shiftPixelsX / 192.0f;
        float dt = 0.03333f;

        // Apply optical flow to face track
        track.applyOpticalFlow(flowXNorm, 0.0f, dt);

        // 1. Verify position updated immediately in jerk direction
        assertEquals("Track drawCenter[0] must update by flowXNorm",
                initialX + flowXNorm, track.drawCenter[0], 1e-4f);
        assertEquals("Track values[0] must update by flowXNorm",
                initialX + flowXNorm, track.values[0], 1e-4f);

        // 2. Verify velocity updated in jerk direction
        assertTrue("Track velocity[0] (" + track.velocity[0] + ") must be > 0.20 after jerk",
                track.velocity[0] > 0.20f);
        assertTrue("Track peakSpeed (" + track.peakSpeed + ") must increase",
                track.peakSpeed > 0.20f);

        // 3. Verify prediction at render horizon reflects jerked position before detector update
        long renderTime = baseTimeNs + 5 * DT_30FPS_NS + 16_000_000L;
        Main.FaceGeometry rendered = tracker.geometryAt(renderTime, TEST_SOURCE);
        assertNotNull("Rendered geometry must not be null", rendered);
        assertTrue("Rendered face center (" + rendered.faces[0] + ") must be displaced in jerk direction",
                rendered.faces[0] > initialX);

        // 4. Also verify full pipeline through SourceTracks.applyOpticalFlow
        int width = 192, height = 192;
        float[] prev = generateSyntheticFacePattern(track.drawCenter[0] * width, track.drawCenter[1] * height, 16f);
        // Second jerk: another 3 pixels right
        float[] curr = generateSyntheticFacePattern(track.drawCenter[0] * width + 3.0f, track.drawCenter[1] * height, 16f);

        float prevCenter = track.drawCenter[0];
        tracker.applyOpticalFlow(new SparseLucasKanadeTracker(), prev, curr, width, height, dt);

        assertTrue("drawCenter must advance further after second jerk", track.drawCenter[0] > prevCenter);
        assertTrue("Velocity must remain positive during jerk", track.velocity[0] > 0.20f);
    }

    /**
     * Item 8 requirement: negligible CPU overhead (< 0.5 ms).
     * Tests RGBA to grayscale conversion, point selection, and 8-point 3-iteration LK tracking.
     */
    @Test
    public void testOpticalFlowCpuOverheadUnderHalfMillisecond() {
        int width = 192, height = 192;
        byte[] rgba = new byte[width * height * 4];
        for (int i = 0; i < rgba.length; i += 4) {
            rgba[i] = (byte) (i % 255);
            rgba[i + 1] = (byte) ((i / 2) % 255);
            rgba[i + 2] = (byte) ((i / 3) % 255);
            rgba[i + 3] = (byte) 255;
        }

        SparseLucasKanadeTracker tracker = new SparseLucasKanadeTracker();
        float[] prevGray = generateSyntheticFacePattern(96f, 96f, 16f);
        float[] currGray = generateSyntheticFacePattern(98f, 96f, 16f);
        float[] gray = new float[width * height];
        float[] ptsX = new float[8];
        float[] ptsY = new float[8];

        // Warm up JIT
        for (int i = 0; i < 50; i++) {
            SparseLucasKanadeTracker.rgbaToGrayscale(rgba, gray, width, height, true);
            int pts = SparseLucasKanadeTracker.selectTrackPoints(0.5f, 0.5f, 0.1f, 0.12f, width, height, ptsX, ptsY);
            tracker.track(prevGray, currGray, width, height, ptsX, ptsY, pts);
        }

        // Benchmark 200 frames
        int iterations = 200;
        long startNanos = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            SparseLucasKanadeTracker.rgbaToGrayscale(rgba, gray, width, height, true);
            int pts = SparseLucasKanadeTracker.selectTrackPoints(0.5f, 0.5f, 0.1f, 0.12f, width, height, ptsX, ptsY);
            tracker.track(prevGray, currGray, width, height, ptsX, ptsY, pts);
        }
        long elapsedNanos = System.nanoTime() - startNanos;
        double avgMs = (elapsedNanos / (double) iterations) / 1_000_000.0;

        assertTrue("Average CPU time per frame (" + avgMs + " ms) must be < 0.5 ms", avgMs < 0.5);
    }

    private static float[] generateSyntheticFacePattern(float centerX, float centerY, float sigma) {
        int width = 192, height = 192;
        float[] img = new float[width * height];
        for (int y = 0; y < height; y++) {
            float dy = y - centerY;
            for (int x = 0; x < width; x++) {
                float dx = x - centerX;
                // Rich multi-directional gradient patch simulating facial luminance
                float r = (float) Math.hypot(dx, dy);
                float base = 128.0f - 2.5f * dx + 1.8f * dy;
                float spot = (float) (Math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) * 60.0);
                float cross = (float) (Math.sin(dx * 0.15) * 20.0 + Math.cos(dy * 0.15) * 20.0);
                img[y * width + x] = base + spot + cross;
            }
        }
        return img;
    }

    private static boolean isPointInsideEllipse(float px, float py, float cx, float cy, float rx, float ry) {
        if (rx <= 1e-5f || ry <= 1e-5f) return false;
        float dx = (px - cx) / rx;
        float dy = (py - cy) / ry;
        return (dx * dx + dy * dy) <= 1.0f;
    }

    @Test
    public void testSampleBilinearBoundariesAndEdgeSafety() {
        int width = 192, height = 192;
        float[] img = new float[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img[y * width + x] = y * 1000f + x;
            }
        }

        // 1. Exact corners
        assertEquals(0f, SparseLucasKanadeTracker.sampleBilinear(img, 0f, 0f, width, height), 1e-4f);
        float expectedTopRight = 191f;
        assertEquals(expectedTopRight, SparseLucasKanadeTracker.sampleBilinear(img, 191f, 0f, width, height), 1e-4f);
        float expectedBottomLeft = 191f * 1000f;
        assertEquals(expectedBottomLeft, SparseLucasKanadeTracker.sampleBilinear(img, 0f, 191f, width, height), 1e-4f);
        float expectedBottomRight = 191f * 1000f + 191f;
        assertEquals(expectedBottomRight, SparseLucasKanadeTracker.sampleBilinear(img, 191f, 191f, width, height), 1e-4f);

        // 2. Subpixel near boundary (no ArrayIndexOutOfBoundsException)
        float vNearEnd = SparseLucasKanadeTracker.sampleBilinear(img, 190.999f, 190.999f, width, height);
        assertTrue(Float.isFinite(vNearEnd));
        float vAtEnd = SparseLucasKanadeTracker.sampleBilinear(img, 191.0f, 191.0f, width, height);
        assertEquals(expectedBottomRight, vAtEnd, 1e-4f);
        float vPastEnd = SparseLucasKanadeTracker.sampleBilinear(img, 191.5f, 191.5f, width, height);
        assertEquals(expectedBottomRight, vPastEnd, 1e-4f);

        // 3. Negative and far out of bounds clamp safely
        assertEquals(0f, SparseLucasKanadeTracker.sampleBilinear(img, -100f, -50f, width, height), 1e-4f);
        assertEquals(expectedBottomRight, SparseLucasKanadeTracker.sampleBilinear(img, 500f, 500f, width, height), 1e-4f);

        // 4. NaN / Infinity safety
        assertEquals(0f, SparseLucasKanadeTracker.sampleBilinear(img, Float.NaN, Float.NaN, width, height), 1e-4f);
        assertEquals(expectedBottomRight, SparseLucasKanadeTracker.sampleBilinear(img, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, width, height), 1e-4f);
        assertEquals(0f, SparseLucasKanadeTracker.sampleBilinear(img, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, width, height), 1e-4f);

        // 5. Null / invalid buffer safety
        assertEquals(0f, SparseLucasKanadeTracker.sampleBilinear(null, 10f, 10f, width, height), 1e-4f);
        assertEquals(0f, SparseLucasKanadeTracker.sampleBilinear(new float[10], 10f, 10f, width, height), 1e-4f);
        assertEquals(0f, SparseLucasKanadeTracker.sampleBilinear(img, 10f, 10f, 0, height), 1e-4f);

        // 6. Minimal image sizes: 1x1, 2x2
        float[] single = new float[] { 42f };
        assertEquals(42f, SparseLucasKanadeTracker.sampleBilinear(single, 0f, 0f, 1, 1), 1e-4f);
        assertEquals(42f, SparseLucasKanadeTracker.sampleBilinear(single, 0.5f, 0.5f, 1, 1), 1e-4f);
        assertEquals(42f, SparseLucasKanadeTracker.sampleBilinear(single, 10f, 10f, 1, 1), 1e-4f);
    }

    @Test
    public void testGradientAndTrackingNearImageBoundaries() {
        SparseLucasKanadeTracker tracker = new SparseLucasKanadeTracker();
        int width = 192, height = 192;
        float[] prev = generateSyntheticFacePattern(96f, 96f, 16f);
        float[] curr = generateSyntheticFacePattern(98f, 96f, 16f);

        // Test points right at boundary edges: (0..3) and (188..191)
        float[] edgePtsX = new float[] { 0f, 1f, 2f, 3f, 188f, 189f, 190f, 191f };
        float[] edgePtsY = new float[] { 0f, 1f, 2f, 3f, 188f, 189f, 190f, 191f };

        // Should safely skip points without throwing ArrayIndexOutOfBoundsException
        SparseLucasKanadeTracker.FlowResult resEdge = tracker.track(prev, curr, width, height, edgePtsX, edgePtsY, 8);
        assertNotNull(resEdge);
        assertFalse("Points on window boundaries cannot be tracked and result should be invalid", resEdge.valid);

        // Test with NaN and Infinity points
        float[] nanPtsX = new float[] { Float.NaN, Float.POSITIVE_INFINITY, 96f, 96f };
        float[] nanPtsY = new float[] { Float.NaN, Float.NEGATIVE_INFINITY, 96f, 96f };
        SparseLucasKanadeTracker.FlowResult resNaN = tracker.track(prev, curr, width, height, nanPtsX, nanPtsY, 4);
        assertNotNull(resNaN);

        // Test boundary points at margin boundary (rx0 = 4, ry0 = 4) and (rx0 = 187, ry0 = 187)
        float[] marginPtsX = new float[] { 4f, 187f, 96f, 96f, 96f, 96f, 96f, 96f };
        float[] marginPtsY = new float[] { 4f, 187f, 96f, 96f, 96f, 96f, 96f, 96f };
        SparseLucasKanadeTracker.FlowResult resMargin = tracker.track(prev, curr, width, height, marginPtsX, marginPtsY, 8);
        assertNotNull(resMargin);
    }

    @Test
    public void testOpticalFlowMemoryAllocationAndThreadSafety() throws Exception {
        // 1. CameraState preallocated grayscale buffers to eliminate GC churn (30 fps * 192*192*4 bytes = 4.4MB/s)
        Class<?> csClass = Class.forName("com.makey.blurfaces.g2.Main$CameraState");
        java.lang.reflect.Constructor<?> csCtor = csClass.getDeclaredConstructor();
        csCtor.setAccessible(true);
        Object csInstance = csCtor.newInstance();

        java.lang.reflect.Field prevGrayField = csClass.getDeclaredField("prevFlowGray");
        prevGrayField.setAccessible(true);
        float[] prevGray = (float[]) prevGrayField.get(csInstance);
        assertNotNull("CameraState.prevFlowGray must be preallocated", prevGray);
        assertEquals(CleanFrameTap.SIZE * CleanFrameTap.SIZE, prevGray.length);

        java.lang.reflect.Field currGrayField = csClass.getDeclaredField("currFlowGray");
        currGrayField.setAccessible(true);
        float[] currGray = (float[]) currGrayField.get(csInstance);
        assertNotNull("CameraState.currFlowGray must be preallocated", currGray);
        assertEquals(CleanFrameTap.SIZE * CleanFrameTap.SIZE, currGray.length);

        // 2. SourceTracks preallocated point arrays to eliminate per-frame point allocations
        Class<?> stClass = Class.forName("com.makey.blurfaces.g2.Main$SourceTracks");
        java.lang.reflect.Field ptsXField = stClass.getDeclaredField("flowPtsX");
        ptsXField.setAccessible(true);
        java.lang.reflect.Field ptsYField = stClass.getDeclaredField("flowPtsY");
        ptsYField.setAccessible(true);

        Main.SourceTracks sourceTracks = new Main.SourceTracks();
        float[] flowPtsX = (float[]) ptsXField.get(sourceTracks);
        float[] flowPtsY = (float[]) ptsYField.get(sourceTracks);
        assertNotNull("SourceTracks.flowPtsX must be preallocated", flowPtsX);
        assertNotNull("SourceTracks.flowPtsY must be preallocated", flowPtsY);
        assertEquals(8, flowPtsX.length);
        assertEquals(8, flowPtsY.length);

        // 3. Thread safety: applyOpticalFlow must be synchronized on SourceTracks
        Method flowMethod = stClass.getDeclaredMethod("applyOpticalFlow",
                SparseLucasKanadeTracker.class, float[].class, float[].class, int.class, int.class, float.class);
        assertTrue("SourceTracks.applyOpticalFlow must be synchronized for thread safety",
                Modifier.isSynchronized(flowMethod.getModifiers()));
    }

    @Test
    public void testPrivacySelfTest() {
        for (int mode = 0; mode <= 2; mode++) {
            for (float scale : new float[]{0.65f, 0.82f, 1.0f}) {
                String result = Main.runPrivacySelfTest(scale, mode);
                assertNotNull(result);
                assertTrue("Self-test must pass for mode " + mode + " scale " + scale + " but was: " + result,
                        result.startsWith("PASSED:"));
                assertTrue(result.contains("Synthetic CPU mask contrast reduced by"));
                assertTrue(result.contains("detector, GPU and encoder not tested"));
                assertFalse(result.contains("detector score"));
            }
        }
        // Also test zero/default parameters
        String defaultResult = Main.runPrivacySelfTest(0.0f, 0);
        assertNotNull(defaultResult);
        assertTrue(defaultResult.startsWith("PASSED:"));

        String noArgResult = Main.runPrivacySelfTest();
        assertNotNull(noArgResult);
        assertTrue(noArgResult.startsWith("PASSED:"));
    }

    @Test
    public void testFirstFrameDetectionSynchronization() throws Exception {
        String source = "cam-first-frame-test-" + System.nanoTime();
        long now = System.nanoTime();

        // 1. Initially, source has no fresh result
        assertFalse("New source must not have fresh result initially", Main.hasFreshResult(source, now));

        // 2. Setting up first-frame latch
        CountDownLatch latch = new CountDownLatch(1);
        Main.FIRST_DETECTION_LATCH.put(source, latch);
        assertEquals(1L, latch.getCount());

        // 3. Simulate asynchronous background worker completing detection after 15ms
        AtomicBoolean renderCompleted = new AtomicBoolean(false);
        AtomicReference<Main.FaceGeometry> renderedGeometry = new AtomicReference<>();

        Main.SOURCE_ACTIVE_SINCE.put(source, now);

        Thread renderThread = new Thread(() -> {
            long threadNow = System.nanoTime();
            if (!Main.hasFreshResult(source, threadNow)) {
                Main.awaitFirstDetectionLatch(source);
                threadNow = System.nanoTime();
            }
            Main.FaceGeometry geom = Main.geometryFor(source, threadNow);
            renderedGeometry.set(geom);
            renderCompleted.set(true);
        }, "TestGLRenderThread");

        renderThread.start();

        // Simulate MediaPipe inference delay (~15ms)
        Thread.sleep(15);
        assertFalse("Render thread must be waiting on latch and not finished yet", renderCompleted.get());

        // Worker finishes inference and updates tracks
        Main.FaceGeometry detection = makeSingleDetection(0.45f, 0.55f, 0.08f, 0.10f, 0.92f, System.nanoTime());
        Main.updateTracks(source, detection);
        Main.releaseFirstDetectionLatch(source);

        // Wait for render thread to finish
        renderThread.join(1000);
        assertTrue("Render thread must complete after latch release", renderCompleted.get());
        assertNull("Latch must be removed from FIRST_DETECTION_LATCH after release", Main.FIRST_DETECTION_LATCH.get(source));

        // Verify that the very first frame obtained exact face geometry!
        Main.FaceGeometry firstFrameGeom = renderedGeometry.get();
        assertNotNull("First frame must obtain non-null FaceGeometry", firstFrameGeom);
        assertEquals("First frame must detect exactly 1 face", 1, firstFrameGeom.count);
        assertEquals(0.45f, firstFrameGeom.faces[0], 0.05f);
        assertEquals(0.55f, firstFrameGeom.faces[1], 0.05f);

        // 4. Test safe timeout on missing detection (must not hang, max ~40ms)
        String timeoutSource = "cam-timeout-test-" + System.nanoTime();
        CountDownLatch timeoutLatch = new CountDownLatch(1);
        Main.FIRST_DETECTION_LATCH.put(timeoutSource, timeoutLatch);

        long t0 = System.nanoTime();
        Main.awaitFirstDetectionLatch(timeoutSource);
        long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;
        assertTrue("Timeout must elapse around 40ms (was " + elapsedMs + "ms)", elapsedMs >= 35 && elapsedMs < 250);

        // 5. Test clearFirstDetectionLatches on unload releases any pending latches
        String pendingSource = "cam-pending-test-" + System.nanoTime();
        CountDownLatch pendingLatch = new CountDownLatch(1);
        Main.FIRST_DETECTION_LATCH.put(pendingSource, pendingLatch);
        assertEquals(1L, pendingLatch.getCount());

        Main.clearFirstDetectionLatches();
        assertEquals("clearFirstDetectionLatches must count down all latches", 0L, pendingLatch.getCount());
        assertTrue("FIRST_DETECTION_LATCH must be empty", Main.FIRST_DETECTION_LATCH.isEmpty());
    }

    private static double computeVariance(List<Float> values) {
        if (values.size() <= 1) return 0.0;
        double sum = 0.0;
        for (float v : values) sum += v;
        double mean = sum / values.size();
        double variance = 0.0;
        for (float v : values) {
            double diff = v - mean;
            variance += diff * diff;
        }
        return variance / (values.size() - 1);
    }
}
