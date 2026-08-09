package com.makey.blurfaces;

import android.os.SystemClock;

/**
 * Tracks a single face: Kalman prediction + One Euro smoothing.
 * Maintains position, size, and smooth animation between detections.
 */
public final class FaceTracker {
    private final FaceKalmanFilter kalman = new FaceKalmanFilter();
    private final OneEuroFilter xFilter;
    private final OneEuroFilter yFilter;
    private final OneEuroFilter wFilter;
    private final OneEuroFilter hFilter;
    
    private float smoothedX, smoothedY, smoothedW, smoothedH;
    private long lastDetectionTime;
    private boolean active = false;
    private int missedFrames = 0;
    private static final int MAX_MISSED_FRAMES = 30; // ~1 second at 30fps
    
    // Expansion factor for the blur box. The shader masks an ellipse inscribed
    // in this box, and an inscribed ellipse only covers about 78% of the box
    // area, so a box sized to the detection leaves the chin and forehead
    // outside the mask. Expanding well past the detection makes the ellipse
    // itself fully contain the face.
    private static final float EXPANSION = 1.5f;

    public FaceTracker() {
        // One Euro params: minCutoff, beta, dCutoff.
        // These are tuned for updates on EVERY frame (the optical-flow tracker
        // feeds us at ~30 Hz), not for the rare detector passes. With a 1 Hz
        // cutoff at dt=33ms the filter only moves ~17% toward the new position
        // per frame, which reads as the mask lagging behind the face. Position
        // gets a high cutoff plus strong beta so fast motion is followed
        // immediately; size stays slow because a breathing box looks worse than
        // a slightly stale one.
        xFilter = new OneEuroFilter(4.0f, 1.5f, 1.0f);
        yFilter = new OneEuroFilter(4.0f, 1.5f, 1.0f);
        wFilter = new OneEuroFilter(1.0f, 0.2f, 1.0f);
        hFilter = new OneEuroFilter(1.0f, 0.2f, 1.0f);
    }

    public synchronized void update(float x, float y, float w, float h) {
        long now = SystemClock.elapsedRealtime();
        float centerX = x + w / 2;
        float centerY = y + h / 2;
        
        kalman.update(centerX, centerY);
        
        // Apply One Euro smoothing
        smoothedX = xFilter.filter(x, now);
        smoothedY = yFilter.filter(y, now);
        smoothedW = wFilter.filter(w, now);
        smoothedH = hFilter.filter(h, now);
        
        lastDetectionTime = now;
        missedFrames = 0;
        active = true;
    }

    /**
     * Get predicted position for rendering.
     * @param aheadMs How far ahead to predict (typically frame time)
     * @return [x, y, w, h] expanded and predicted, or null if inactive
     */
    public synchronized float[] getPredictedRect(float aheadMs) {
        if (!active) return null;
        
        long now = SystemClock.elapsedRealtime();
        float age = now - lastDetectionTime;
        
        // Fade out if no detection for too long
        if (age > 1000) { // 1 second
            active = false;
            return null;
        }
        
        // Get Kalman prediction
        float[] pred = kalman.predict(aheadMs / 1000.0f);
        float predCenterX = pred[0];
        float predCenterY = pred[1];
        
        // Blend Kalman prediction with One Euro smoothed position
        // Kalman is better for prediction, One Euro for smoothness
        float blend = Math.min(1.0f, age / 100.0f); // trust Kalman more as detection ages
        float centerX = smoothedX + smoothedW/2 + (predCenterX - (smoothedX + smoothedW/2)) * blend;
        float centerY = smoothedY + smoothedH/2 + (predCenterY - (smoothedY + smoothedH/2)) * blend;
        
        // Reconstruct rect with expansion
        float w = smoothedW * EXPANSION;
        float h = smoothedH * EXPANSION;
        float x = centerX - w / 2;
        float y = centerY - h / 2;
        
        // Deliberately NOT clamped into the frame. Forcing the whole box inside
        // (min(1 - w, x)) slid the box sideways whenever the face approached an
        // edge, so the mask came off the face exactly when it mattered — and
        // with EXPANSION the box can be wider than the frame, which made that
        // clamp produce nonsense. The shader evaluates coverage per fragment,
        // so a box hanging off the edge is harmless.
        return new float[]{x, y, w, h};
    }

    public synchronized boolean isActive() {
        return active;
    }

    public synchronized void onMissedFrame() {
        missedFrames++;
        if (missedFrames > MAX_MISSED_FRAMES) {
            active = false;
        }
    }

    public synchronized void reset() {
        active = false;
        missedFrames = 0;
        kalman.reset();
        xFilter.reset();
        yFilter.reset();
        wFilter.reset();
        hFilter.reset();
    }
}