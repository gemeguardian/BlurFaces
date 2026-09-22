package com.makey.blurfaces.g2;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import static org.junit.Assert.*;

public class DebugCaptureTest {
    @Rule public TemporaryFolder temp = new TemporaryFolder(new java.io.File("build"));
    private Path directory() { return temp.getRoot().toPath().resolve("captures"); }

    @Test public void consentRateExpiryAndGeneration() throws Exception {
        DebugCaptureStore store = new DebugCaptureStore(directory());
        assertEquals(0, store.reserve(1));
        assertFalse(Files.exists(directory()));
        store.configure(true, 100, 1000);
        long token = store.reserve(100);
        assertTrue(token > 0);
        assertEquals(0, store.reserve(100 + DebugCaptureStore.INTERVAL_NS)); // busy
        assertTrue(store.write(token, 101, new byte[]{1}));
        store.release();
        assertEquals(0, store.reserve(100 + DebugCaptureStore.INTERVAL_NS - 1));
        assertTrue(store.reserve(100 + DebugCaptureStore.INTERVAL_NS) > 0);
        store.release();
        assertFalse(store.enabled(100 + DebugCaptureStore.SESSION_NS));
        assertFalse(store.write(token, 100 + DebugCaptureStore.SESSION_NS, new byte[]{2}));
        assertTrue(store.status(100 + DebugCaptureStore.SESSION_NS).contains("session_expired"));
        store.configure(true, 400_000_000_000L, 1000);
        assertFalse(store.accepts(token, 400_000_000_000L));
        store.failure(token); // late writer failure from the old session
        assertTrue(store.enabled(400_000_000_000L));
        assertFalse(new DebugCaptureStore(directory()).enabled(400_000_000_000L));
        store.configure(false, 400_000_000_001L, 1000);
        assertEquals(0, store.reserve(400_000_000_001L));
    }

    @Test public void clearPreservesUnrelatedFilesAndSymlinks() throws Exception {
        DebugCaptureStore store = new DebugCaptureStore(directory());
        store.configure(true, 1, 1000);
        long token = store.reserve(1);
        assertTrue(store.write(token, 2, new byte[]{1, 2, 3}));
        store.release();
        Path keep = directory().resolve("keep.txt");
        Files.write(keep, new byte[]{7});
        Path link = directory().resolve("sample_00000000-0000-0000-0000-000000000000.zip");
        Files.createSymbolicLink(link, keep);
        assertEquals(1, store.clear());
        assertTrue(Files.exists(keep));
        assertTrue(Files.isSymbolicLink(link));
        assertFalse(store.write(token, 3, new byte[]{4}));
        assertFalse(store.enabled(3));
    }

    @Test public void byteAndFileLimitsStopRecording() throws Exception {
        DebugCaptureStore store = new DebugCaptureStore(directory());
        store.configure(true, 1, 1000);
        byte[] payload = new byte[DebugCaptureStore.MAX_RECORD_BYTES];
        for (int i = 0; i < 8; i++) {
            long now = 1 + i * DebugCaptureStore.INTERVAL_NS;
            assertTrue(store.write(store.reserve(now), now, payload));
            store.release();
        }
        long now = 1 + 8 * DebugCaptureStore.INTERVAL_NS;
        assertFalse(store.write(store.reserve(now), now, new byte[]{1}));
        store.release();
        assertFalse(store.enabled(now));
        assertTrue(store.status(now).contains("storage_limit"));
        assertEquals(8, store.clear());
        store.configure(true, 1, 1000);
        for (int i = 0; i < DebugCaptureStore.MAX_FILES; i++) {
            now = 1 + i * DebugCaptureStore.INTERVAL_NS;
            assertTrue(store.write(store.reserve(now), now, new byte[]{1}));
            store.release();
        }
        now += DebugCaptureStore.INTERVAL_NS;
        assertFalse(store.write(store.reserve(now), now, new byte[]{1}));
        assertFalse(store.enabled(now));
    }

    @Test public void retentionAndUnsafeDirectory() throws Exception {
        DebugCaptureStore store = new DebugCaptureStore(directory());
        store.configure(true, 1, 1000);
        assertTrue(store.write(store.reserve(1), 1, new byte[]{1}));
        store.release();
        Path capture;
        try (java.util.stream.Stream<Path> paths = Files.list(directory())) {
            capture = paths.findFirst().get();
        }
        Files.setLastModifiedTime(capture, FileTime.fromMillis(1));
        store.configure(false, 2, DebugCaptureStore.RETENTION_MS + 2);
        assertFalse(Files.exists(capture));
        Files.delete(directory());
        Files.createSymbolicLink(directory(), temp.getRoot().toPath());
        try {
            store.configure(true, 3, 1000);
            fail("Must reject symlink directory");
        } catch (java.io.IOException expected) {
            assertFalse(store.enabled(3));
        }
    }

    @Test public void snapshotAndCropBoundaries() {
        float[] snapshot = new float[DebugCapture.SNAPSHOT_FLOATS];
        assertFalse(DebugCapture.validSnapshot(snapshot));
        snapshot[0] = 1;
        assertTrue(DebugCapture.validSnapshot(snapshot));
        snapshot[8] = 65;
        assertFalse(DebugCapture.validSnapshot(snapshot));
        snapshot[8] = 0;
        snapshot[10] = Float.NaN;
        assertFalse(DebugCapture.validSnapshot(snapshot));
        assertArrayEquals(new int[]{0, 0, 2, 2}, DebugCapture.cropBounds(0, 0, 1, 1, 2, 2));
        assertNull(DebugCapture.cropBounds(.5f, 0, .4f, 1, 2, 2));
        assertNull(DebugCapture.cropBounds(Float.NaN, 0, 1, 1, 2, 2));
        byte[] pixels = {(byte)255,0,0,(byte)255, 0,(byte)255,0,(byte)255,
                         0,0,(byte)255,(byte)255, (byte)255,(byte)255,(byte)255,(byte)255};
        assertArrayEquals(new int[]{0xffff0000,0xff00ff00,0xff0000ff,0xffffffff},
                DebugCapture.cropArgb(pixels, 2, 2, new int[]{0,0,2,2}));
        assertArrayEquals(new int[]{0xff0000ff}, DebugCapture.cropArgb(pixels, 2, 2, new int[]{0,1,1,2}));
    }

    @Test public void javaDiagnosticSnapshotDoesNotAdvancePrediction() {
        Main.SourceTracks tracks = new Main.SourceTracks();
        float[] geometry = {.5f,.5f,.1f,0,0,.12f};
        tracks.update(new Main.FaceGeometry(geometry, new float[]{.8f}, new float[]{0}, 1,
                1_000_000_000L, "test"), 1_100_000_000L);
        Main.FaceTrack track = tracks.tracks[0];
        long predictionTime = track.lastPredictNanos;
        float[] before = track.drawCenter.clone();
        float[] state = tracks.debugSnapshot(1_200_000_000L);
        assertEquals(10, state.length);
        assertEquals(200f, state[8], 0f);
        assertEquals(100f, state[9], 0f);
        assertEquals(predictionTime, track.lastPredictNanos);
        assertArrayEquals(before, track.drawCenter, 0f);
        tracks.invalidate();
        assertEquals(0, tracks.debugSnapshot(1_300_000_000L).length);
    }
}
