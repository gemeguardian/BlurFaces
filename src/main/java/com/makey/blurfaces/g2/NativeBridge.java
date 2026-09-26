package com.makey.blurfaces.g2;

import java.nio.ByteBuffer;

public final class NativeBridge {
    /**
     * {@link #process} status when {@link #reset()} was requested while that frame's
     * inference was running. The tracker was not advanced; drop the result.
     */
    public static final int STALE_AFTER_RESET = -6;

    private static volatile boolean loaded = false;

    public static synchronized void ensureLoaded(String soPath) {
        if (!loaded) {
            if (soPath != null) {
                System.load(soPath);
            } else {
                System.loadLibrary("blur_faces");
            }
            loaded = true;
        }
    }

    public static native int init(String paramPath, String binPath);

    public static native int process(ByteBuffer rgbaBuffer, int width, int height,
                                     float[] outGeometry, float[] outScores, float[] outYaws,
                                     int maxFaces, float minConfidence);

    // Same inference and tracker update; additionally returns a bounded diagnostic snapshot.
    public static native int processDebug(ByteBuffer rgbaBuffer, int width, int height,
                                          float[] outGeometry, float[] outScores, float[] outYaws,
                                          int maxFaces, float minConfidence, float[] debugSnapshot);

    /**
     * Requests a tracker reset. Lock-free and safe to call from the GL/UI thread: it
     * never waits for an in-flight inference; the worker applies it on its next frame.
     */
    public static native void reset();

    public static native void cleanup();
}
