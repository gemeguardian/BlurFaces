package com.makey.blurfaces.g2;

import java.nio.ByteBuffer;

public final class NativeBridge {
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

    public static native void reset();

    public static native void cleanup();
}
