package com.makey.blurfaces;

import java.nio.ByteBuffer;

final class NativeBridge {
    private NativeBridge() {}

    static native int init(String paramPath, String binPath);
    static native int process(ByteBuffer rgba, int width, int height, boolean detect);
    static native int getLastFaceCount();
    static native boolean wasLastProcessDetection();
    // A source switch must never inherit optical-flow points from the prior camera.
    static native void resetTracking();
    // Per face: box x/y/w/h plus five x/y facial anchors in upright UV space.
    static native void getFaceAnchors(float[] out);
    static native void cleanup();
}