package com.makey.blurfaces.g2;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES20;
import org.json.JSONObject;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** Standalone Android smoke test: synthetic pixels only, no camera or app data access. */
public final class DebugCaptureDeviceTest {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }

    private static Map<String, byte[]> unzip(byte[] archive) throws Exception {
        Map<String, byte[]> files = new HashMap<>();
        try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(archive))) {
            ZipEntry entry;
            byte[] buffer = new byte[4096];
            while ((entry = zip.getNextEntry()) != null) {
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                int n;
                while ((n = zip.read(buffer)) != -1) bytes.write(buffer, 0, n);
                files.put(entry.getName(), bytes.toByteArray());
            }
        }
        return files;
    }

    private static void testPbo() throws Exception {
        EGLDisplay display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        check(EGL14.eglInitialize(display, new int[2], 0, new int[2], 0), "EGL initialize");
        EGLConfig[] configs = new EGLConfig[1];
        int[] count = new int[1];
        check(EGL14.eglChooseConfig(display, new int[]{
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_RENDERABLE_TYPE, 0x40,
                EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_NONE}, 0, configs, 0, 1, count, 0)
                && count[0] > 0, "ES3 config");
        EGLContext context = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT,
                new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE}, 0);
        EGLSurface surface = EGL14.eglCreatePbufferSurface(display, configs[0], new int[]{
                EGL14.EGL_WIDTH, 192, EGL14.EGL_HEIGHT, 192, EGL14.EGL_NONE}, 0);
        check(EGL14.eglMakeCurrent(display, surface, surface, context), "ES3 pbuffer current");
        CleanFrameTap tap = new CleanFrameTap();
        try {
            java.lang.reflect.Method read = CleanFrameTap.class.getDeclaredMethod("readPixelsAsync", ByteBuffer.class);
            read.setAccessible(true);
            ByteBuffer pixels = ByteBuffer.allocateDirect(192 * 192 * 4);
            GLES20.glClearColor(1, 0, 0, 1); GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
            long redTime = (Long) read.invoke(tap, pixels);
            check((pixels.get(0) & 255) == 255 && pixels.get(1) == 0, "PBO warm-up direct red");
            GLES20.glClearColor(0, 1, 0, 1); GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
            long duplicateTime = (Long) read.invoke(tap, pixels);
            check(redTime == duplicateTime && (pixels.get(0) & 255) == 255,
                    "PBO warm-up duplicate retains original timestamp");
            GLES20.glClearColor(0, 0, 1, 1); GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
            long greenTime = (Long) read.invoke(tap, pixels);
            check(greenTime > redTime && (pixels.get(1) & 255) == 255 && pixels.get(2) == 0,
                    "PBO previous-frame pixels and timestamp stay paired");
            tap.resetPbo();
            long blueTime = (Long) read.invoke(tap, pixels);
            check(blueTime > greenTime && (pixels.get(2) & 255) == 255,
                    "camera reset flushes pending PBO contents");
            int[] fbo = new int[1];
            GLES20.glGenFramebuffers(1, fbo, 0);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0]); // deliberately incomplete
            try {
                read.invoke(tap, pixels);
                throw new AssertionError("failed PBO/direct read must not publish stale pixels");
            } catch (java.lang.reflect.InvocationTargetException expected) {
                check(expected.getCause() instanceof IllegalStateException, "GL failure propagated");
            } finally {
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
                GLES20.glDeleteFramebuffers(1, fbo, 0);
            }
        } finally {
            tap.release();
            EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            EGL14.eglDestroySurface(display, surface);
            EGL14.eglDestroyContext(display, context);
            EGL14.eglTerminate(display);
        }
        System.out.println("PASS: real ES3 PBO pixels/timestamps, warm-up duplicate, reset and failed-read rejection");
    }

    public static void main(String[] args) throws Exception {
        testPbo();
        Path root = Paths.get(args[0]);
        int size = 192;
        byte[] pixels = new byte[size * size * 4];
        for (int p = 0; p < pixels.length; p += 4) {
            pixels[p] = (byte) 180;
            pixels[p + 1] = (byte) 80;
            pixels[p + 2] = (byte) 40;
            pixels[p + 3] = (byte) 255;
        }
        ByteBuffer rgba = ByteBuffer.allocateDirect(pixels.length);
        rgba.put(pixels).position(0);
        float[] snapshot = new float[DebugCapture.SNAPSHOT_FLOATS];
        snapshot[0] = 1; snapshot[1] = 1;
        snapshot[2] = .35f; snapshot[3] = .19f; snapshot[4] = .4f;
        snapshot[5] = 100; snapshot[6] = 110;
        snapshot[7] = 1; snapshot[8] = 1; snapshot[9] = 1;
        snapshot[10] = .25f; snapshot[11] = .25f;
        snapshot[12] = .75f; snapshot[13] = .75f; snapshot[14] = .9f;
        int t = DebugCapture.TRACK_OFFSET;
        snapshot[t] = 7; snapshot[t+1] = 2; snapshot[t+2] = 3; snapshot[t+3] = .8f;
        snapshot[t+4] = .25f; snapshot[t+5] = .25f;
        snapshot[t+6] = .75f; snapshot[t+7] = .75f;
        float[] javaTracks = {.5f,.5f,.25f,0,0,.25f,.5f,.5f,25,10};
        byte[] archive = DebugCapture.encode(pixels, size, size, snapshot, 1, .35f,
                1_000_000_000L, 1_025_000_000L, 20_000_000L, "synthetic", 1000,
                "b".repeat(64), javaTracks, true, 0);
        Map<String, byte[]> entries = unzip(archive);
        check(entries.size() == 4, "raw + metadata + two JPEG crops");
        check(Arrays.equals(pixels, entries.get("frame.rgba")), "lossless RGBA round trip");
        JSONObject metadata = new JSONObject(new String(entries.get("metadata.json"), "UTF-8"));
        check(metadata.getBoolean("full_frame_saved"), "raw capture explicitly reported");
        check(metadata.getDouble("capture_to_result_ms") == 25, "PBO capture age");
        check(metadata.getJSONArray("java_tracks").length() == 1, "Java state serialized");
        check(metadata.getJSONArray("boxes").getJSONObject(1).getString("kind").equals("predicted_track"),
                "native coast distinguishable from detector observation");
        byte[] jpeg = entries.get("candidate_0.jpg");
        Bitmap crop = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        check(crop != null && crop.getWidth() == 144 && crop.getHeight() == 144, "real Android JPEG crop");
        int color = crop.getPixel(72, 72);
        check(Math.abs(((color >> 16) & 255) - 180) < 5 && Math.abs((color & 255) - 40) < 5,
                "RGBA channel order");
        crop.recycle();

        DebugCapture capture = new DebugCapture(root.resolve("captures"));
        check(!capture.store.enabled(System.nanoTime()), "off by default");
        capture.configure(true, "b".repeat(64));
        long token = capture.store.reserve(System.nanoTime());
        capture.submit(token, rgba, size, size, snapshot, 1, .35f,
                1_000_000_000L, 1_025_000_000L, 20_000_000L, "synthetic", javaTracks, true, 0);
        // The writer must own a copy, not the pooled buffer.
        for (int p = 0; p < rgba.capacity(); p++) rgba.put(p, (byte) 0);
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (!capture.store.status(System.nanoTime()).contains("saved=1") && System.nanoTime() < deadline)
            Thread.sleep(10);
        check(capture.store.status(System.nanoTime()).contains("saved=1"), "asynchronous atomic write");
        Path saved;
        try (java.util.stream.Stream<Path> paths = Files.list(root.resolve("captures"))) {
            saved = paths.findFirst().get();
        }
        check(Arrays.equals(pixels, unzip(Files.readAllBytes(saved)).get("frame.rgba")), "pooled buffer isolated");
        capture.stop();
        check(!capture.store.enabled(System.nanoTime()), "stop resets consent");
        check(capture.store.clear() == 1, "local erase");

        NativeBridge.ensureLoaded(root.resolve("libblur_faces.so").toString());
        String param = root.resolve("head_det.param").toString();
        String bin = root.resolve("head_det.bin").toString();
        for (int brightness : new int[]{0, 128, 255}) {
            for (int p = 0; p < pixels.length; p += 4) {
                pixels[p] = pixels[p+1] = pixels[p+2] = (byte) brightness;
                pixels[p+3] = (byte) 255;
            }
            rgba.clear(); rgba.put(pixels).position(0);
            check(NativeBridge.init(param, bin) == 0, "native model init");
            float[] geometry = new float[24], scores = new float[4], yaws = new float[4];
            int ordinary = NativeBridge.process(rgba, size, size, geometry, scores, yaws, 4, .35f);
            check(ordinary >= 0, "ordinary inference shape/status");
            NativeBridge.cleanup();
            check(NativeBridge.init(param, bin) == 0, "fresh debug model init");
            float[] debug = new float[DebugCapture.SNAPSHOT_FLOATS];
            float[] debugGeometry = new float[24], debugScores = new float[4];
            int diagnostic = NativeBridge.processDebug(rgba, size, size,
                    debugGeometry, debugScores, new float[4], 4, .35f, debug);
            check(diagnostic == ordinary, "debug must not change inference result");
            check(Arrays.equals(geometry, debugGeometry) && Arrays.equals(scores, debugScores), "debug geometry parity");
            check(DebugCapture.validSnapshot(debug), "native/Java snapshot ABI");
            check(NativeBridge.processDebug(rgba, size, size, debugGeometry, debugScores,
                    new float[4], 4, .35f, new float[1]) == -2, "short debug array rejected");
            NativeBridge.cleanup();
        }
        check(NativeBridge.processDebug(rgba, size, size, new float[24], new float[4],
                new float[4], 4, .35f, new float[DebugCapture.SNAPSHOT_FLOATS]) == -1, "uninitialized native error");
        System.out.println("PASS: Android JPEG/JSON/ZIP, async buffer isolation, local store, NCNN debug ABI/parity at 0/128/255 luminance");
    }
}
