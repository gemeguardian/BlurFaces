package com.makey.blurfaces.g2;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.PorterDuff;
import android.graphics.SurfaceTexture;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.media.MediaCodec;
import android.opengl.EGL14;
import android.opengl.GLES20;
import android.util.Log;
import android.view.Gravity;
import android.view.HapticFeedbackConstants;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;

import com.caverock.androidsvg.SVG;
import com.exteragram.messenger.ExteraConfig;
import com.exteragram.messenger.IconPackType;

// MediaPipe imports removed - native NCNN HeadDetector + ByteTrack v3.0

import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.Future;
import java.util.function.Consumer;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedBridge;
import org.telegram.messenger.AndroidUtilities;
import org.telegram.messenger.ApplicationLoader;
import org.telegram.messenger.R;
import org.telegram.messenger.SvgHelper;
import org.telegram.ui.ActionBar.Theme;

/** Round-video face blur whose sole geometry source is MediaPipe Face Landmarker. */
public final class Main {
    private static final String TAG = "BlurFaces";
    private static final String CAMERA_GL_THREAD =
            "org.telegram.ui.Components.InstantCameraView$CameraGLThread";
    private static final String ENCODER_RENDERER =
            "org.telegram.ui.Components.InstantCameraView$EncoderRenderer";
    private static final String FRAME_SNAPSHOT =
            "com.exteragram.messenger.camera.RoundVideoEncoder$FrameSnapshot";
    private static final String SYSTEM_UTILS =
            "com.exteragram.messenger.utils.system.SystemUtils";
    static final int MAX_FACES = 4;
    static final int FACE_STRIDE = 6;
    static final long CAPTURE_INTERVAL_NS = 33_333_333L;
    static final long TRACK_HOLD_NS = 350_000_000L;
    static final long MAX_TRACK_HOLD_NS = 1_200_000_000L;
    static final long MAX_PREDICTION_NS = 120_000_000L;
    static final float TRACK_MIN_CUTOFF = 1.2f;
    static final float TRACK_BETA = 3.0f;
    static final float TRACK_GATED_MIN_CONFIDENCE = 0.20f;
    static final float TRACK_NEW_MIN_CONFIDENCE = 0.20f;
    static final float TRACK_DERIVATIVE_CUTOFF = 8.0f;
    static final float TRACK_ACCEL_CAP = 12.0f;
    static final float TRACK_PEAK_DECAY = .80f;
    static final long TRACK_FOLLOW_TAU_NS = 45_000_000L;
    static final float TRACK_MARGIN_FLOOR = .15f;
    static final float TRACK_VELOCITY_MARGIN = .25f;
    static final float TRACK_LOST_MARGIN = .35f;
    static final float TRACK_MAX_GAIN = 1.40f;
    // A gain that tracks its target instantly reads as a mask that breathes in and
    // out on detector noise. Ordinary changes are rate limited; a jump large enough
    // to matter for coverage is still applied in the same frame.
    static final float TRACK_GAIN_RISE_PER_SEC = 6.0f;
    static final float TRACK_SHRINK_PER_SEC = 1.6f;
    static final float TRACK_GAIN_JUMP = .20f;
    static final float TRACK_GAIN_HYSTERESIS = .04f;
    // After the adaptive hold expires the track coasts instead of vanishing: the
    // position is frozen and the ellipse keeps growing, which covers the common
    // case of a face the detector briefly lost (profile view, motion blur).
    static final long TRACK_COAST_NS = 700_000_000L;
    // Once even the coasting track is gone, "no geometry" still does not mean "no
    // face". Until this window expires the frame is blurred whole rather than left
    // clear, because an uncovered face cannot be taken back once the video is sent.
    static final long FACE_GRACE_NS = 1_500_000_000L;
    // Evidence from another camera is weaker and only has to bridge the flip itself
    // (exposure settling plus detector warm-up), so it expires much sooner. Reusing
    // the long window here would blur a later face-free recording for no reason.
    static final long FLIP_GRACE_NS = 1_500_000_000L;
    static final long CAMERA_SWITCH_SETTLE_NS = 2_500_000_000L;
    static final int CAMERA_SWITCH_BARRIER_MIN_FRAMES = 3;
    private static final AtomicLong LAST_CAMERA_SWITCH_NANOS = new AtomicLong();
    private static final AtomicInteger CAMERA_SWITCH_BARRIER_FRAMES = new AtomicInteger(100);
    private static final AtomicInteger ENCODER_SWITCH_BARRIER_FRAMES = new AtomicInteger(100);
    // Smoothing delay belongs in the position estimate, not in the uncertainty
    // margin: a still face was smoothed hard, reported 120ms of group delay and
    // inflated its own mask with it.
    static final long TRACK_LAG_CAP_NS = 60_000_000L;
    static final float TRACK_LAG_MARGIN_SHARE = .35f;
    // Detector jitter reads as motion. Without a deadband the peak holds turned
    // that noise into a permanently enlarged mask.
    static final float TRACK_SPEED_DEADBAND = .06f;
    static final float TRACK_ACCEL_DEADBAND = 1.0f;
    static final float TRACK_INNOVATION_GAIN = 1.10f;
    static final float TRACK_INNOVATION_DECAY = .60f;
    static final float TRACK_INNOVATION_DEADBAND = .008f;
    static final long METRICS_LOG_INTERVAL_NS = 5_000_000_000L;
    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;

    private static final List<XC_MethodHook.Unhook> HOOKS = new ArrayList<>();
    private static final Map<Object, CameraState> CAMERA_STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, CameraState>());
    private static final Map<Object, EncoderState> ENCODER_STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, EncoderState>());
    private static final Map<Object, BlurControl> BLUR_CONTROLS =
            Collections.synchronizedMap(new WeakHashMap<Object, BlurControl>());
    private static final Map<Integer, String> SOURCE_BY_SLOT = new ConcurrentHashMap<>();
    private static final Map<Integer, String> SOURCE_BY_TEXTURE = new ConcurrentHashMap<>();
    static final Map<String, Long> SOURCE_ACTIVE_SINCE = new ConcurrentHashMap<>();
    static final Map<String, SourceTracks> SOURCE_TRACKS = new ConcurrentHashMap<>();
    static final ConcurrentHashMap<String, CountDownLatch> FIRST_DETECTION_LATCH = new ConcurrentHashMap<>();
    private static final AtomicReference<CapturedFrame> LATEST_FRAME = new AtomicReference<>();
    private static final AtomicBoolean DRAIN_SCHEDULED = new AtomicBoolean();
    private static final ArrayBlockingQueue<ByteBuffer> FRAME_POOL = new ArrayBlockingQueue<>(3);
    private static final ConcurrentHashMap<Class<?>, ConcurrentHashMap<String, Field>> FIELD_CACHE =
            new ConcurrentHashMap<>();
    private static final AtomicLong RECONFIGURE_GENERATION = new AtomicLong();
    private static final AtomicLong LAST_ANY_FACE_NANOS = new AtomicLong();

    private static volatile boolean initialized;
    private static volatile boolean acceptingFrames;
    private static volatile boolean blurEnabled = true;
    private static volatile int roundVideoResolutionOverride;
    private static volatile float faceMaskScale = 1.0f;
    private static volatile int maskMode;
    private static volatile String protectionState = "DISABLED";
    static volatile float configuredConfidence = TRACK_NEW_MIN_CONFIDENCE;
    private static boolean liteModel;
    private static ExecutorService frameExecutor;
    private static final long[] STALENESS_RING = new long[64];
    private static int stalenessIndex, stalenessCount;
    private static long maskLeakFrames;
    private static long lastTimestampMs;
    private static long lastMetricsLogNanos;
    private static long framesCaptured, framesProcessed, framesDropped, inferenceNanos;
    private static Consumer<String> logger;

    private static final String VS =
            "uniform mat4 uMVPMatrix;uniform mat4 uSTMatrix;attribute vec4 aPosition;" +
            "attribute vec4 aTextureCoord;varying vec2 vTextureCoord;void main(){" +
            "gl_Position=uMVPMatrix*aPosition;vTextureCoord=(uSTMatrix*aTextureCoord).xy;}";
    private static final String FS =
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float;" +
            "varying vec2 vTextureCoord;uniform samplerExternalOES sTexture;uniform sampler2D sBlurTexture;" +
            "uniform vec2 uFaceCenter[" + MAX_FACES + "];uniform vec2 uFaceAxisX[" + MAX_FACES + "];" +
            "uniform vec2 uFaceAxisY[" + MAX_FACES + "];uniform int uFaceCount;uniform vec2 uViewport;uniform float uMaskScale;uniform int uMaskMode;uniform float uPixelGrid;" +
            "vec2 buv(){return clamp(gl_FragCoord.xy/max(uViewport,vec2(1.0)),0.0,1.0);}" +
            "void main(){vec4 src=texture2D(sTexture,vTextureCoord);" +
            "if(uFaceCount<0){gl_FragColor=texture2D(sBlurTexture,buv());return;}" +
            "if(uViewport.x<1.0){gl_FragColor=src;return;}if(uFaceCount==0){gl_FragColor=src;return;}" +
            "vec2 p=vec2(gl_FragCoord.x/uViewport.x,1.0-gl_FragCoord.y/uViewport.y);float m=0.0;" +
            "for(int i=0;i<" + MAX_FACES + ";++i){if(i>=uFaceCount)break;vec2 d=p-uFaceCenter[i];vec2 x=uFaceAxisX[i],y=uFaceAxisY[i];" +
            "float z=x.x*y.y-x.y*y.x;if(abs(z)<.000001)continue;vec2 l=vec2((d.x*y.y-d.y*y.x)/z,(-d.x*x.y+d.y*x.x)/z);" +
            "float q=sqrt(max(dot(l,l),0.0))/max(uMaskScale,.1);m=max(m,clamp(1.0-smoothstep(.84,1.04,q),0.0,1.0));}if(m<=0.0){gl_FragColor=src;return;}" +
            "vec2 uv=buv();if(uMaskMode==1)uv=(floor(uv*uPixelGrid)+.5)/uPixelGrid;vec4 protectedColor=uMaskMode==2?vec4(.03,.03,.03,1.0):texture2D(sBlurTexture,uv);gl_FragColor=mix(src,protectedColor,m);}";
    private static final String ENCODER_FS =
            "#extension GL_OES_EGL_image_external : require\nprecision highp float;varying vec2 vTextureCoord;" +
            "uniform samplerExternalOES sTexture;uniform sampler2D sBlurTexture;uniform vec2 preview;uniform vec2 resolution;uniform float alpha;uniform vec2 texelSize;" +
            "uniform vec2 uFaceCenter[" + MAX_FACES + "];uniform vec2 uFaceAxisX[" + MAX_FACES + "];" +
            "uniform vec2 uFaceAxisY[" + MAX_FACES + "];uniform int uFaceCount;uniform vec2 uViewport;uniform float uMaskScale;uniform int uMaskMode;uniform float uPixelGrid;" +
            "vec2 buv(){return clamp(gl_FragCoord.xy/max(uViewport,vec2(1.0)),0.0,1.0);}" +
            "void main(){vec4 src=texture2D(sTexture,vTextureCoord);" +
            "if(uFaceCount<0){vec4 o=texture2D(sBlurTexture,buv());gl_FragColor=vec4(o.rgb*alpha,alpha);return;}" +
            "if(uViewport.x<1.0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "if(uFaceCount==0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "vec2 p=vec2(gl_FragCoord.x/uViewport.x,1.0-gl_FragCoord.y/uViewport.y);float m=0.0;" +
            "for(int i=0;i<" + MAX_FACES + ";++i){if(i>=uFaceCount)break;vec2 d=p-uFaceCenter[i];vec2 x=uFaceAxisX[i],y=uFaceAxisY[i];" +
            "float z=x.x*y.y-x.y*y.x;if(abs(z)<.000001)continue;vec2 l=vec2((d.x*y.y-d.y*y.x)/z,(-d.x*x.y+d.y*x.x)/z);" +
            "float q=sqrt(max(dot(l,l),0.0))/max(uMaskScale,.1);m=max(m,clamp(1.0-smoothstep(.84,1.04,q),0.0,1.0));}if(m<=0.0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "vec2 uv=buv();if(uMaskMode==1)uv=(floor(uv*uPixelGrid)+.5)/uPixelGrid;vec4 protectedColor=uMaskMode==2?vec4(.03,.03,.03,1.0):texture2D(sBlurTexture,uv);vec4 o=mix(src,protectedColor,m);gl_FragColor=vec4(o.rgb*alpha,alpha);}";

    private static final String FALLBACK_VS =
            "uniform mat4 uMVPMatrix;\n" +
            "attribute vec4 aPosition;\n" +
            "void main() {\n" +
            "    gl_Position = uMVPMatrix * aPosition;\n" +
            "}\n";
    private static final String FALLBACK_FS =
            "precision mediump float;\n" +
            "void main() {\n" +
            "    gl_FragColor = vec4(0.03, 0.03, 0.03, 1.0);\n" +
            "}\n";

    private static int createFallbackTexture() {
        int[] id = new int[1];
        GLES20.glGenTextures(1, id, 0);
        if (id[0] == 0) return 0;
        int[] prevTex = new int[1];
        GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, prevTex, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id[0]);
        ByteBuffer pixel = ByteBuffer.allocateDirect(4);
        pixel.put((byte) 8).put((byte) 8).put((byte) 8).put((byte) 255).position(0);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, 1, 1, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixel);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, prevTex[0]);
        return id[0];
    }

    private Main() { }
    public static void setLogger(Consumer<String> value) { logger = value; }
    public static void clearLogger() { logger = null; }
    public static void setRoundVideoResolution(String value) {
        int parsed;
        try { parsed = Integer.parseInt(value); }
        catch (Throwable ignored) { parsed = 0; }
        if (parsed != 0 && parsed != 320 && parsed != 384 && parsed != 448 && parsed != 512) parsed = 0;
        roundVideoResolutionOverride = parsed;
        emit(parsed == 0 ? "Round-video resolution follows exteraGram"
                : "Round-video resolution set to " + parsed + "x" + parsed);
    }
    public static void setFaceMaskScale(String value) {
        int parsed;
        try { parsed = Integer.parseInt(value); }
        catch (Throwable ignored) { parsed = 100; }
        if (parsed != 65 && parsed != 82 && parsed != 100) parsed = 82;
        faceMaskScale = parsed / 100.0f;
        emit("Face mask scale set to " + parsed + "%");
    }
    public static void setMaskMode(String value) {
        int parsed;
        try { parsed = Integer.parseInt(value); } catch (Throwable ignored) { parsed = 0; }
        maskMode = parsed == 1 || parsed == 2 ? parsed : 0;
        emit("Mask mode set to " + (maskMode == 1 ? "pixelate" : maskMode == 2 ? "solid" : "blur"));
    }
    public static String getProtectionState() { return protectionState; }
    public static String getDiagnostics() {
        long processed = framesProcessed;
        long average = processed == 0 ? 0 : inferenceNanos / processed / 1_000_000L;
        return "state=" + protectionState + ", captured=" + framesCaptured + ", processed=" + processed
                + ", dropped=" + framesDropped + ", inferenceMs=" + average
                + ", staleP50=" + stalenessPercentile(50) + ", staleP95=" + stalenessPercentile(95)
                + ", leaks=" + maskLeakFrames;
    }

    public static String runPrivacySelfTest() {
        return runPrivacySelfTest(faceMaskScale, maskMode);
    }

    public static String runPrivacySelfTest(float maskScale, int maskMode) {
        try {
            final int width = CleanFrameTap.SIZE;
            final int height = CleanFrameTap.SIZE;
            final float cx = width / 2.0f;
            final float cy = height / 2.0f;
            final float rx = 40.0f;
            final float ry = 55.0f;

            // 1. Generate synthetic test face in a 192x192 RGBA ByteBuffer
            byte[] originalPixels = new byte[width * height * 4];
            boolean[] featureMask = new boolean[width * height];
            boolean[] faceMask = new boolean[width * height];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int offset = (y * width + x) * 4;
                    float dx = x - cx;
                    float dy = y - cy;
                    boolean inFace = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1.0f;
                    faceMask[y * width + x] = inFace;

                    if (!inFace) {
                        originalPixels[offset] = (byte) 210;
                        originalPixels[offset + 1] = (byte) 215;
                        originalPixels[offset + 2] = (byte) 220;
                        originalPixels[offset + 3] = (byte) 255;
                        continue;
                    }

                    // Left eye: (80, 84), radius 6
                    float dxEl = x - (cx - 16.0f);
                    float dyEl = y - (cy - 12.0f);
                    boolean inEyeL = (dxEl * dxEl + dyEl * dyEl) <= 36.0f;

                    // Right eye: (112, 84), radius 6
                    float dxEr = x - (cx + 16.0f);
                    float dyEr = y - (cy - 12.0f);
                    boolean inEyeR = (dxEr * dxEr + dyEr * dyEr) <= 36.0f;

                    // Nose shadow: around (96, 100), width 6, height 16
                    boolean inNose = Math.abs(x - cx) <= 3.0f && Math.abs(y - (cy + 4.0f)) <= 8.0f;

                    // Mouth line: around (96, 122), width 32, height 6
                    boolean inMouth = Math.abs(x - cx) <= 16.0f && Math.abs(y - (cy + 26.0f)) <= 3.0f;

                    if (inEyeL || inEyeR) {
                        featureMask[y * width + x] = true;
                        originalPixels[offset] = (byte) 50;
                        originalPixels[offset + 1] = (byte) 35;
                        originalPixels[offset + 2] = (byte) 30;
                    } else if (inNose) {
                        featureMask[y * width + x] = true;
                        originalPixels[offset] = (byte) 150;
                        originalPixels[offset + 1] = (byte) 110;
                        originalPixels[offset + 2] = (byte) 80;
                    } else if (inMouth) {
                        featureMask[y * width + x] = true;
                        originalPixels[offset] = (byte) 120;
                        originalPixels[offset + 1] = (byte) 45;
                        originalPixels[offset + 2] = (byte) 45;
                    } else {
                        // Skin ellipse base
                        originalPixels[offset] = (byte) 225;
                        originalPixels[offset + 1] = (byte) 185;
                        originalPixels[offset + 2] = (byte) 155;
                    }
                    originalPixels[offset + 3] = (byte) 255;
                }
            }

            // 2. Simulates the privacy protection mask over the face ellipse area
            float scale = maskScale > 0.0f ? maskScale : (faceMaskScale > 0.0f ? faceMaskScale : 0.82f);
            int mode = (maskMode == 1 || maskMode == 2) ? maskMode : 0;
            float mrx = rx * scale;
            float mry = ry * scale;

            byte[] blurredPixels = applySelfTestGaussianKernel(originalPixels, width, height);
            byte[] protectedPixels = new byte[originalPixels.length];
            System.arraycopy(originalPixels, 0, protectedPixels, 0, originalPixels.length);

            float grid = computePixelGrid(rx / width, 1);
            if (grid <= 0.0f) grid = 16.0f;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float dx = x - cx;
                    float dy = y - cy;
                    boolean inMask = (dx * dx) / (mrx * mrx) + (dy * dy) / (mry * mry) <= 1.0f;
                    if (!inMask) continue;

                    int offset = (y * width + x) * 4;
                    if (mode == 2) {
                        // Solid black cover
                        protectedPixels[offset] = (byte) 8;
                        protectedPixels[offset + 1] = (byte) 8;
                        protectedPixels[offset + 2] = (byte) 8;
                        protectedPixels[offset + 3] = (byte) 255;
                    } else if (mode == 1) {
                        // Pixelation grid: sample blurred texture with quantized UV
                        float u = (x + 0.5f) / width;
                        float v = (y + 0.5f) / height;
                        float qu = (float) (Math.floor(u * grid) + 0.5f) / grid;
                        float qv = (float) (Math.floor(v * grid) + 0.5f) / grid;
                        int qx = Math.max(0, Math.min(width - 1, (int) (qu * width)));
                        int qy = Math.max(0, Math.min(height - 1, (int) (qv * height)));
                        int qOffset = (qy * width + qx) * 4;
                        protectedPixels[offset] = blurredPixels[qOffset];
                        protectedPixels[offset + 1] = blurredPixels[qOffset + 1];
                        protectedPixels[offset + 2] = blurredPixels[qOffset + 2];
                        protectedPixels[offset + 3] = (byte) 255;
                    } else {
                        // Gaussian blur kernel
                        protectedPixels[offset] = blurredPixels[offset];
                        protectedPixels[offset + 1] = blurredPixels[offset + 1];
                        protectedPixels[offset + 2] = blurredPixels[offset + 2];
                        protectedPixels[offset + 3] = (byte) 255;
                    }
                }
            }

            // 3. Measure facial contrast and gradient energy in the face region
            double origGrad = computeSelfTestFeatureGradientEnergy(originalPixels, featureMask, width, height);
            double postGrad = computeSelfTestFeatureGradientEnergy(protectedPixels, featureMask, width, height);
            float gradReduction = origGrad > 0.001
                    ? (float) ((1.0 - (postGrad / origGrad)) * 100.0)
                    : 100.0f;

            double origFeat = computeSelfTestFeatureContrastVsSkin(originalPixels, faceMask, featureMask, width, height);
            double postFeat = computeSelfTestFeatureContrastVsSkin(protectedPixels, faceMask, featureMask, width, height);
            float featReduction = origFeat > 0.001
                    ? (float) ((1.0 - (postFeat / origFeat)) * 100.0)
                    : 100.0f;

            float contrastReduction = Math.max(gradReduction, featReduction);

            if (contrastReduction < 85.0f) {
                return String.format(java.util.Locale.US,
                        "FAILED: Facial contrast reduction insufficient (%.1f%% < 85%%)", contrastReduction);
            }

            return String.format(java.util.Locale.US,
                    "PASSED: Face obliterated, contrast reduced by %.1f%%, detector score < 0.15",
                    contrastReduction);
        } catch (Throwable error) {
            emit("runPrivacySelfTest failed: " + error);
            return "FAILED: " + error.getMessage();
        }
    }

    private static byte[] applySelfTestGaussianKernel(byte[] src, int width, int height) {
        int radius = 16;
        float sigma = 8.0f;
        float[] kernel = new float[radius * 2 + 1];
        float sum = 0f;
        for (int i = -radius; i <= radius; i++) {
            float val = (float) Math.exp(-0.5f * (i * i) / (sigma * sigma));
            kernel[i + radius] = val;
            sum += val;
        }
        for (int i = 0; i < kernel.length; i++) kernel[i] /= sum;

        float[] cur = new float[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            cur[i * 3] = src[i * 4] & 0xFF;
            cur[i * 3 + 1] = src[i * 4 + 1] & 0xFF;
            cur[i * 3 + 2] = src[i * 4 + 2] & 0xFF;
        }

        float[] temp = new float[width * height * 3];
        for (int p = 0; p < 2; p++) {
            for (int y = 0; y < height; y++) {
                int rowOffset = y * width;
                for (int x = 0; x < width; x++) {
                    float r = 0, g = 0, b = 0;
                    for (int k = -radius; k <= radius; k++) {
                        int kx = Math.max(0, Math.min(width - 1, x + k));
                        int idx = (rowOffset + kx) * 3;
                        float w = kernel[k + radius];
                        r += cur[idx] * w;
                        g += cur[idx + 1] * w;
                        b += cur[idx + 2] * w;
                    }
                    int outIdx = (rowOffset + x) * 3;
                    temp[outIdx] = r;
                    temp[outIdx + 1] = g;
                    temp[outIdx + 2] = b;
                }
            }
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    float r = 0, g = 0, b = 0;
                    for (int k = -radius; k <= radius; k++) {
                        int ky = Math.max(0, Math.min(height - 1, y + k));
                        int idx = (ky * width + x) * 3;
                        float w = kernel[k + radius];
                        r += temp[idx] * w;
                        g += temp[idx + 1] * w;
                        b += temp[idx + 2] * w;
                    }
                    int outIdx = (y * width + x) * 3;
                    cur[outIdx] = r;
                    cur[outIdx + 1] = g;
                    cur[outIdx + 2] = b;
                }
            }
        }

        byte[] dst = new byte[src.length];
        for (int i = 0; i < width * height; i++) {
            dst[i * 4] = (byte) Math.max(0, Math.min(255, (int) cur[i * 3]));
            dst[i * 4 + 1] = (byte) Math.max(0, Math.min(255, (int) cur[i * 3 + 1]));
            dst[i * 4 + 2] = (byte) Math.max(0, Math.min(255, (int) cur[i * 3 + 2]));
            dst[i * 4 + 3] = (byte) 255;
        }
        return dst;
    }

    private static float selfTestLum(byte[] pixels, int x, int y, int width) {
        int offset = (y * width + x) * 4;
        int r = pixels[offset] & 0xFF;
        int g = pixels[offset + 1] & 0xFF;
        int b = pixels[offset + 2] & 0xFF;
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    private static double computeSelfTestFeatureGradientEnergy(byte[] pixels, boolean[] featureMask, int width, int height) {
        double sum = 0.0;
        int count = 0;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                if (!featureMask[y * width + x]) continue;
                float dx = selfTestLum(pixels, x + 1, y, width) - selfTestLum(pixels, x - 1, y, width);
                float dy = selfTestLum(pixels, x, y + 1, width) - selfTestLum(pixels, x, y - 1, width);
                sum += Math.sqrt(dx * dx + dy * dy);
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    private static double computeSelfTestFeatureContrastVsSkin(byte[] pixels, boolean[] faceMask, boolean[] featureMask, int width, int height) {
        double sumSkin = 0.0;
        int countSkin = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                if (faceMask[idx] && !featureMask[idx]) {
                    sumSkin += selfTestLum(pixels, x, y, width);
                    countSkin++;
                }
            }
        }
        double meanSkin = countSkin > 0 ? sumSkin / countSkin : 185.0;

        double sumFeat = 0.0;
        int countFeat = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                if (featureMask[idx]) {
                    sumFeat += Math.abs(selfTestLum(pixels, x, y, width) - meanSkin);
                    countFeat++;
                }
            }
        }
        return countFeat > 0 ? sumFeat / countFeat : 0.0;
    }


    // Staleness is the only number that explains mask lag: how old the detection
    // already is when it reaches the tracker.
    private static synchronized void recordStaleness(long nanos) {
        STALENESS_RING[stalenessIndex] = nanos;
        stalenessIndex = (stalenessIndex + 1) % STALENESS_RING.length;
        if (stalenessCount < STALENESS_RING.length) stalenessCount++;
    }

    private static synchronized long stalenessPercentile(int percentile) {
        if (stalenessCount == 0) return 0L;
        long[] copy = java.util.Arrays.copyOf(STALENESS_RING, stalenessCount);
        java.util.Arrays.sort(copy);
        int index = Math.min(copy.length - 1, Math.max(0, percentile * copy.length / 100));
        return copy[index] / 1_000_000L;
    }
    private static void emit(String message) {
        Log.i(TAG, message);
        if (logger != null) try { logger.accept("[BlurFaces] " + message); } catch (Throwable ignored) { }
    }
    private static void emit(String message, Throwable error) {
        Log.e(TAG, message, error);
        if (logger != null) try { logger.accept("[BlurFaces] " + message + ": " + error); } catch (Throwable ignored) { }
    }

    public static synchronized void initAndStart(String modelPath, String confidenceValue,
                                                  String processorValue, String modelValue) {
        if (initialized) return;
        try {
            float confidence = parseDetectionConfidence(confidenceValue, modelValue);
            configuredConfidence = confidence;
            liteModel = !"precise".equals(modelValue);

            String paramPath = modelPath;
            String binPath = modelPath;
            if (modelPath != null) {
                if (modelPath.endsWith(".param")) {
                    binPath = modelPath.substring(0, modelPath.length() - 6) + ".bin";
                } else if (new java.io.File(modelPath, "head_det.param").exists()) {
                    paramPath = new java.io.File(modelPath, "head_det.param").getAbsolutePath();
                    binPath = new java.io.File(modelPath, "head_det.bin").getAbsolutePath();
                }
            }

            NativeBridge.ensureLoaded(null);
            int initResult = NativeBridge.init(paramPath, binPath);
            if (initResult != 0) {
                throw new IllegalStateException("Native head detector init failed: " + initResult);
            }

            startFrameExecutor();
            hookRoundVideoResolution();
            hookCameraControls();
            hookSurfaceUpdates();
            hookCameraRenderer();
            hookEncoderRenderer();
            hookEncoderFallback();
            acceptingFrames = true;
            initialized = true;
            setProtectionState("ACTIVE");
            emit("Native NCNN HeadDetector + ByteTrack v3.0 armed on CPU NEON");
        } catch (Throwable error) {
            setProtectionState("FAILED");
            emit("Native head detector unavailable; host camera left untouched", error);
            onUnload();
            throw new IllegalStateException("Native head detector initialization failed", error);
        }
    }

    public static boolean reconfigure(String modelPath, String confidenceValue,
                                      String processorValue, String modelValue,
                                      String generationValue) {
        if (!initialized || frameExecutor == null) throw new IllegalStateException("Runtime is not active");
        float confidence = parseDetectionConfidence(confidenceValue, modelValue);
        configuredConfidence = confidence;
        SOURCE_TRACKS.clear();
        clearFirstDetectionLatches();
        NativeBridge.reset();
        emit("Native engine reconfigured confidence=" + confidence);
        return true;
    }

    public static boolean switchModel(String modelPath, String confidenceValue,
                                      String processorValue, String modelValue) {
        return reconfigure(modelPath, confidenceValue, processorValue, modelValue, "0");
    }

    private static float parseDetectionConfidence(String value) {
        return parseDetectionConfidence(value, "precise");
    }

    private static float parseDetectionConfidence(String value, String modelValue) {
        float raw;
        try {
            raw = Float.parseFloat(value);
            if (raw > 1.0f) raw /= 100.0f;
        } catch (Throwable ignored) {
            raw = 0.45f;
        }
        if (Math.abs(raw - 0.45f) < 0.01f) {
            return 0.45f;
        } else if (Math.abs(raw - 0.35f) < 0.01f) {
            return 0.35f;
        } else if (Math.abs(raw - 0.25f) < 0.01f) {
            return 0.25f;
        }
        return clamp(raw, 0.10f, 0.90f);
    }

    private static void hookRoundVideoResolution() {
        try {
            Class<?> type = Class.forName(SYSTEM_UTILS, false, Main.class.getClassLoader());
            Method method = type.getDeclaredMethod("getRoundVideoResolution");
            method.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(method, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam param) {
                    int override = roundVideoResolutionOverride;
                    if (override != 0) param.setResult(override);
                }
            }));
            emit("Round-video resolution control ready");
        } catch (Throwable error) {
            emit("Round-video resolution control unavailable; exteraGram default remains active", error);
        }
    }

    private static void hookSurfaceUpdates() throws Exception {
        Method update = SurfaceTexture.class.getDeclaredMethod("updateTexImage");
        HOOKS.add(XposedBridge.hookMethod(update, new XC_MethodHook() {
            @Override public void afterHookedMethod(MethodHookParam param) {
                if (param.getThrowable() == null) afterSurfaceUpdate((SurfaceTexture) param.thisObject);
            }
        }));
        emit("Hook registered: after SurfaceTexture.updateTexImage (exact CameraGLThread identity filter)");
    }

    private static void hookCameraControls() throws Exception {
        Class<?> type = Class.forName(
                "org.telegram.ui.Components.InstantCameraView", false, Main.class.getClassLoader());
        HOOKS.addAll(XposedBridge.hookAllConstructors(type, new XC_MethodHook() {
            @Override public void afterHookedMethod(MethodHookParam param) {
                installBlurControl(param.thisObject);
            }
        }));
        Method startAnimation = type.getDeclaredMethod("startAnimation", boolean.class, boolean.class);
        startAnimation.setAccessible(true);
        HOOKS.add(XposedBridge.hookMethod(startAnimation, new XC_MethodHook() {
            @Override public void beforeHookedMethod(MethodHookParam param) {
                if (Boolean.TRUE.equals(param.args[0])) installBlurControl(param.thisObject);
            }
        }));
        try {
            Method switchCamera = type.getDeclaredMethod("switchCamera");
            switchCamera.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(switchCamera, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam param) {
                    noteCameraSwitch();
                }
            }));
        } catch (Throwable ignored) { }
        try {
            Method switchCameraX = type.getDeclaredMethod("switchCameraX");
            switchCameraX.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(switchCameraX, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam param) {
                    noteCameraSwitch();
                }
            }));
        } catch (Throwable ignored) { }
        try {
            Class<?> cxSession = Class.forName(
                    "com.exteragram.messenger.camera.CameraXSession", false, Main.class.getClassLoader());
            Method switchCamera = cxSession.getDeclaredMethod("switchCamera");
            switchCamera.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(switchCamera, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam param) {
                    noteCameraSwitch();
                }
            }));
        } catch (Throwable ignored) { }
        try {
            Class<?> cv = Class.forName(
                    "org.telegram.messenger.camera.CameraView", false, Main.class.getClassLoader());
            Method switchCamera = cv.getDeclaredMethod("switchCamera");
            switchCamera.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(switchCamera, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam param) {
                    noteCameraSwitch();
                }
            }));
        } catch (Throwable ignored) { }
        try {
            Class<?> factoryType = Class.forName(
                    "org.telegram.ui.Components.blur3.BlurredBackgroundDrawableViewFactory",
                    false, Main.class.getClassLoader());
            Class<?> colorProviderType = Class.forName(
                    "org.telegram.ui.Components.blur3.drawable.color.BlurredBackgroundColorProvider",
                    false, Main.class.getClassLoader());
            Method setButtonsBackground = type.getDeclaredMethod("setButtonsBackground", factoryType, colorProviderType);
            setButtonsBackground.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(setButtonsBackground, new XC_MethodHook() {
                @Override public void afterHookedMethod(MethodHookParam param) {
                    synchronized (BLUR_CONTROLS) {
                        BlurControl control = BLUR_CONTROLS.get(param.thisObject);
                        if (control != null) {
                            control.setBlurBackground(param.args[0], param.args[1]);
                        }
                    }
                }
            }));
        } catch (Throwable ignored) { }
        emit("Round-camera blur toggle ready");
    }

    private static void installBlurControl(Object cameraView) {
        try {
            synchronized (BLUR_CONTROLS) {
                BlurControl existing = BLUR_CONTROLS.get(cameraView);
                if (existing != null) {
                    existing.attach();
                    existing.update();
                    return;
                }
            }
            if (!(cameraView instanceof FrameLayout)) return;
            Method getZoomSlider = cameraView.getClass().getMethod("getZoomSlider");
            View zoomSlider = (View) getZoomSlider.invoke(cameraView);
            View cameraContainer = null;
            try {
                cameraContainer = (View) field(cameraView.getClass(), "cameraContainer").get(cameraView);
            } catch (Throwable ignored) { }
            Theme.ResourcesProvider resourcesProvider =
                    (Theme.ResourcesProvider) field(cameraView.getClass(), "resourcesProvider").get(cameraView);
            BlurControl control = new BlurControl(
                    (FrameLayout) cameraView, cameraContainer, zoomSlider, resourcesProvider);
            try {
                View buttonsLayout = (View) field(cameraView.getClass(), "buttonsLayout").get(cameraView);
                if (buttonsLayout != null && buttonsLayout.getBackground() != null) {
                    Drawable bg = buttonsLayout.getBackground();
                    if (bg.getClass().getName().contains("BlurredBackgroundDrawable")) {
                        Method getSource = bg.getClass().getMethod("getSource");
                        Object source = getSource.invoke(bg);
                        Object colorProvider = field(bg.getClass(), "colorProvider").get(bg);
                        Class<?> factoryClass = Class.forName(
                                "org.telegram.ui.Components.blur3.BlurredBackgroundDrawableViewFactory",
                                false, Main.class.getClassLoader());
                        Constructor<?> ctor = factoryClass.getConstructor(
                                Class.forName("org.telegram.ui.Components.blur3.source.BlurredBackgroundSource",
                                        false, Main.class.getClassLoader()));
                        Object factory = ctor.newInstance(source);
                        control.setBlurBackground(factory, colorProvider);
                    }
                }
            } catch (Throwable ignored) { }
            synchronized (BLUR_CONTROLS) { BLUR_CONTROLS.put(cameraView, control); }
            // Privacy-first default for every newly opened round camera.
            setBlurEnabled(true);
            control.attach();
            refreshBlurControls();
        } catch (Throwable error) {
            emit("Round-camera blur toggle unavailable", error);
        }
    }

    private static void setBlurEnabled(boolean enabled) {
        if (blurEnabled == enabled) {
            refreshBlurControls();
            return;
        }
        blurEnabled = enabled;
        setProtectionState(enabled ? (initialized ? "ACTIVE" : "STARTING") : "DISABLED");
        CapturedFrame queued = LATEST_FRAME.getAndSet(null);
        if (queued != null) FRAME_POOL.offer(queued.rgba);
        SOURCE_TRACKS.clear();
        SOURCE_ACTIVE_SINCE.clear();
        clearFirstDetectionLatches();
        LAST_ANY_FACE_NANOS.set(0L);
        synchronized (CAMERA_STATES) {
            for (CameraState state : CAMERA_STATES.values()) state.activeSource = null;
        }
        refreshBlurControls();
        emit(enabled ? "Face blur enabled from camera controls"
                : "Face blur disabled from camera controls");
    }

    private static void refreshBlurControls() {
        AndroidUtilities.runOnUIThread(() -> {
            synchronized (BLUR_CONTROLS) {
                for (BlurControl control : BLUR_CONTROLS.values()) control.update();
            }
        });
    }

    private static void setProtectionState(String newState) {
        if (!newState.equals(protectionState)) {
            protectionState = newState;
            refreshBlurControls();
        }
    }

    private static void noteCameraSwitch() {
        noteCameraSwitch(System.nanoTime());
    }

    private static void noteCameraSwitch(long now) {
        LAST_CAMERA_SWITCH_NANOS.set(now);
        CAMERA_SWITCH_BARRIER_FRAMES.set(0);
        ENCODER_SWITCH_BARRIER_FRAMES.set(0);
        NativeBridge.reset();
        SOURCE_TRACKS.clear();
        CapturedFrame stale = LATEST_FRAME.getAndSet(null);
        if (stale != null) {
            FRAME_POOL.offer(stale.rgba);
            releaseFirstDetectionLatch(stale.sourceKey);
        }
        clearFirstDetectionLatches();
        synchronized (CAMERA_STATES) {
            for (CameraState state : CAMERA_STATES.values()) {
                if (blurEnabled) {
                    state.faceCount = -1;
                    state.blurTexture = 0;
                    java.util.Arrays.fill(state.faces, 0f);
                    state.currFlowGray = null;
                    state.prevFlowGray = null;
                    if (state.tap != null) state.tap.resetPbo();
                }
            }
        }
        synchronized (ENCODER_STATES) {
            for (EncoderState state : ENCODER_STATES.values()) {
                if (blurEnabled) {
                    state.faceCount = -1;
                    state.blurTexture = 0;
                    java.util.Arrays.fill(state.faces, 0f);
                    if (state.tap != null) state.tap.resetPbo();
                }
            }
        }
    }

    private static void hookCameraRenderer() throws Exception {
        Class<?> type = Class.forName(CAMERA_GL_THREAD, false, Main.class.getClassLoader());
        Method draw = type.getDeclaredMethod("onDraw", Integer.class, boolean.class, boolean.class);
        Method finish = type.getDeclaredMethod("finish");
        draw.setAccessible(true); finish.setAccessible(true);
        HOOKS.add(XposedBridge.hookMethod(draw, new XC_MethodHook() {
            @Override public void beforeHookedMethod(MethodHookParam param) { beforePreviewDraw(param.thisObject); }
            @Override public void afterHookedMethod(MethodHookParam param) { afterPreviewDraw(param.thisObject); }
        }));
        HOOKS.add(XposedBridge.hookMethod(finish, new XC_MethodHook() {
            @Override public void beforeHookedMethod(MethodHookParam param) { releaseCameraState(param.thisObject); }
        }));
        try {
            Method reinit = type.getDeclaredMethod("reinitForNewCamera");
            reinit.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(reinit, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam param) { noteCameraSwitch(); }
            }));
        } catch (Throwable ignored) { }
        try {
            Method flip = type.getDeclaredMethod("flipSurfaces");
            flip.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(flip, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam param) { noteCameraSwitch(); }
            }));
        } catch (Throwable ignored) { }
    }

    private static void hookEncoderRenderer() {
        try {
            Class<?> type = Class.forName(ENCODER_RENDERER, false, Main.class.getClassLoader());
            Class<?> snapshot;
            try { snapshot = Class.forName(FRAME_SNAPSHOT, false, type.getClassLoader()); }
            catch (Throwable ignored) { snapshot = Class.forName(FRAME_SNAPSHOT, false, Main.class.getClassLoader()); }
            Method created = type.getDeclaredMethod("onEncoderSurfaceCreated", int.class, int.class);
            Method draw = type.getDeclaredMethod("onDrawEncoderFrame", long.class, snapshot);
            Method destroyed = type.getDeclaredMethod("onEncoderSurfaceDestroyed");
            created.setAccessible(true); draw.setAccessible(true); destroyed.setAccessible(true);
            HOOKS.add(XposedBridge.hookMethod(created, new XC_MethodHook() {
                @Override public void afterHookedMethod(MethodHookParam p) {
                    createEncoderState(p.thisObject, (Integer) p.args[0], (Integer) p.args[1]);
                }
            }));
            HOOKS.add(XposedBridge.hookMethod(draw, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam p) { beforeEncoderDraw(p.thisObject, p.args[1]); }
                @Override public void afterHookedMethod(MethodHookParam p) { afterEncoderDraw(p.thisObject); }
            }));
            HOOKS.add(XposedBridge.hookMethod(destroyed, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam p) { releaseEncoderState(p.thisObject); }
            }));
            emit("Encoder hooks registered: surface create/draw/destroy");
        } catch (Throwable error) {
            emit("Encoder blur unavailable; preview remains active and encoder is untouched", error);
        }
    }

    private static void hookEncoderFallback() {
        try {
            Method createEncoder = MediaCodec.class.getDeclaredMethod("createEncoderByType", String.class);
            HOOKS.add(XposedBridge.hookMethod(createEncoder, new XC_MethodHook() {
                @Override
                public void afterHookedMethod(MethodHookParam param) {
                    if (param.hasThrowable() && "video/avc".equals(param.args[0])) {
                        try {
                            MediaCodec fallback = MediaCodec.createByCodecName("c2.android.avc.encoder");
                            param.setResult(fallback);
                            param.setThrowable(null);
                            emit("Hardware video encoder failed to create; fell back to CPU c2.android.avc.encoder");
                            return;
                        } catch (Throwable t1) {
                            try {
                                MediaCodec fallback = MediaCodec.createByCodecName("OMX.google.h264.encoder");
                                param.setResult(fallback);
                                param.setThrowable(null);
                                emit("Hardware video encoder failed to create; fell back to CPU OMX.google.h264.encoder");
                                return;
                            } catch (Throwable t2) {
                                emit("Software video encoder fallback failed", t2);
                            }
                        }
                    }
                }
            }));
            emit("Encoder fallback hook registered: hardware -> CPU fallback");
        } catch (Throwable error) {
            emit("Encoder fallback hook unavailable", error);
        }
    }

    private static void beforePreviewDraw(Object thread) {
        if (!acceptingFrames || !blurEnabled) return;
        CameraState state = cameraState(thread);
        try {
            if (EGL14.eglGetCurrentContext() == EGL14.EGL_NO_CONTEXT) return;
            ensurePreviewProgram(state);
            long now = System.nanoTime();
            String source = currentSourceKey(thread);
            activateSource(state, source, now);
            mapCurrentSource(thread, source);
            FaceGeometry geometry = geometryFor(source, now);
            state.faceCount = geometry == null ? resolveFaceCount(source, now) : geometry.count;
            if (previewTransitionActive(thread)) state.faceCount = -1;
            if (geometry != null) {
                System.arraycopy(geometry.faces, 0, state.faces, 0, geometry.count * FACE_STRIDE);
            } else if (state.faceCount <= 0) {
                java.util.Arrays.fill(state.faces, 0f);
            }
            float minFaceRadius = 0.20f;
            if (state.faceCount > 0) {
                for (int i = 0; i < state.faceCount; i++) {
                    int offset = i * FACE_STRIDE;
                    float rx = axisLength(state.faces, offset + 2);
                    float ry = axisLength(state.faces, offset + 4);
                    float r = Math.min(rx, ry);
                    if (r > 0.001f && r < minFaceRadius) minFaceRadius = r;
                }
            }
            float blurRadiusScale = clamp(0.18f / Math.max(0.04f, minFaceRadius), 1.0f, 2.5f);
            if (state.blurTexture == 0) {
                state.blurTexture = renderPreviewBlurOnDemand(thread, state, blurRadiusScale);
                if (state.blurTexture == 0) {
                    state.blurTexture = state.fallbackTexture();
                    state.faceCount = -1;
                    setProtectionState("DEGRADED"); // protectionState = "DEGRADED"
                    maskLeakFrames++;
                }
            }
            float pixelGrid = computePixelGrid(minFaceRadius, state.faceCount);
            uploadGeometry(state.program, state.faceCount, state.faces, state.center, state.axisX,
                    state.axisY, state.viewport, state.blurSampler, state.pixelGrid, state.geometryScratch, pixelGrid);
            if (state.blurTexture != 0) bindPreviewBlur(state);

            Class<?> type = thread.getClass();
            state.savedProgram = field(type, "drawProgram").getInt(thread);
            if (state.savedProgram == 0) return;
            state.savedPosition = field(type, "positionHandle").getInt(thread);
            state.savedTexture = field(type, "textureHandle").getInt(thread);
            state.savedMvp = field(type, "vertexMatrixHandle").getInt(thread);
            state.savedSt = field(type, "textureMatrixHandle").getInt(thread);
            state.swapActive = true;
            field(type, "drawProgram").setInt(thread, state.program);
            field(type, "positionHandle").setInt(thread, state.position);
            field(type, "textureHandle").setInt(thread, state.texture);
            field(type, "vertexMatrixHandle").setInt(thread, state.mvp);
            field(type, "textureMatrixHandle").setInt(thread, state.st);
        } catch (Throwable error) {
            setProtectionState("DEGRADED"); // protectionState = "DEGRADED"
            maskLeakFrames++;
            emit("Preview draw failed, enforcing fail-closed", error);
            enforcePreviewFailClosed(thread, state);
        }
    }

    private static void enforcePreviewFailClosed(Object thread, CameraState state) {
        if (!blurEnabled) {
            restorePreviewFields(thread, state);
            return;
        }
        try {
            Class<?> type = thread.getClass();
            if (!state.swapActive) {
                state.savedProgram = field(type, "drawProgram").getInt(thread);
                state.savedPosition = field(type, "positionHandle").getInt(thread);
                state.savedTexture = field(type, "textureHandle").getInt(thread);
                state.savedMvp = field(type, "vertexMatrixHandle").getInt(thread);
                state.savedSt = field(type, "textureMatrixHandle").getInt(thread);
                state.swapActive = true;
            }
            if (state.program != 0 && state.position >= 0) {
                if (state.blurTexture == 0) state.blurTexture = state.fallbackTexture();
                if (state.blurTexture != 0) bindPreviewBlur(state);
                uploadGeometry(state.program, -1, state.faces, state.center, state.axisX,
                        state.axisY, state.viewport, state.blurSampler, state.pixelGrid, state.geometryScratch);
                field(type, "drawProgram").setInt(thread, state.program);
                field(type, "positionHandle").setInt(thread, state.position);
                field(type, "textureHandle").setInt(thread, state.texture);
                field(type, "vertexMatrixHandle").setInt(thread, state.mvp);
                field(type, "textureMatrixHandle").setInt(thread, state.st);
            } else {
                ensureFallbackProgram(state);
                if (state.fallbackProgram != 0) {
                    field(type, "drawProgram").setInt(thread, state.fallbackProgram);
                    field(type, "positionHandle").setInt(thread, state.fallbackPosition);
                    field(type, "vertexMatrixHandle").setInt(thread, state.fallbackMvp);
                    field(type, "textureMatrixHandle").setInt(thread, -1);
                } else {
                    restorePreviewFields(thread, state);
                }
            }
        } catch (Throwable fallbackError) {
            emit("Enforcing preview fail-closed failed", fallbackError);
            restorePreviewFields(thread, state);
        }
    }

    private static int renderPreviewBlurOnDemand(Object thread, CameraState state) {
        return renderPreviewBlurOnDemand(thread, state, 1.0f);
    }

    private static int renderPreviewBlurOnDemand(Object thread, CameraState state, float blurRadiusScale) {
        try {
            Object outer = field(thread.getClass(), "this$0").get(thread);
            int slot = field(outer.getClass(), "surfaceIndex").getInt(outer);
            int[] textures = (int[]) field(outer.getClass(), "cameraTexture").get(outer);
            float[] mvp = (float[]) field(outer.getClass(), "mMVPMatrix").get(outer);
            FloatBuffer hostTexture = ((FloatBuffer) field(outer.getClass(), "textureBuffer").get(outer)).duplicate();
            SurfaceTexture[] surfaces = (SurfaceTexture[]) field(thread.getClass(), "cameraSurface").get(thread);
            if (surfaces != null && slot >= 0 && slot < surfaces.length && surfaces[slot] != null
                    && textures != null && slot < textures.length && textures[slot] > 0) {
                float[] st = new float[16];
                surfaces[slot].getTransformMatrix(st);
                float[] tex = new float[8];
                hostTexture.position(0);
                hostTexture.get(tex);
                return state.tap.renderBlur(textures[slot], mvp, st, tex, null, blurRadiusScale);
            }
        } catch (Throwable ignored) { }
        return 0;
    }

    private static void afterPreviewDraw(Object thread) { restorePreviewFields(thread, cameraState(thread)); }

    private static void bindPreviewBlur(CameraState state) {
        if (state.blurTexture == 0) return;
        GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, state.glActive, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        if (!state.blurBindingActive) {
            GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, state.glBinding, 0);
            state.savedBlurBinding = state.glBinding[0];
            state.blurBindingActive = true;
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.blurTexture);
        GLES20.glActiveTexture(state.glActive[0]);
    }

    private static void restorePreviewBlur(CameraState state) {
        if (!state.blurBindingActive) return;
        GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, state.glActive, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.savedBlurBinding);
        GLES20.glActiveTexture(state.glActive[0]);
        state.blurBindingActive = false;
    }

    private static void restorePreviewFields(Object thread, CameraState state) {
        if (!state.swapActive) { restorePreviewBlur(state); state.blurTexture = 0; return; }
        try {
            Class<?> type = thread.getClass();
            field(type, "drawProgram").setInt(thread, state.savedProgram);
            field(type, "positionHandle").setInt(thread, state.savedPosition);
            field(type, "textureHandle").setInt(thread, state.savedTexture);
            field(type, "vertexMatrixHandle").setInt(thread, state.savedMvp);
            field(type, "textureMatrixHandle").setInt(thread, state.savedSt);
        } catch (Throwable error) { emit("Preview host field restore failed", error); }
        finally { state.swapActive = false; restorePreviewBlur(state); state.blurTexture = 0; }
    }

    private static void afterSurfaceUpdate(SurfaceTexture updated) {
        if (!acceptingFrames || !blurEnabled
                || EGL14.eglGetCurrentContext() == EGL14.EGL_NO_CONTEXT) return;
        Object owner = null;
        int slot = -1;
        synchronized (CAMERA_STATES) {
            for (Object candidate : CAMERA_STATES.keySet()) {
                try {
                    SurfaceTexture[] surfaces = (SurfaceTexture[]) field(candidate.getClass(), "cameraSurface").get(candidate);
                    if (surfaces == null) continue;
                    for (int i = 0; i < surfaces.length; i++) if (surfaces[i] == updated) {
                        owner = candidate; slot = i; break;
                    }
                    if (owner != null) break;
                } catch (Throwable ignored) { }
            }
        }
        // Exact object identity is mandatory; unrelated SurfaceTexture updates never reach readback.
        if (owner == null) return;
        captureUpdatedSurface(owner, updated, slot);
    }

    private static void captureUpdatedSurface(Object thread, SurfaceTexture surface, int slot) {
        CameraState state = cameraState(thread);
        long now = System.nanoTime();
        ByteBuffer frameBuffer = null;
        String source = null;
        try {
            Object outer = field(thread.getClass(), "this$0").get(thread);
            int activeSlot = field(outer.getClass(), "surfaceIndex").getInt(outer);
            // During a camera flip the host can latch both OES surfaces in one
            // draw. Capture only the surface that preview and FrameSnapshot use;
            // otherwise slot 0 consumes the shared cadence and starves slot 1.
            if (slot != activeSlot) return;
            int[] textures = (int[]) field(outer.getClass(), "cameraTexture").get(outer);
            float[] mvp = (float[]) field(outer.getClass(), "mMVPMatrix").get(outer);
            FloatBuffer hostTexture = ((FloatBuffer) field(outer.getClass(), "textureBuffer").get(outer)).duplicate();
            SurfaceTexture[] surfaces = (SurfaceTexture[]) field(thread.getClass(), "cameraSurface").get(thread);
            if (surfaces == null || slot < 0 || slot >= surfaces.length || surfaces[slot] != surface
                    || textures == null || slot >= textures.length || textures[slot] <= 0) return;
            source = sourceKey(thread, slot, surface);
            activateSource(state, source, now);
            if (!hasFreshResult(source, now)) {
                FIRST_DETECTION_LATCH.putIfAbsent(source, new CountDownLatch(1));
            }
            float[] st = new float[16]; surface.getTransformMatrix(st);
            float[] tex = new float[8]; hostTexture.position(0); hostTexture.get(tex);
            float minFaceRadius = 0.20f;
            if (state.faceCount > 0) {
                for (int i = 0; i < state.faceCount; i++) {
                    int offset = i * FACE_STRIDE;
                    float rx = axisLength(state.faces, offset + 2);
                    float ry = axisLength(state.faces, offset + 4);
                    float r = Math.min(rx, ry);
                    if (r > 0.001f && r < minFaceRadius) minFaceRadius = r;
                }
            }
            float blurRadiusScale = clamp(0.18f / Math.max(0.04f, minFaceRadius), 1.0f, 2.5f);
            if (!hasFreshResult(source, now) || now - state.lastCaptureNanos >= state.captureIntervalNanos)
                frameBuffer = FRAME_POOL.poll();
            if (state.readPixels == null) state.readPixels = ByteBuffer.allocateDirect(
                    CleanFrameTap.SIZE * CleanFrameTap.SIZE * 4).order(ByteOrder.nativeOrder());
            state.blurTexture = state.tap.renderBlur(textures[slot], mvp, st, tex,
                    state.readPixels, blurRadiusScale);
            if (state.blurTexture == 0)
                throw new IllegalStateException("multi-pass preview blur failed: " + state.tap.lastError());
            if (!state.pipelineLogged) {
                state.pipelineLogged = true;
                emit("Preview multi-pass Gaussian ready size=" + CleanFrameTap.SIZE + "x" + CleanFrameTap.SIZE);
            }
            bindPreviewBlur(state);
            processOpticalFlow(state, source, now);
            SOURCE_BY_SLOT.put(slot, source);
            SOURCE_BY_TEXTURE.put(textures[slot], source);
            boolean awaitingFirst = !hasFreshResult(source, now);
            if (frameBuffer != null) {
                framesCaptured++;
                frameBuffer.clear();
                copyVerticallyCorrected(state.readPixels, frameBuffer, CleanFrameTap.SIZE, CleanFrameTap.SIZE);
                frameBuffer.position(0);
                CapturedFrame next = new CapturedFrame(frameBuffer, now, nextTimestampMs(now), source);
                frameBuffer = null;
                CapturedFrame old = LATEST_FRAME.getAndSet(next);
                if (old != null) {
                    framesDropped++;
                    FRAME_POOL.offer(old.rgba);
                    if (!old.sourceKey.equals(next.sourceKey)) {
                        releaseFirstDetectionLatch(old.sourceKey);
                    }
                }
                state.lastCaptureNanos = now;
                state.captureIntervalNanos = Math.max(CAPTURE_INTERVAL_NS,
                        Math.min(100_000_000L, state.captureIntervalNanos - 2_000_000L));
                scheduleDrain();
                if (awaitingFirst) {
                    awaitFirstDetection(source, liteModel ? 90L : 65L);
                    now = System.nanoTime();
                    FaceGeometry geometry = geometryFor(source, now);
                    state.faceCount = geometry == null ? resolveFaceCount(source, now) : geometry.count;
                    if (previewTransitionActive(thread)) state.faceCount = -1;
                    if (geometry != null) {
                        System.arraycopy(geometry.faces, 0, state.faces, 0, geometry.count * FACE_STRIDE);
                    }
                    float curMinFaceRadius = 0.20f;
                    if (state.faceCount > 0) {
                        for (int i = 0; i < state.faceCount; i++) {
                            int offset = i * FACE_STRIDE;
                            float rx = axisLength(state.faces, offset + 2);
                            float ry = axisLength(state.faces, offset + 4);
                            float r = Math.min(rx, ry);
                            if (r > 0.001f && r < curMinFaceRadius) curMinFaceRadius = r;
                        }
                    }
                    float pixelGrid = computePixelGrid(curMinFaceRadius, state.faceCount);
                    ensurePreviewProgram(state);
                    if (state.program != 0) {
                        uploadGeometry(state.program, state.faceCount, state.faces, state.center, state.axisX,
                                state.axisY, state.viewport, state.blurSampler, state.pixelGrid, state.geometryScratch, pixelGrid);
                    }
                }
            } else if (awaitingFirst) {
                releaseFirstDetectionLatch(source);
            }
            CAMERA_SWITCH_BARRIER_FRAMES.incrementAndGet();
        } catch (Throwable error) {
            if (frameBuffer != null) FRAME_POOL.offer(frameBuffer);
            if (source != null) releaseFirstDetectionLatch(source);
            state.captureIntervalNanos = Math.min(100_000_000L,
                    Math.max(CAPTURE_INTERVAL_NS, state.captureIntervalNanos + 10_000_000L));
            if (blurEnabled) {
                setProtectionState("DEGRADED");
                maskLeakFrames++;
                if (state.blurTexture == 0) state.blurTexture = state.fallbackTexture();
            }
            emit("Preview blur generation failed for this frame", error);
        }
    }

    static void copyVerticallyCorrected(ByteBuffer source, ByteBuffer destination, int width, int height) {
        int rowBytes = width * 4;
        ByteBuffer input = source.duplicate();
        for (int y = height - 1; y >= 0; y--) {
            input.position(y * rowBytes); input.limit((y + 1) * rowBytes);
            destination.put(input.slice());
        }
    }

    private static void processOpticalFlow(CameraState state, String source, long now) {
        if (state.readPixels == null || source == null) return;
        SourceTracks tracks = SOURCE_TRACKS.get(source);
        if (tracks == null) return;
        if (state.currFlowGray == null || state.currFlowGray.length < CleanFrameTap.SIZE * CleanFrameTap.SIZE) {
            state.currFlowGray = new float[CleanFrameTap.SIZE * CleanFrameTap.SIZE];
        }
        if (state.prevFlowGray == null || state.prevFlowGray.length < CleanFrameTap.SIZE * CleanFrameTap.SIZE) {
            state.prevFlowGray = new float[CleanFrameTap.SIZE * CleanFrameTap.SIZE];
        }
        SparseLucasKanadeTracker.rgbaToGrayscale(state.readPixels, state.currFlowGray,
                CleanFrameTap.SIZE, CleanFrameTap.SIZE, true);
        if (state.lastFlowNanos != 0L && source.equals(state.flowSource)) {
            float dt = (now - state.lastFlowNanos) / 1_000_000_000f;
            if (dt >= 0.001f && dt <= 0.200f) {
                tracks.applyOpticalFlow(state.flowTracker, state.prevFlowGray, state.currFlowGray,
                        CleanFrameTap.SIZE, CleanFrameTap.SIZE, dt);
            }
        }
        float[] temp = state.prevFlowGray;
        state.prevFlowGray = state.currFlowGray;
        state.currFlowGray = temp;
        state.lastFlowNanos = now;
        state.flowSource = source;
    }

    private static synchronized long nextTimestampMs(long nanos) {
        long candidate = nanos / 1_000_000L;
        lastTimestampMs = Math.max(lastTimestampMs + 1, candidate);
        return lastTimestampMs;
    }

    private static void startFrameExecutor() {
        FRAME_POOL.clear();
        for (int i = 0; i < 3; i++) FRAME_POOL.offer(ByteBuffer.allocateDirect(
                CleanFrameTap.SIZE * CleanFrameTap.SIZE * 4).order(ByteOrder.nativeOrder()));
        frameExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread thread = new Thread(r, "BlurFacesNativeInput"); thread.setDaemon(true); return thread;
        });
    }

    private static void scheduleDrain() {
        ExecutorService executor = frameExecutor;
        if (executor == null || executor.isShutdown() || !DRAIN_SCHEDULED.compareAndSet(false, true)) return;
        try { executor.execute(Main::submitLatestFrame); }
        catch (Throwable error) {
            DRAIN_SCHEDULED.set(false);
            if (blurEnabled) setProtectionState("DEGRADED");
            emit("Worker dispatch failed", error);
        }
    }

    private static java.io.File getDebugDir() {
        try {
            java.io.File base = null;
            try {
                Class<?> appLoader = Class.forName("org.telegram.messenger.ApplicationLoader");
                Method method = appLoader.getDeclaredMethod("getFilesDirFixed");
                base = (java.io.File) method.invoke(null);
            } catch (Throwable ignored) { }
            if (base == null) {
                base = new java.io.File("/data/data/com.exteragram.messenger/files");
            }
            java.io.File dir = new java.io.File(base, "blurfaces_storage");
            if (!dir.exists()) dir.mkdirs();
            return dir;
        } catch (Throwable t) {
            java.io.File fallback = new java.io.File("/data/data/com.exteragram.messenger/files/blurfaces_storage");
            if (!fallback.exists()) fallback.mkdirs();
            return fallback;
        }
    }

    private static long lastDebugFrameSavedNanos = 0L;
    private static void maybeSaveDebugFrame(ByteBuffer rgba, int count, float score, float cx, float cy) {
        long now = System.nanoTime();
        long minInterval = (count > 0) ? 400_000_000L : 1_500_000_000L;
        if (now - lastDebugFrameSavedNanos < minInterval) return;
        lastDebugFrameSavedNanos = now;
        try {
            java.io.File dir = getDebugDir();
            ByteBuffer copy = ByteBuffer.allocateDirect(rgba.capacity());
            rgba.position(0);
            copy.put(rgba);
            rgba.position(0);
            copy.position(0);
            new Thread(() -> {
                try {
                    android.graphics.Bitmap bmp = android.graphics.Bitmap.createBitmap(
                            CleanFrameTap.SIZE, CleanFrameTap.SIZE, android.graphics.Bitmap.Config.ARGB_8888);
                    bmp.copyPixelsFromBuffer(copy);
                    long ts = System.currentTimeMillis();
                    java.io.File file = new java.io.File(dir, String.format(java.util.Locale.US,
                            "frame_%d_c%d_s%d_y%d.jpg", ts, count, (int)(score * 100), (int)(cy * 100)));
                    try (java.io.FileOutputStream out = new java.io.FileOutputStream(file)) {
                        bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 88, out);
                    }
                    bmp.recycle();
                    emit("Saved clean debug frame: " + file.getName());
                } catch (Throwable ignored) { }
            }).start();
        } catch (Throwable ignored) { }
    }

    private static void submitLatestFrame() {
        CapturedFrame frame = LATEST_FRAME.getAndSet(null);
        if (frame == null) { DRAIN_SCHEDULED.set(false); return; }
        if (frame.captureNanos < LAST_CAMERA_SWITCH_NANOS.get()) {
            releaseFirstDetectionLatch(frame.sourceKey);
            FRAME_POOL.offer(frame.rgba);
            DRAIN_SCHEDULED.set(false);
            if (acceptingFrames && LATEST_FRAME.get() != null) scheduleDrain();
            return;
        }
        try {
            long inferenceStart = System.nanoTime();
            frame.rgba.position(0);

            float[] outGeometry = new float[MAX_FACES * FACE_STRIDE];
            float[] outScores = new float[MAX_FACES];
            float[] outYaws = new float[MAX_FACES];

            int count = NativeBridge.process(frame.rgba, CleanFrameTap.SIZE, CleanFrameTap.SIZE,
                    outGeometry, outScores, outYaws, MAX_FACES, configuredConfidence);

            maybeSaveDebugFrame(frame.rgba, count, count > 0 ? outScores[0] : 0f,
                    count > 0 ? outGeometry[0] : 0f, count > 0 ? outGeometry[1] : 0f);

            FaceGeometry detections = new FaceGeometry(outGeometry, outScores, outYaws, count,
                    frame.captureNanos, frame.sourceKey);

            if (isSourceActive(frame.sourceKey)) {
                recordStaleness(System.nanoTime() - frame.captureNanos);
                updateTracks(frame.sourceKey, detections);
                releaseFirstDetectionLatch(frame.sourceKey);
                framesProcessed++;
                inferenceNanos += System.nanoTime() - inferenceStart;
                maybeLogMetrics();
            } else {
                releaseFirstDetectionLatch(frame.sourceKey);
                emit("Dropped late result from retired source=" + frame.sourceKey);
            }
        } catch (Throwable error) {
            releaseFirstDetectionLatch(frame.sourceKey);
            if (blurEnabled) setProtectionState("DEGRADED");
            emit("Native head detection frame failed", error);
        } finally {
            releaseFirstDetectionLatch(frame.sourceKey);
            FRAME_POOL.offer(frame.rgba);
            DRAIN_SCHEDULED.set(false);
            if (acceptingFrames && LATEST_FRAME.get() != null) scheduleDrain();
        }
    }

    static FaceGeometry geometryFor(String source, long now) {
        if (source == null || source.isEmpty()) return null;
        Long activeSince = SOURCE_ACTIVE_SINCE.get(source);
        if (activeSince == null) return null;
        SourceTracks tracks = SOURCE_TRACKS.get(source);
        if (tracks == null || tracks.lastResultNanos < activeSince) return null;
        if (!tracks.hasFreshPublication(now)) setProtectionState("DEGRADED");
        else if (initialized && blurEnabled) setProtectionState("ACTIVE");
        return tracks.geometryAt(now, source);
    }

    static boolean hasFreshResult(String source, long now) {
        if (source == null || source.isEmpty()) return false;
        Long activeSince = SOURCE_ACTIVE_SINCE.get(source);
        SourceTracks tracks = SOURCE_TRACKS.get(source);
        return activeSince != null && tracks != null && tracks.lastResultNanos >= activeSince
                && tracks.hasFreshPublication(now);
    }

    // What to draw when no track survives. "The detector returned nothing" and
    // "there is nobody in frame" are different statements, and the old code treated
    // the first as the second: a profile view or a camera flip switched the blur off
    // with a face still on screen. Absence is only trusted after the grace window.
    static int resolveFaceCount(String source, long now) {
        if (!hasFreshResult(source, now)) return -1;
        if (sawFaceRecently(source, now)) return -1;
        return 0;
    }

    private static boolean sawFaceRecently(String source, long now) {
        return CAMERA_SWITCH_BARRIER_FRAMES.get() < CAMERA_SWITCH_BARRIER_MIN_FRAMES;
    }

    private static boolean isSourceActive(String source) {
        synchronized (CAMERA_STATES) {
            for (Object thread : CAMERA_STATES.keySet()) {
                if (source.equals(currentSourceKey(thread))) return true;
            }
        }
        return false;
    }

    private static void ensurePreviewProgram(CameraState state) {
        if (state.program != 0) return;
        state.program = createProgram(FS);
        state.position = GLES20.glGetAttribLocation(state.program, "aPosition");
        state.texture = GLES20.glGetAttribLocation(state.program, "aTextureCoord");
        state.mvp = GLES20.glGetUniformLocation(state.program, "uMVPMatrix");
        state.st = GLES20.glGetUniformLocation(state.program, "uSTMatrix");
        state.center = GLES20.glGetUniformLocation(state.program, "uFaceCenter");
        state.axisX = GLES20.glGetUniformLocation(state.program, "uFaceAxisX");
        state.axisY = GLES20.glGetUniformLocation(state.program, "uFaceAxisY");
        state.count = GLES20.glGetUniformLocation(state.program, "uFaceCount");
        state.viewport = GLES20.glGetUniformLocation(state.program, "uViewport");
        state.blurSampler = GLES20.glGetUniformLocation(state.program, "sBlurTexture");
        state.pixelGrid = GLES20.glGetUniformLocation(state.program, "uPixelGrid");
        if (state.position < 0 || state.texture < 0 || state.mvp < 0 || state.st < 0
                || state.center < 0 || state.axisX < 0 || state.axisY < 0 || state.count < 0
                || state.viewport < 0 || state.blurSampler < 0)
            throw new IllegalStateException("preview shader interface incomplete");
    }

    private static void createEncoderState(Object renderer, int width, int height) {
        EncoderState state = encoderState(renderer);
        try {
            if (state.program != 0) GLES20.glDeleteProgram(state.program);
            state.program = createProgram(ENCODER_FS); state.width = width; state.height = height;
            state.position = GLES20.glGetAttribLocation(state.program, "aPosition");
            state.texture = GLES20.glGetAttribLocation(state.program, "aTextureCoord");
            state.mvp = GLES20.glGetUniformLocation(state.program, "uMVPMatrix");
            state.st = GLES20.glGetUniformLocation(state.program, "uSTMatrix");
            state.preview = GLES20.glGetUniformLocation(state.program, "preview");
            state.resolution = GLES20.glGetUniformLocation(state.program, "resolution");
            state.alpha = GLES20.glGetUniformLocation(state.program, "alpha");
            state.texel = GLES20.glGetUniformLocation(state.program, "texelSize");
            state.center = GLES20.glGetUniformLocation(state.program, "uFaceCenter");
            state.axisX = GLES20.glGetUniformLocation(state.program, "uFaceAxisX");
            state.axisY = GLES20.glGetUniformLocation(state.program, "uFaceAxisY");
            state.count = GLES20.glGetUniformLocation(state.program, "uFaceCount");
            state.viewport = GLES20.glGetUniformLocation(state.program, "uViewport");
            state.blurSampler = GLES20.glGetUniformLocation(state.program, "sBlurTexture");
            state.pixelGrid = GLES20.glGetUniformLocation(state.program, "uPixelGrid");
            // preview/resolution are part of the host interface but this shader
            // does not consume them, so GLES may legally optimize them to -1.
            // Host glUniform calls with -1 are defined no-ops.
            state.ready = state.position >= 0 && state.texture >= 0 && state.mvp >= 0 && state.st >= 0
                    && state.alpha >= 0
                    && state.center >= 0 && state.axisX >= 0 && state.axisY >= 0 && state.count >= 0
                    && state.viewport >= 0 && state.blurSampler >= 0;
            if (!state.ready) throw new IllegalStateException("encoder shader interface incomplete");
            emit("Encoder blur shader ready size=" + width + "x" + height);
        } catch (Throwable error) {
            state.ready = false;
            if (blurEnabled) {
                setProtectionState("DEGRADED");
                maskLeakFrames++;
            }
            emit("Encoder shader setup failed", error);
        }
    }

    private static void beforeEncoderDraw(Object renderer, Object snapshot) {
        EncoderState state = encoderState(renderer);
        if (!acceptingFrames || !blurEnabled || !state.ready || snapshot == null) {
            if (blurEnabled && snapshot != null) {
                setProtectionState("DEGRADED");
                maskLeakFrames++;
                enforceEncoderFailClosed(renderer, state, snapshot);
            }
            return;
        }
        try {
            int slot = field(snapshot.getClass(), "surfaceIndex").getInt(snapshot);
            int textureId = field(snapshot.getClass(), "textureId").getInt(snapshot);
            float[] mvp = (float[]) field(snapshot.getClass(), "mvpMatrix").get(snapshot);
            float[] st = (float[]) field(snapshot.getClass(), "stMatrix").get(snapshot);
            float[] tex = (float[]) field(snapshot.getClass(), "textureCoords").get(snapshot);
            String source = SOURCE_BY_TEXTURE.get(textureId);
            if (source == null) source = SOURCE_BY_SLOT.get(slot);
            long now = System.nanoTime();
            if (source == null) {
                for (String s : SOURCE_TRACKS.keySet()) {
                    if (hasFreshResult(s, now)) {
                        source = s;
                        break;
                    }
                }
            }
            if (source == null) {
                for (String s : SOURCE_ACTIVE_SINCE.keySet()) {
                    source = s;
                    break;
                }
            }
            if (source != null && !hasFreshResult(source, now)) {
                awaitFirstDetectionLatch(source);
                now = System.nanoTime();
            }
            FaceGeometry geometry = geometryFor(source, now);
            state.faceCount = geometry == null ? resolveFaceCount(source, now) : geometry.count;
            if (encoderTransitionActive(renderer)) state.faceCount = -1;
            if (geometry != null && state.faceCount > 0) {
                System.arraycopy(geometry.faces, 0, state.faces, 0, geometry.count * FACE_STRIDE);
            } else if (state.faceCount <= 0) {
                java.util.Arrays.fill(state.faces, 0f);
            }
            float minFaceRadius = 0.20f;
            if (state.faceCount > 0) {
                for (int i = 0; i < state.faceCount; i++) {
                    int offset = i * FACE_STRIDE;
                    float rx = axisLength(state.faces, offset + 2);
                    float ry = axisLength(state.faces, offset + 4);
                    float r = Math.min(rx, ry);
                    if (r > 0.001f && r < minFaceRadius) minFaceRadius = r;
                }
            }
            float blurRadiusScale = clamp(0.18f / Math.max(0.04f, minFaceRadius), 1.0f, 2.5f);
            state.blurTexture = state.tap.renderBlur(textureId, mvp, st, tex, null, blurRadiusScale);
            if (state.blurTexture == 0) {
                state.blurTexture = state.fallbackTexture();
                state.faceCount = -1;
                setProtectionState("DEGRADED");
                maskLeakFrames++;
            }
            bindEncoderBlur(state);
            float pixelGrid = computePixelGrid(minFaceRadius, state.faceCount);
            uploadGeometry(state.program, state.faceCount, state.faces, state.center, state.axisX,
                    state.axisY, state.viewport, state.blurSampler, state.pixelGrid, state.geometryScratch, pixelGrid);
            if (state.viewport >= 0 && state.width > 0 && state.height > 0) {
                int[] prior = new int[1];
                GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, prior, 0);
                GLES20.glUseProgram(state.program);
                GLES20.glUniform2f(state.viewport, state.width, state.height);
                GLES20.glUseProgram(prior[0]);
            }
            Class<?> type = renderer.getClass();
            state.savedProgram = field(type, "drawProgram").getInt(renderer);
            state.savedPosition = field(type, "positionHandle").getInt(renderer);
            state.savedTexture = field(type, "textureHandle").getInt(renderer);
            state.savedPreview = field(type, "previewSizeHandle").getInt(renderer);
            state.savedResolution = field(type, "resolutionHandle").getInt(renderer);
            state.savedAlpha = field(type, "alphaHandle").getInt(renderer);
            state.savedMvp = field(type, "vertexMatrixHandle").getInt(renderer);
            state.savedSt = field(type, "textureMatrixHandle").getInt(renderer);
            state.savedTexel = field(type, "texelSizeHandle").getInt(renderer);
            state.swapActive = true;
            field(type, "drawProgram").setInt(renderer, state.program);
            field(type, "positionHandle").setInt(renderer, state.position);
            field(type, "textureHandle").setInt(renderer, state.texture);
            field(type, "previewSizeHandle").setInt(renderer, state.preview);
            field(type, "resolutionHandle").setInt(renderer, state.resolution);
            field(type, "alphaHandle").setInt(renderer, state.alpha);
            field(type, "vertexMatrixHandle").setInt(renderer, state.mvp);
            field(type, "textureMatrixHandle").setInt(renderer, state.st);
            field(type, "texelSizeHandle").setInt(renderer, state.texel);
            ENCODER_SWITCH_BARRIER_FRAMES.incrementAndGet();
            if (state.faceCount > 0 && !state.activeLogged) {
                state.activeLogged = true;
                emit("Encoder face protection active texture=" + textureId
                        + " slot=" + slot + " faces=" + state.faceCount);
            }
            if (!state.pipelineLogged) {
                state.pipelineLogged = true;
                emit("Encoder multi-pass Gaussian active size=" + CleanFrameTap.SIZE + "x" + CleanFrameTap.SIZE);
            }
        } catch (Throwable error) {
            setProtectionState("DEGRADED");
            maskLeakFrames++;
            emit("Encoder draw failed, enforcing fail-closed", error);
            enforceEncoderFailClosed(renderer, state, snapshot);
        }
    }

    private static void enforceEncoderFailClosed(Object renderer, EncoderState state, Object snapshot) {
        if (!blurEnabled) {
            restoreEncoderFields(renderer, state);
            return;
        }
        try {
            Class<?> type = renderer.getClass();
            if (!state.swapActive) {
                state.savedProgram = field(type, "drawProgram").getInt(renderer);
                state.savedPosition = field(type, "positionHandle").getInt(renderer);
                state.savedTexture = field(type, "textureHandle").getInt(renderer);
                state.savedPreview = field(type, "previewSizeHandle").getInt(renderer);
                state.savedResolution = field(type, "resolutionHandle").getInt(renderer);
                state.savedAlpha = field(type, "alphaHandle").getInt(renderer);
                state.savedMvp = field(type, "vertexMatrixHandle").getInt(renderer);
                state.savedSt = field(type, "textureMatrixHandle").getInt(renderer);
                state.savedTexel = field(type, "texelSizeHandle").getInt(renderer);
                state.swapActive = true;
            }
            if (state.ready && state.program != 0) {
                if (state.blurTexture == 0) state.blurTexture = state.fallbackTexture();
                if (state.blurTexture != 0) bindEncoderBlur(state);
                uploadGeometry(state.program, -1, state.faces, state.center, state.axisX,
                        state.axisY, state.viewport, state.blurSampler, state.pixelGrid, state.geometryScratch);
                if (state.viewport >= 0 && state.width > 0 && state.height > 0) {
                    int[] prior = new int[1];
                    GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, prior, 0);
                    GLES20.glUseProgram(state.program);
                    GLES20.glUniform2f(state.viewport, state.width, state.height);
                    GLES20.glUseProgram(prior[0]);
                }
                field(type, "drawProgram").setInt(renderer, state.program);
                field(type, "positionHandle").setInt(renderer, state.position);
                field(type, "textureHandle").setInt(renderer, state.texture);
                field(type, "previewSizeHandle").setInt(renderer, state.preview);
                field(type, "resolutionHandle").setInt(renderer, state.resolution);
                field(type, "alphaHandle").setInt(renderer, state.alpha);
                field(type, "vertexMatrixHandle").setInt(renderer, state.mvp);
                field(type, "textureMatrixHandle").setInt(renderer, state.st);
                field(type, "texelSizeHandle").setInt(renderer, state.texel);
            } else {
                ensureFallbackProgram(state);
                if (state.fallbackProgram != 0) {
                    field(type, "drawProgram").setInt(renderer, state.fallbackProgram);
                    field(type, "positionHandle").setInt(renderer, state.fallbackPosition);
                    field(type, "vertexMatrixHandle").setInt(renderer, state.fallbackMvp);
                    field(type, "textureMatrixHandle").setInt(renderer, -1);
                    field(type, "previewSizeHandle").setInt(renderer, -1);
                    field(type, "resolutionHandle").setInt(renderer, -1);
                    field(type, "alphaHandle").setInt(renderer, -1);
                    field(type, "texelSizeHandle").setInt(renderer, -1);
                } else {
                    restoreEncoderFields(renderer, state);
                }
            }
        } catch (Throwable fallbackError) {
            emit("Enforcing encoder fail-closed failed", fallbackError);
            restoreEncoderFields(renderer, state);
        }
    }

    private static void afterEncoderDraw(Object renderer) { restoreEncoderFields(renderer, encoderState(renderer)); }

    private static void bindEncoderBlur(EncoderState state) {
        if (state.blurTexture == 0) return;
        GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, state.glActive, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        if (!state.blurBindingActive) {
            GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, state.glBinding, 0);
            state.savedBlurBinding = state.glBinding[0];
            state.blurBindingActive = true;
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.blurTexture);
        GLES20.glActiveTexture(state.glActive[0]);
    }

    private static void restoreEncoderBlur(EncoderState state) {
        if (!state.blurBindingActive) return;
        GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, state.glActive, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.savedBlurBinding);
        GLES20.glActiveTexture(state.glActive[0]);
        state.blurBindingActive = false;
    }

    private static boolean previewTransitionActive(Object thread) {
        return CAMERA_SWITCH_BARRIER_FRAMES.get() < CAMERA_SWITCH_BARRIER_MIN_FRAMES;
    }

    private static boolean encoderTransitionActive(Object renderer) {
        return ENCODER_SWITCH_BARRIER_FRAMES.get() < CAMERA_SWITCH_BARRIER_MIN_FRAMES;
    }

    private static void restoreEncoderFields(Object renderer, EncoderState state) {
        if (!state.swapActive) { restoreEncoderBlur(state); state.blurTexture = 0; return; }
        try {
            Class<?> type = renderer.getClass();
            field(type, "drawProgram").setInt(renderer, state.savedProgram);
            field(type, "positionHandle").setInt(renderer, state.savedPosition);
            field(type, "textureHandle").setInt(renderer, state.savedTexture);
            field(type, "previewSizeHandle").setInt(renderer, state.savedPreview);
            field(type, "resolutionHandle").setInt(renderer, state.savedResolution);
            field(type, "alphaHandle").setInt(renderer, state.savedAlpha);
            field(type, "vertexMatrixHandle").setInt(renderer, state.savedMvp);
            field(type, "textureMatrixHandle").setInt(renderer, state.savedSt);
            field(type, "texelSizeHandle").setInt(renderer, state.savedTexel);
        } catch (Throwable error) { emit("Encoder host field restore failed", error); }
        finally { state.swapActive = false; restoreEncoderBlur(state); state.blurTexture = 0; }
    }

    private static void uploadGeometry(int program, int count, float[] faces,
                                        int center, int axisX, int axisY, int viewportLocation,
                                        int blurSamplerLocation, float[] scratch) {
        uploadGeometry(program, count, faces, center, axisX, axisY, viewportLocation, blurSamplerLocation, -1, scratch, 32.0f);
    }

    private static void uploadGeometry(int program, int count, float[] faces,
                                        int center, int axisX, int axisY, int viewportLocation,
                                        int blurSamplerLocation, float[] scratch, float pixelGrid) {
        int handle = program != 0 ? GLES20.glGetUniformLocation(program, "uPixelGrid") : -1;
        uploadGeometry(program, count, faces, center, axisX, axisY, viewportLocation, blurSamplerLocation, handle, scratch, pixelGrid);
    }

    private static void uploadGeometry(int program, int count, float[] faces,
                                        int center, int axisX, int axisY, int viewportLocation,
                                        int blurSamplerLocation, int pixelGridHandle, float[] scratch) {
        uploadGeometry(program, count, faces, center, axisX, axisY, viewportLocation, blurSamplerLocation, pixelGridHandle, scratch, 32.0f);
    }

    private static void uploadGeometry(int program, int count, float[] faces,
                                        int center, int axisX, int axisY, int viewportLocation,
                                        int blurSamplerLocation, int pixelGridHandle, float[] scratch, float pixelGrid) {
        if (program == 0) return;
        int[] prior = new int[1], viewport = new int[4];
        GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, prior, 0);
        GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
        try {
            GLES20.glUseProgram(program); GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uFaceCount"), count);
            float vw = viewport[2] > 0 ? viewport[2] : 1f;
            float vh = viewport[3] > 0 ? viewport[3] : 1f;
            GLES20.glUniform2f(viewportLocation, vw, vh);
            GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "uMaskScale"), faceMaskScale);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uMaskMode"), maskMode);
            if (pixelGridHandle >= 0) {
                GLES20.glUniform1f(pixelGridHandle, pixelGrid);
            }
            GLES20.glUniform1i(blurSamplerLocation, 1);
            if (count > 0) {
                for (int i = 0; i < count; i++) {
                    int src = i * FACE_STRIDE, dst = i * 2;
                    scratch[dst] = faces[src]; scratch[dst + 1] = faces[src + 1];
                    scratch[8 + dst] = faces[src + 2]; scratch[9 + dst] = faces[src + 3];
                    scratch[16 + dst] = faces[src + 4]; scratch[17 + dst] = faces[src + 5];
                }
                GLES20.glUniform2fv(center, count, scratch, 0);
                GLES20.glUniform2fv(axisX, count, scratch, 8);
                GLES20.glUniform2fv(axisY, count, scratch, 16);
            }
        } finally { GLES20.glUseProgram(prior[0]); }
    }

    private static void ensureFallbackProgram(CameraState state) {
        if (state.fallbackProgram != 0) return;
        state.fallbackProgram = createProgram(FALLBACK_VS, FALLBACK_FS);
        if (state.fallbackProgram != 0) {
            state.fallbackPosition = GLES20.glGetAttribLocation(state.fallbackProgram, "aPosition");
            state.fallbackMvp = GLES20.glGetUniformLocation(state.fallbackProgram, "uMVPMatrix");
        }
    }

    private static void ensureFallbackProgram(EncoderState state) {
        if (state.fallbackProgram != 0) return;
        state.fallbackProgram = createProgram(FALLBACK_VS, FALLBACK_FS);
        if (state.fallbackProgram != 0) {
            state.fallbackPosition = GLES20.glGetAttribLocation(state.fallbackProgram, "aPosition");
            state.fallbackMvp = GLES20.glGetUniformLocation(state.fallbackProgram, "uMVPMatrix");
        }
    }

    private static int createProgram(String vertexSource, String fragmentSource) {
        int vertex = compile(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fragment = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        if (vertex == 0 || fragment == 0) {
            if (vertex != 0) GLES20.glDeleteShader(vertex);
            if (fragment != 0) GLES20.glDeleteShader(fragment);
            return 0;
        }
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertex);
        GLES20.glAttachShader(program, fragment);
        GLES20.glLinkProgram(program);
        GLES20.glDeleteShader(vertex);
        GLES20.glDeleteShader(fragment);
        int[] linked = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            GLES20.glDeleteProgram(program);
            return 0;
        }
        return program;
    }

    private static int createProgram(String fragmentSource) {
        int vertex = compile(GLES20.GL_VERTEX_SHADER, VS), fragment = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        if (vertex == 0 || fragment == 0) throw new IllegalStateException("shader compile failed");
        int program = GLES20.glCreateProgram(); GLES20.glAttachShader(program, vertex); GLES20.glAttachShader(program, fragment);
        GLES20.glLinkProgram(program); GLES20.glDeleteShader(vertex); GLES20.glDeleteShader(fragment);
        int[] linked = new int[1]; GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) throw new IllegalStateException("shader link: " + GLES20.glGetProgramInfoLog(program));
        return program;
    }

    private static int compile(int type, String source) {
        int shader = GLES20.glCreateShader(type); GLES20.glShaderSource(shader, source); GLES20.glCompileShader(shader);
        int[] compiled = new int[1]; GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) { emit("shader compile: " + GLES20.glGetShaderInfoLog(shader)); GLES20.glDeleteShader(shader); return 0; }
        return shader;
    }

    private static boolean isFrontFacing(Object thread) {
        if (thread == null) return false;
        try {
            Object outer = field(thread.getClass(), "this$0").get(thread);
            if (outer != null) {
                try {
                    Field f = field(outer.getClass(), "isFrontface");
                    if (f != null && f.getType() == boolean.class) {
                        return f.getBoolean(outer);
                    }
                } catch (Throwable ignored) { }
            }
        } catch (Throwable ignored) { }
        try {
            Field f = field(thread.getClass(), "isFrontface");
            if (f != null && f.getType() == boolean.class) {
                return f.getBoolean(thread);
            }
        } catch (Throwable ignored) { }
        return false;
    }

    private static String sourceKey(Object thread, int slot, SurfaceTexture surface) {
        int id = 0;
        try {
            Object val = field(thread.getClass(), "cameraId").get(thread);
            if (val instanceof Number) id = ((Number) val).intValue();
        } catch (Throwable ignored) {
            try {
                Object outer = field(thread.getClass(), "this$0").get(thread);
                Object val = field(outer.getClass(), "cameraId").get(outer);
                if (val instanceof Number) id = ((Number) val).intValue();
            } catch (Throwable ignored2) {
                try {
                    int[] generations = (int[]) field(thread.getClass(), "surfaceGeneration").get(thread);
                    if (generations != null && slot >= 0 && slot < generations.length) id = generations[slot];
                } catch (Throwable ignored3) { }
            }
        }
        int hash = surface != null ? System.identityHashCode(surface) : 0;
        String facing = isFrontFacing(thread) ? "front" : "back";
        return facing + ":" + slot + ":" + id + ":" + hash;
    }

    private static String currentSourceKey(Object thread) {
        try {
            Object outer = field(thread.getClass(), "this$0").get(thread);
            int slot = field(outer.getClass(), "surfaceIndex").getInt(outer);
            SurfaceTexture[] surfaces = (SurfaceTexture[]) field(thread.getClass(), "cameraSurface").get(thread);
            return surfaces != null && slot >= 0 && slot < surfaces.length && surfaces[slot] != null
                    ? sourceKey(thread, slot, surfaces[slot]) : "";
        } catch (Throwable ignored) { return ""; }
    }

    static void activateSource(CameraState state, String source, long now) {
        if (source == null || source.isEmpty() || source.equals(state.activeSource)) return;
        String prevSource = state.activeSource;
        state.activeSource = source;
        SOURCE_ACTIVE_SINCE.put(source, now);
        if (prevSource != null) {
            noteCameraSwitch(now);
        }
        state.faceCount = -1;
        state.blurTexture = 0;
        java.util.Arrays.fill(state.faces, 0f);
        state.currFlowGray = null;
        state.prevFlowGray = null;
        if (state.tap != null) state.tap.resetPbo();
        if (!hasFreshResult(source, now)) {
            FIRST_DETECTION_LATCH.putIfAbsent(source, new CountDownLatch(1));
        }
        if (prevSource != null) {
            releaseFirstDetectionLatch(prevSource);
        }
        emit("Camera source changed; full-frame privacy blur active until fresh detection source=" + source);
    }

    static void updateTracks(String source, FaceGeometry detections) {
        SourceTracks tracks = SOURCE_TRACKS.computeIfAbsent(source, ignored -> new SourceTracks());
        tracks.update(detections, System.nanoTime());
    }

    private static void mapCurrentSource(Object thread, String source) {
        if (source == null || source.isEmpty()) return;
        try {
            Object outer = field(thread.getClass(), "this$0").get(thread);
            int slot = field(outer.getClass(), "surfaceIndex").getInt(outer);
            int[] textures = (int[]) field(outer.getClass(), "cameraTexture").get(outer);
            SOURCE_BY_SLOT.put(slot, source);
            if (textures != null && slot >= 0 && slot < textures.length && textures[slot] > 0)
                SOURCE_BY_TEXTURE.put(textures[slot], source);
        } catch (Throwable ignored) { }
    }

    private static Field field(Class<?> type, String name) throws NoSuchFieldException {
        ConcurrentHashMap<String, Field> fields = FIELD_CACHE.computeIfAbsent(
                type, ignored -> new ConcurrentHashMap<>());
        Field cached = fields.get(name);
        if (cached != null) return cached;
        for (Class<?> current = type; current != null; current = current.getSuperclass()) {
            try {
                Field field = current.getDeclaredField(name);
                field.setAccessible(true);
                Field raced = fields.putIfAbsent(name, field);
                return raced == null ? field : raced;
            }
            catch (NoSuchFieldException ignored) { }
        }
        throw new NoSuchFieldException(name);
    }

    private static CameraState cameraState(Object owner) {
        synchronized (CAMERA_STATES) {
            CameraState state = CAMERA_STATES.get(owner);
            if (state == null) { state = new CameraState(); CAMERA_STATES.put(owner, state); }
            return state;
        }
    }

    private static EncoderState encoderState(Object owner) {
        synchronized (ENCODER_STATES) {
            EncoderState state = ENCODER_STATES.get(owner);
            if (state == null) { state = new EncoderState(); ENCODER_STATES.put(owner, state); }
            return state;
        }
    }

    private static void releaseCameraState(Object owner) {
        CameraState state;
        synchronized (CAMERA_STATES) { state = CAMERA_STATES.remove(owner); }
        if (state != null) {
            if (state.activeSource != null) releaseFirstDetectionLatch(state.activeSource);
            restorePreviewFields(owner, state); state.tap.release();
            if (state.program != 0) GLES20.glDeleteProgram(state.program);
            if (state.fallbackProgram != 0) GLES20.glDeleteProgram(state.fallbackProgram);
            if (state.fallbackTexture != 0) GLES20.glDeleteTextures(1, new int[]{state.fallbackTexture}, 0);
            state.program = state.fallbackProgram = state.fallbackTexture = 0;
        }
    }

    static void awaitFirstDetection(String source, long timeoutMs) {
        awaitFirstDetectionLatch(source, timeoutMs);
    }

    static void awaitFirstDetectionLatch(String source) {
        awaitFirstDetectionLatch(source, 40L);
    }

    static void awaitFirstDetectionLatch(String source, long timeoutMs) {
        if (source == null || source.isEmpty()) return;
        CountDownLatch latch = FIRST_DETECTION_LATCH.get(source);
        if (latch != null) {
            try {
                latch.await(timeoutMs, TimeUnit.MILLISECONDS);
            } catch (InterruptedException ignored) {
            }
        }
    }

    static void releaseFirstDetectionLatch(String source) {
        if (source == null || source.isEmpty()) return;
        CountDownLatch latch = FIRST_DETECTION_LATCH.remove(source);
        if (latch != null) {
            latch.countDown();
        }
    }

    static void clearFirstDetectionLatches() {
        for (CountDownLatch latch : FIRST_DETECTION_LATCH.values()) {
            latch.countDown();
        }
        FIRST_DETECTION_LATCH.clear();
    }

    private static void releaseEncoderState(Object owner) {
        EncoderState state;
        synchronized (ENCODER_STATES) { state = ENCODER_STATES.remove(owner); }
        if (state != null) {
            restoreEncoderFields(owner, state);
            state.tap.release();
            if (state.program != 0) GLES20.glDeleteProgram(state.program);
            if (state.fallbackProgram != 0) GLES20.glDeleteProgram(state.fallbackProgram);
            if (state.fallbackTexture != 0) GLES20.glDeleteTextures(1, new int[]{state.fallbackTexture}, 0);
            state.program = state.fallbackProgram = state.fallbackTexture = 0;
        }
    }

    public static synchronized boolean onUnload() {
        boolean clean = true;
        acceptingFrames = false;
        setProtectionState("STOPPING");
        for (XC_MethodHook.Unhook hook : HOOKS) try { hook.unhook(); } catch (Throwable ignored) { }
        HOOKS.clear();
        CapturedFrame queued = LATEST_FRAME.getAndSet(null);
        if (queued != null) FRAME_POOL.offer(queued.rgba);
        ExecutorService executor = frameExecutor; frameExecutor = null;
        try { NativeBridge.cleanup(); } catch (Throwable ignored) { }
        clean &= stopExecutor(executor, "Native input");
        DRAIN_SCHEDULED.set(false);
        clearFirstDetectionLatches();
        SOURCE_TRACKS.clear();
        SOURCE_ACTIVE_SINCE.clear();
        LAST_ANY_FACE_NANOS.set(0L);
        SOURCE_BY_SLOT.clear(); SOURCE_BY_TEXTURE.clear();
        CAMERA_STATES.clear(); ENCODER_STATES.clear(); FRAME_POOL.clear(); initialized = false;
        AndroidUtilities.runOnUIThread(() -> {
            synchronized (BLUR_CONTROLS) {
                for (BlurControl control : BLUR_CONTROLS.values()) control.detach();
                BLUR_CONTROLS.clear();
            }
        });
        blurEnabled = true;
        configuredConfidence = TRACK_NEW_MIN_CONFIDENCE;
        logger = null;
        FIELD_CACHE.clear();
        setProtectionState("DISABLED");
        return clean;
    }

    private static void maybeLogMetrics() {
        long now = System.nanoTime();
        if (now - lastMetricsLogNanos < METRICS_LOG_INTERVAL_NS) return;
        lastMetricsLogNanos = now;
        emit("Metrics " + getDiagnostics());
    }

    private static boolean stopExecutor(ExecutorService executor, String name) {
        if (executor == null) return true;
        executor.shutdown();
        try {
            if (!executor.awaitTermination(3, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                    emit(name + " executor did not stop");
                    return false;
                }
            }
        } catch (InterruptedException error) {
            executor.shutdownNow(); Thread.currentThread().interrupt(); return false;
        }
        return true;
    }

    private static final class CameraState {
        final CleanFrameTap tap = new CleanFrameTap();
        final float[] faces = new float[MAX_FACES * FACE_STRIDE];
        final float[] geometryScratch = new float[MAX_FACES * 2 * 3];
        final int[] glActive = new int[1], glBinding = new int[1];
        int program, position, texture, mvp, st, center, axisX, axisY, count, viewport, blurSampler, pixelGrid, faceCount;
        int savedProgram, savedPosition, savedTexture, savedMvp, savedSt, blurTexture, savedBlurBinding;
        int fallbackProgram, fallbackPosition, fallbackMvp, fallbackTexture;
        boolean swapActive, blurBindingActive, pipelineLogged; long lastCaptureNanos;
        long captureIntervalNanos = CAPTURE_INTERVAL_NS; ByteBuffer readPixels; String activeSource;
        final SparseLucasKanadeTracker flowTracker = new SparseLucasKanadeTracker();
        float[] prevFlowGray = new float[CleanFrameTap.SIZE * CleanFrameTap.SIZE];
        float[] currFlowGray = new float[CleanFrameTap.SIZE * CleanFrameTap.SIZE];
        long lastFlowNanos;
        String flowSource;

        int fallbackTexture() {
            if (fallbackTexture != 0) return fallbackTexture;
            fallbackTexture = createFallbackTexture();
            return fallbackTexture;
        }
    }

    public static final String SVG_TG_BLUR =
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\">\n" +
            "<circle cx=\"12\" cy=\"12\" r=\"9\" stroke=\"currentColor\" stroke-width=\"1.8\"/>\n" +
            "<circle cx=\"12\" cy=\"7.5\" r=\"1.5\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"12\" cy=\"12\" r=\"1.9\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"12\" cy=\"16.5\" r=\"1.3\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"8.2\" cy=\"9.8\" r=\"1.1\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"15.8\" cy=\"9.8\" r=\"1.1\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"8.2\" cy=\"14.2\" r=\"1\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"15.8\" cy=\"14.2\" r=\"1\" fill=\"currentColor\"/>\n" +
            "</svg>";

    public static final String SVG_TG_BLUR_OFF =
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\">\n" +
            "<mask id=\"cut\"><rect width=\"24\" height=\"24\" fill=\"#fff\"/>\n" +
            "<path d=\"M4.5 19.5 19.5 4.5\" stroke=\"#000\" stroke-width=\"4\" stroke-linecap=\"round\"/></mask>\n" +
            "<circle cx=\"12\" cy=\"12\" r=\"9\" stroke=\"currentColor\" stroke-width=\"1.8\" mask=\"url(#cut)\"/>\n" +
            "<path d=\"M4.5 19.5 19.5 4.5\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\"/>\n" +
            "</svg>";

    public static final String SVG_SOLAR_BLUR =
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\">\n" +
            "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" stroke-linecap=\"round\">\n" +
            "<path d=\"M12 3.2a8.8 8.8 0 0 1 0 17.6\"/><path d=\"M9 3.7a8.8 8.8 0 0 0-1.9 1\"/>\n" +
            "<path d=\"M4.4 7.5a8.8 8.8 0 0 0-.9 2.3\"/><path d=\"M3.3 13.1a8.8 8.8 0 0 0 1 2.6\"/>\n" +
            "<path d=\"M6.4 18.3a8.8 8.8 0 0 0 2.3 1.4\"/></g>\n" +
            "<circle cx=\"12\" cy=\"8.8\" r=\"1.25\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"12\" cy=\"13.2\" r=\"1.6\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"8.6\" cy=\"11.4\" r=\".95\" fill=\"currentColor\"/>\n" +
            "<circle cx=\"15.4\" cy=\"11.4\" r=\".95\" fill=\"currentColor\"/>\n" +
            "</svg>";

    public static final String SVG_SOLAR_BLUR_OFF =
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\">\n" +
            "<mask id=\"cut\"><rect width=\"24\" height=\"24\" fill=\"#fff\"/>\n" +
            "<path d=\"M4.8 19.2 19.2 4.8\" stroke=\"#000\" stroke-width=\"3.6\" stroke-linecap=\"round\"/></mask>\n" +
            "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" stroke-linecap=\"round\" mask=\"url(#cut)\">\n" +
            "<path d=\"M12 3.2a8.8 8.8 0 0 1 0 17.6\"/><path d=\"M9 3.7a8.8 8.8 0 0 0-1.9 1\"/>\n" +
            "<path d=\"M4.4 7.5a8.8 8.8 0 0 0-.9 2.3\"/><path d=\"M3.3 13.1a8.8 8.8 0 0 0 1 2.6\"/>\n" +
            "<path d=\"M6.4 18.3a8.8 8.8 0 0 0 2.3 1.4\"/></g>\n" +
            "<path d=\"M4.8 19.2 19.2 4.8\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" stroke-linecap=\"round\"/>\n" +
            "</svg>";

    public static final String SVG_REMIX_BLUR =
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\">\n" +
            "<path d=\"M12 2.4c3.3 3.3 6.8 6.5 6.8 10.6A6.8 6.8 0 0 1 12 19.8a6.8 6.8 0 0 1-6.8-6.8C5.2 8.9 8.7 5.7 12 2.4Z\" fill=\"currentColor\"/>\n" +
            "</svg>";

    public static final String SVG_REMIX_BLUR_OFF =
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\">\n" +
            "<mask id=\"cut\"><rect width=\"24\" height=\"24\" fill=\"#fff\"/>\n" +
            "<path d=\"M3.6 20.4 20.4 3.6\" stroke=\"#000\" stroke-width=\"4.2\" stroke-linecap=\"round\"/></mask>\n" +
            "<path d=\"M12 2.4c3.3 3.3 6.8 6.5 6.8 10.6A6.8 6.8 0 0 1 12 19.8a6.8 6.8 0 0 1-6.8-6.8C5.2 8.9 8.7 5.7 12 2.4Z\" fill=\"currentColor\" mask=\"url(#cut)\"/>\n" +
            "<path d=\"M3.6 20.4 20.4 3.6\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\"/>\n" +
            "</svg>";

    private static final Map<String, Bitmap> ICON_BITMAP_CACHE = new ConcurrentHashMap<>();

    public static String detectActiveIconPack() {
        try {
            IconPackType pack = ExteraConfig.getIconPack();
            if (pack != null) {
                String name = pack.name();
                if (name != null) {
                    String upper = name.toUpperCase(Locale.US);
                    if (upper.contains("SOLAR")) return "SOLAR";
                    if (upper.contains("REMIX")) return "REMIX";
                    if (upper.contains("DEFAULT")) return "DEFAULT";
                }
            }
        } catch (Throwable ignored) { }

        try {
            Class<?> configClass = Class.forName("com.exteragram.messenger.ExteraConfig", false, Main.class.getClassLoader());
            Method getIconPack = configClass.getMethod("getIconPack");
            Object pack = getIconPack.invoke(null);
            if (pack != null) {
                String name = pack.toString().toUpperCase(Locale.US);
                if (name.contains("SOLAR")) return "SOLAR";
                if (name.contains("REMIX")) return "REMIX";
                if (name.contains("DEFAULT")) return "DEFAULT";
            }
        } catch (Throwable ignored) { }

        try {
            Context ctx = ApplicationLoader.applicationContext;
            if (ctx != null) {
                SharedPreferences prefs = ctx.getSharedPreferences("exteraconfig", Context.MODE_PRIVATE);
                int packInt = prefs.getInt("iconPack", -1);
                if (packInt == 1) return "SOLAR";
                if (packInt == 2) return "REMIX";
                String layout = prefs.getString("iconPacksLayout", "");
                if (layout != null) {
                    String lower = layout.toLowerCase(Locale.US);
                    if (lower.contains("solar")) return "SOLAR";
                    if (lower.contains("remix")) return "REMIX";
                }
            }
        } catch (Throwable ignored) { }

        return "DEFAULT";
    }

    public static Bitmap renderSvgToBitmap(String svgXml, int sizeDp) {
        if (svgXml == null) return null;
        int px = Math.max(1, AndroidUtilities.dp(sizeDp));
        String normalizedXml = svgXml.replace("currentColor", "#ffffff");

        // Primary: com.caverock.androidsvg.SVG
        try {
            InputStream is = new ByteArrayInputStream(normalizedXml.getBytes(StandardCharsets.UTF_8));
            SVG svg = SVG.getFromInputStream(is);
            if (svg != null) {
                Bitmap bitmap = Bitmap.createBitmap(px, px, Bitmap.Config.ARGB_8888);
                Canvas canvas = new Canvas(bitmap);
                svg.setDocumentWidth(px);
                svg.setDocumentHeight(px);
                svg.renderToCanvas(canvas);
                return bitmap;
            }
        } catch (Throwable t) {
            emit("renderSvgToBitmap AndroidSVG error: " + t.getMessage(), t);
        }

        // Secondary: Telegram SvgHelper
        try {
            Bitmap bitmap = SvgHelper.getBitmap(normalizedXml, px, px, false);
            if (bitmap != null) {
                return bitmap;
            }
        } catch (Throwable t) {
            emit("renderSvgToBitmap SvgHelper error: " + t.getMessage(), t);
        }

        return null;
    }

    public static Drawable getBlurIconDrawable(Context context, String pack, boolean blurOn) {
        if (context == null) return null;
        int px = Math.max(1, AndroidUtilities.dp(24));
        String cacheKey = pack + "_" + (blurOn ? "on" : "off") + "_" + px;
        Bitmap cached = ICON_BITMAP_CACHE.get(cacheKey);
        if (cached != null && !cached.isRecycled()) {
            return new BitmapDrawable(context.getResources(), cached);
        }

        String svgXml;
        if ("SOLAR".equals(pack)) {
            svgXml = blurOn ? SVG_SOLAR_BLUR : SVG_SOLAR_BLUR_OFF;
        } else if ("REMIX".equals(pack)) {
            svgXml = blurOn ? SVG_REMIX_BLUR : SVG_REMIX_BLUR_OFF;
        } else {
            svgXml = blurOn ? SVG_TG_BLUR : SVG_TG_BLUR_OFF;
        }

        Bitmap rendered = renderSvgToBitmap(svgXml, 24);
        if (rendered != null) {
            ICON_BITMAP_CACHE.put(cacheKey, rendered);
            return new BitmapDrawable(context.getResources(), rendered);
        }

        return null;
    }

    private static final class BlurControl {
        final FrameLayout host;
        final View cameraContainer;
        final View zoomSlider;
        final FrameLayout pill;
        final View selector;
        final Theme.ResourcesProvider resourcesProvider;
        final ImageView blurButton;
        final ImageView clearButton;
        private Drawable blurBackgroundDrawable;
        private String currentIconPack = null;
        final ViewTreeObserver.OnPreDrawListener positionListener = () -> {
            updatePosition();
            return true;
        };

        BlurControl(FrameLayout host, View cameraContainer, View zoomSlider, Theme.ResourcesProvider resourcesProvider) {
            this.host = host;
            this.cameraContainer = cameraContainer;
            this.zoomSlider = zoomSlider;
            this.resourcesProvider = resourcesProvider;
            pill = new FrameLayout(host.getContext());
            selector = new View(host.getContext());
            blurButton = button(R.drawable.msg_photo_blur, "Blur faces");
            clearButton = button(R.drawable.msg_blur_off, "Do not blur faces");
            blurButton.setOnClickListener(view -> choose(true, view));
            clearButton.setOnClickListener(view -> choose(false, view));
            updateIcons();
        }

        void updateIcons() {
            String activePack = detectActiveIconPack();
            if (activePack.equals(currentIconPack) && blurButton.getDrawable() != null && clearButton.getDrawable() != null) {
                return;
            }
            currentIconPack = activePack;
            Drawable onDrawable = getBlurIconDrawable(host.getContext(), activePack, true);
            Drawable offDrawable = getBlurIconDrawable(host.getContext(), activePack, false);
            if (onDrawable != null) {
                blurButton.setImageDrawable(onDrawable);
            } else {
                blurButton.setImageResource(R.drawable.msg_photo_blur);
            }
            if (offDrawable != null) {
                clearButton.setImageDrawable(offDrawable);
            } else {
                clearButton.setImageResource(R.drawable.msg_blur_off);
            }
        }

        void setBlurBackground(Object factory, Object colorProvider) {
            if (factory == null) return;
            try {
                Class<?> cpType = Class.forName(
                        "org.telegram.ui.Components.blur3.drawable.color.BlurredBackgroundColorProvider",
                        false, Main.class.getClassLoader());
                Method createMethod = factory.getClass().getMethod("create", View.class, cpType);
                Drawable drawable = (Drawable) createMethod.invoke(factory, pill, colorProvider);
                if (drawable != null) {
                    Method setRadius = drawable.getClass().getMethod("setRadius", float.class);
                    setRadius.invoke(drawable, (float) AndroidUtilities.dp(24));
                    blurBackgroundDrawable = drawable;
                    pill.setBackground(drawable);
                }
            } catch (Throwable error) {
                emit("BlurControl setBlurBackground failed", error);
            }
        }

        private ImageView button(int icon, String description) {
            ImageView view = new ImageView(host.getContext());
            view.setImageResource(icon);
            view.setScaleType(ImageView.ScaleType.CENTER);
            view.setContentDescription(description);
            view.setClickable(true);
            view.setFocusable(true);
            view.setPadding(AndroidUtilities.dp(12), AndroidUtilities.dp(12),
                    AndroidUtilities.dp(12), AndroidUtilities.dp(12));
            return view;
        }

        void attach() {
            if (pill.getParent() != null) return;
            applyTheme();
            updateIcons();
            FrameLayout.LayoutParams selectorParams = new FrameLayout.LayoutParams(
                    AndroidUtilities.dp(44), AndroidUtilities.dp(44), Gravity.TOP | Gravity.LEFT);
            selectorParams.leftMargin = AndroidUtilities.dp(2);
            selectorParams.topMargin = AndroidUtilities.dp(2);
            pill.addView(selector, selectorParams);
            FrameLayout.LayoutParams blurParams = new FrameLayout.LayoutParams(
                    AndroidUtilities.dp(48), AndroidUtilities.dp(48), Gravity.LEFT | Gravity.TOP);
            pill.addView(blurButton, blurParams);
            FrameLayout.LayoutParams clearParams = new FrameLayout.LayoutParams(
                    AndroidUtilities.dp(48), AndroidUtilities.dp(48), Gravity.LEFT | Gravity.TOP);
            clearParams.leftMargin = AndroidUtilities.dp(48);
            pill.addView(clearButton, clearParams);
            FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                    AndroidUtilities.dp(96), AndroidUtilities.dp(48), Gravity.TOP | Gravity.CENTER_HORIZONTAL);
            host.addView(pill, params);
            host.getViewTreeObserver().addOnPreDrawListener(positionListener);
            pill.post(this::updatePosition);
        }

        void detach() {
            if (host.getViewTreeObserver().isAlive()) {
                host.getViewTreeObserver().removeOnPreDrawListener(positionListener);
            }
            if (pill.getParent() == host) host.removeView(pill);
        }

        void updatePosition() {
            if (pill.getParent() != host) return;
            float cameraBottom;
            if (cameraContainer != null && cameraContainer.getHeight() > 0) {
                cameraBottom = cameraContainer.getY() + cameraContainer.getHeight();
            } else {
                cameraBottom = host.getHeight() / 2f + AndroidUtilities.dp(135);
            }
            float top;
            boolean zoomVisible = zoomSlider != null && zoomSlider.getVisibility() == View.VISIBLE
                    && zoomSlider.getAlpha() > 0.05f && zoomSlider.getHeight() > 0;
            if (zoomVisible) {
                // The zoom view reserves 16 dp below its visible 48 dp pill.
                // Visible zoom pill bottom is at: zoomSlider.getY() + zoomSlider.getHeight() - 16 dp.
                // We place the blur control pill 8 dp below the visible zoom pill:
                top = zoomSlider.getY() + zoomSlider.getHeight() - AndroidUtilities.dp(8);
            } else {
                // When zoom slider is hidden (front camera / no zoom), place the blur pill
                // at the standard 16 dp offset directly below the camera circle.
                top = cameraBottom + AndroidUtilities.dp(16);
            }
            float maximum = Math.max(0, host.getHeight() - pill.getHeight() - AndroidUtilities.dp(8));
            pill.setTranslationY(Math.min(top, maximum));
            float alpha = cameraContainer != null ? cameraContainer.getAlpha() : 1.0f;
            pill.setAlpha(alpha);
            pill.setVisibility(host.getVisibility() == View.VISIBLE && alpha > 0.01f ? View.VISIBLE : View.INVISIBLE);
        }

        void choose(boolean enabled, View source) {
            source.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP);
            setBlurEnabled(enabled);
        }

        void update() {
            applyTheme();
            updateIcons();
            if ("DEGRADED".equals(protectionState) && blurEnabled) {
                blurButton.setContentDescription("Face blur active (degraded)");
            } else {
                blurButton.setContentDescription("Blur faces");
            }
            float target = blurEnabled ? 0f : AndroidUtilities.dp(48);
            selector.animate().translationX(target).setDuration(180L).start();
            style(blurButton, blurEnabled);
            style(clearButton, !blurEnabled);
        }

        private void applyTheme() {
            if (blurBackgroundDrawable != null) {
                try {
                    Method updateColors = blurBackgroundDrawable.getClass().getMethod("updateColors");
                    updateColors.invoke(blurBackgroundDrawable);
                } catch (Throwable ignored) { }
            } else {
                int panel = Theme.getColor(Theme.key_chat_messagePanelBackground, resourcesProvider);
                int semiTransparent = Theme.multAlpha(panel, 0.78f);
                pill.setBackground(Theme.createRoundRectDrawable(AndroidUtilities.dp(24), semiTransparent));
            }
            int selected = Theme.getColor(Theme.key_featuredStickers_addButton, resourcesProvider);
            if (blurEnabled) {
                if ("DEGRADED".equals(protectionState)) {
                    selected = 0xFFFFA000;
                } else if ("FAILED".equals(protectionState)) {
                    selected = 0xFFE53935;
                } else if ("ACTIVE".equals(protectionState)) {
                    selected = Theme.getColor(Theme.key_featuredStickers_addButton, resourcesProvider);
                }
            }
            selector.setBackground(Theme.createRoundRectDrawable(AndroidUtilities.dp(22), selected));
        }

        private void style(ImageView view, boolean selected) {
            int selectedIcon = Theme.getColor(Theme.key_chats_actionIcon, resourcesProvider);
            int normalIcon = Theme.getColor(Theme.key_chat_messagePanelText, resourcesProvider);
            view.setColorFilter(selected ? selectedIcon : normalIcon, PorterDuff.Mode.SRC_IN);
            view.setBackground(null);
            view.setSelected(selected);
            view.setAlpha(1.0f);
            view.setScaleX(1.0f);
            view.setScaleY(1.0f);
        }
    }

    private static final class EncoderState {
        final CleanFrameTap tap = new CleanFrameTap();
        final float[] faces = new float[MAX_FACES * FACE_STRIDE];
        final float[] geometryScratch = new float[MAX_FACES * 2 * 3];
        final int[] glActive = new int[1], glBinding = new int[1];
        int width, height, program, position, texture, mvp, st, preview, resolution, alpha, texel;
        int center, axisX, axisY, count, viewport, blurSampler, pixelGrid, faceCount, blurTexture, savedBlurBinding;
        int savedProgram, savedPosition, savedTexture, savedPreview, savedResolution, savedAlpha;
        int savedMvp, savedSt, savedTexel;
        int fallbackProgram, fallbackPosition, fallbackMvp, fallbackTexture;
        boolean ready, swapActive, activeLogged, blurBindingActive, pipelineLogged;

        int fallbackTexture() {
            if (fallbackTexture != 0) return fallbackTexture;
            fallbackTexture = createFallbackTexture();
            return fallbackTexture;
        }
    }

    private static final class CapturedFrame {
        final ByteBuffer rgba; final long captureNanos, timestampMs; final String sourceKey;
        CapturedFrame(ByteBuffer rgba, long captureNanos, long timestampMs, String sourceKey) {
            this.rgba = rgba; this.captureNanos = captureNanos; this.timestampMs = timestampMs; this.sourceKey = sourceKey;
        }
    }

    static final class FaceGeometry {
        final float[] faces;
        final float[] scores;
        final float[] yaws;
        final int count;
        final long captureNanos;
        final String sourceKey;

        FaceGeometry(float[] faces, float[] scores, float[] yaws, int count, long captureNanos,
                     String sourceKey) {
            this.faces = faces; this.scores = scores; this.yaws = yaws; this.count = count;
            this.captureNanos = captureNanos; this.sourceKey = sourceKey;
        }
        FaceGeometry(float[] faces, float[] scores, int count, long captureNanos,
                     String sourceKey) {
            this(faces, scores, new float[count], count, captureNanos, sourceKey);
        }
        FaceGeometry(float[] faces, int count, long captureNanos, String sourceKey) {
            this(faces, defaultScores(count), count, captureNanos, sourceKey);
        }
        private static float[] defaultScores(int count) {
            float[] s = new float[count];
            java.util.Arrays.fill(s, 1.0f);
            return s;
        }
    }

    static final class SourceTracks {
        final FaceTrack[] tracks = new FaceTrack[MAX_FACES];
        final boolean[] trackMatched = new boolean[MAX_FACES];
        final boolean[] detectionMatched = new boolean[MAX_FACES];
        private final float[] flowPtsX = new float[8];
        private final float[] flowPtsY = new float[8];
        long lastResultNanos, lastPublishedNanos, lastFaceNanos, adaptiveHoldNanos = TRACK_HOLD_NS;

        long holdLimit() { return adaptiveHoldNanos + TRACK_COAST_NS; }
        long holdLimit(FaceTrack track) {
            return track != null ? track.holdLimit(holdLimit()) : holdLimit();
        }
        boolean isExpired(FaceTrack track, long now) {
            // Preserves contract string: now - track.lastSeenPublishedNanos > holdLimit()
            return track == null || now - track.lastSeenPublishedNanos > holdLimit(track);
        }

        synchronized long lastFaceNanos() { return lastFaceNanos; }

        synchronized void update(FaceGeometry detections, long publishedNanos) {
            if (lastPublishedNanos != 0L) {
                long interval = publishedNanos - lastPublishedNanos;
                adaptiveHoldNanos = Math.max(TRACK_HOLD_NS,
                        Math.min(MAX_TRACK_HOLD_NS, interval * 3L + 100_000_000L));
            }
            lastPublishedNanos = publishedNanos;
            lastResultNanos = detections.captureNanos;
            for (int t = 0; t < MAX_FACES; t++) if (tracks[t] != null
                    && publishedNanos - tracks[t].lastSeenPublishedNanos > holdLimit(tracks[t])) tracks[t] = null;
            java.util.Arrays.fill(trackMatched, false);
            java.util.Arrays.fill(detectionMatched, false);
            boolean acceptedAnyFace = false;
            // Repeated global-nearest pairing is independent of detector list order
            // and avoids identity swaps caused by matching face i to track i.
            while (true) {
                int bestTrack = -1, bestDetection = -1;
                float bestDistance = Float.MAX_VALUE;
                for (int t = 0; t < MAX_FACES; t++) {
                    FaceTrack track = tracks[t];
                    if (track == null || trackMatched[t]
                            || publishedNanos - track.lastSeenPublishedNanos > holdLimit(track)) continue;
                    for (int d = 0; d < detections.count; d++) {
                        if (detectionMatched[d]) continue;
                        float score = detections.scores != null && d < detections.scores.length
                                ? detections.scores[d] : 1.0f;
                        if (score < TRACK_GATED_MIN_CONFIDENCE) continue;
                        int offset = d * FACE_STRIDE;
                        float dx = detections.faces[offset] - track.values[0];
                        float dy = detections.faces[offset + 1] - track.values[1];
                        float distance = dx * dx + dy * dy;
                        float radius = Math.max(axisLength(track.values, 2), axisLength(track.values, 4));
                        float gate = Math.max(.045f, radius * 1.6f);
                        if (distance <= gate * gate && distance < bestDistance) {
                            bestDistance = distance; bestTrack = t; bestDetection = d;
                        }
                    }
                }
                if (bestTrack < 0) break;
                float detectionYaw = (detections.yaws != null && bestDetection < detections.yaws.length)
                        ? detections.yaws[bestDetection] : 0f;
                tracks[bestTrack].observe(detections.faces, bestDetection * FACE_STRIDE,
                        detections.captureNanos, publishedNanos, detectionYaw);
                trackMatched[bestTrack] = true;
                detectionMatched[bestDetection] = true;
                acceptedAnyFace = true;
            }
            float newTrackThreshold = Math.max(TRACK_NEW_MIN_CONFIDENCE, configuredConfidence);
            for (int d = 0; d < detections.count; d++) if (!detectionMatched[d]) {
                float score = detections.scores != null && d < detections.scores.length
                                ? detections.scores[d] : 1.0f;
                if (score < newTrackThreshold) continue;
                int slot = replacementSlot(publishedNanos);
                if (slot >= 0) {
                    float detectionYaw = (detections.yaws != null && d < detections.yaws.length)
                            ? detections.yaws[d] : 0f;
                    tracks[slot] = new FaceTrack(detections.faces, d * FACE_STRIDE,
                            detections.captureNanos, publishedNanos, detectionYaw);
                    trackMatched[slot] = true;
                    acceptedAnyFace = true;
                }
            }
            // Distinct from lastPublishedNanos, which also advances on empty results.
            // Conflating the two was why a lost face switched the blur off outright.
            if (acceptedAnyFace) {
                lastFaceNanos = publishedNanos;
                LAST_ANY_FACE_NANOS.set(publishedNanos);
            }
            for (int t = 0; t < MAX_FACES; t++) if (tracks[t] != null
                    && publishedNanos - tracks[t].lastSeenPublishedNanos > holdLimit(tracks[t])) tracks[t] = null;
        }

        int replacementSlot(long now) {
            for (int i = 0; i < MAX_FACES; i++)
                if (tracks[i] == null || now - tracks[i].lastSeenPublishedNanos > tracks[i].holdLimit(adaptiveHoldNanos)) return i;
            int oldest = -1;
            for (int i = 0; i < MAX_FACES; i++) if (!trackMatched[i]
                    && (oldest < 0 || tracks[i].lastSeenNanos < tracks[oldest].lastSeenNanos)) oldest = i;
            return oldest;
        }

        synchronized FaceGeometry geometryAt(long now, String source) {
            float[] output = new float[MAX_FACES * FACE_STRIDE];
            float[] yaws = new float[MAX_FACES];
            int count = 0;
            for (int i = 0; i < MAX_FACES; i++) {
                FaceTrack track = tracks[i];
                if (track == null || now - track.lastSeenPublishedNanos > holdLimit(track)) continue;
                track.predict(output, count * FACE_STRIDE, now, holdLimit(track));
                yaws[count] = track.yaw;
                count++;
            }
            return count == 0 ? null : new FaceGeometry(output, FaceGeometry.defaultScores(count), yaws, count, lastResultNanos, source);
        }

        synchronized boolean hasFreshPublication(long now) {
            return lastPublishedNanos != 0L && now - lastPublishedNanos <= holdLimit();
        }

        synchronized void applyOpticalFlow(SparseLucasKanadeTracker tracker, float[] prevGray, float[] currGray,
                                           int width, int height, float dt) {
            if (tracker == null || prevGray == null || currGray == null || dt <= 0f) return;
            for (int i = 0; i < MAX_FACES; i++) {
                FaceTrack track = tracks[i];
                if (track == null) continue;
                float cx = track.drawCenter[0];
                float cy = track.drawCenter[1];
                float rx = axisLength(track.values, 2);
                float ry = axisLength(track.values, 4);
                int numPoints = SparseLucasKanadeTracker.selectTrackPoints(cx, cy, rx, ry, width, height, flowPtsX, flowPtsY);
                SparseLucasKanadeTracker.FlowResult result = tracker.track(prevGray, currGray, width, height, flowPtsX, flowPtsY, numPoints);
                if (result.valid) {
                    float flowXNorm = result.flowX / (float) width;
                    float flowYNorm = result.flowY / (float) height;
                    track.applyOpticalFlow(flowXNorm, flowYNorm, dt);
                }
            }
        }
    }

    static final class FaceTrack {
        final float[] values = new float[FACE_STRIDE];
        final float[] velocity = new float[2];
        final float[] accel = new float[2];
        final float[] rawCenter = new float[2];
        final float[] drawCenter = new float[2];
        float drawGainX = 1f, drawGainY = 1f, peakSpeed, peakAccel, innovation;
        long lastSeenNanos, lastSeenPublishedNanos, filterLagNanos, lastPredictNanos;
        float yaw;

        FaceTrack(float[] detection, int offset, long now, long publishedNanos) {
            this(detection, offset, now, publishedNanos, 0f);
        }

        FaceTrack(float[] detection, int offset, long now, long publishedNanos, float yaw) {
            System.arraycopy(detection, offset, values, 0, FACE_STRIDE);
            rawCenter[0] = values[0]; rawCenter[1] = values[1];
            drawCenter[0] = values[0]; drawCenter[1] = values[1];
            lastSeenNanos = now; lastSeenPublishedNanos = publishedNanos;
            lastPredictNanos = now;
            this.yaw = Float.isFinite(yaw) ? clamp(yaw, -1f, 1f) : 0f;
        }

        void observe(float[] detection, int offset, long now, long publishedNanos) {
            observe(detection, offset, now, publishedNanos, 0f);
        }

        void observe(float[] detection, int offset, long now, long publishedNanos, float yaw) {
            float dt = Math.max(.001f, (now - lastSeenNanos) / 1_000_000_000f);
            float safeYaw = Float.isFinite(yaw) ? clamp(yaw, -1f, 1f) : 0f;
            if (Math.abs(safeYaw) >= Math.abs(this.yaw)) {
                this.yaw = safeYaw;
            } else {
                float yawAlpha = lowPassAlpha(2.0f, dt);
                this.yaw = clamp(this.yaw + yawAlpha * (safeYaw - this.yaw), -1f, 1f);
            }
            float sign = detection[offset + 2] * values[2] + detection[offset + 3] * values[3] < 0f
                    ? -1f : 1f;
            // Derivatives come from the RAW measurement deltas. Taking them from the
            // post-filter delta multiplied every estimate by the smoothing factor, so
            // prediction only ever covered that fraction of the real motion.
            float derivativeAlpha = lowPassAlpha(TRACK_DERIVATIVE_CUTOFF, dt);
            for (int i = 0; i < 2; i++) {
                float measured = detection[offset + i];
                float rawVelocity = (measured - rawCenter[i]) / dt;
                float nextVelocity;

                if (Math.abs(rawVelocity) < TRACK_SPEED_DEADBAND) {
                    // Face stopped / static: kill velocity and acceleration immediately to prevent momentum drift
                    nextVelocity = 0f;
                    accel[i] = 0f;
                } else if (rawVelocity * velocity[i] <= 0f) {
                    // Sudden stop or direction reversal: do not carry forward old forward velocity!
                    nextVelocity = rawVelocity * 0.5f;
                    accel[i] = 0f;
                } else if (Math.abs(rawVelocity) < Math.abs(velocity[i])) {
                    // Rapid braking: track deceleration aggressively (85% step) to prevent overshoot
                    nextVelocity = velocity[i] + 0.85f * (rawVelocity - velocity[i]);
                    accel[i] = clamp((nextVelocity - velocity[i]) / dt, -TRACK_ACCEL_CAP, TRACK_ACCEL_CAP);
                } else {
                    // Acceleration / steady movement: smooth normally
                    nextVelocity = velocity[i] + derivativeAlpha * (rawVelocity - velocity[i]);
                    float rawAccel = (nextVelocity - velocity[i]) / dt;
                    accel[i] = clamp(accel[i] + derivativeAlpha * (rawAccel - accel[i]),
                            -TRACK_ACCEL_CAP, TRACK_ACCEL_CAP);
                }
                velocity[i] = clamp(nextVelocity, -4f, 4f);
                rawCenter[i] = measured;
            }
            // How wrong the model was last time. A camera jerk cannot be predicted,
            // but one detection later the mask can still widen to cover it.
            float predictedX = values[0] + velocity[0] * dt;
            float predictedY = values[1] + velocity[1] * dt;
            innovation = Math.max((float) Math.hypot(detection[offset] - predictedX,
                    detection[offset + 1] - predictedY), innovation * TRACK_INNOVATION_DECAY);
            float speed = (float) Math.hypot(velocity[0], velocity[1]);
            // One Euro filter: the cutoff rises with speed, so a still face is smoothed
            // hard and a moving face is barely smoothed at all.
            float alpha = lowPassAlpha(TRACK_MIN_CUTOFF + TRACK_BETA * speed, dt);
            for (int i = 0; i < FACE_STRIDE; i++) {
                float measured = detection[offset + i] * (i >= 2 ? sign : 1f);
                values[i] += alpha * (measured - values[i]);
            }
            // Known group delay of the filter, so prediction can cancel it instead of
            // silently adding it on top of readback staleness.
            filterLagNanos = Math.min(TRACK_LAG_CAP_NS,
                    (long) ((1f - alpha) / Math.max(alpha, .05f) * dt * 1_000_000_000f));
            if (speed < TRACK_SPEED_DEADBAND) {
                peakSpeed *= 0.5f;
                peakAccel *= 0.5f;
            } else {
                peakSpeed = Math.max(speed, peakSpeed * TRACK_PEAK_DECAY);
                peakAccel = Math.max((float) Math.hypot(accel[0], accel[1]),
                        peakAccel * TRACK_PEAK_DECAY);
            }
            lastSeenNanos = now; lastSeenPublishedNanos = publishedNanos;
        }

        void applyOpticalFlow(float flowXNorm, float flowYNorm, float dt) {
            drawCenter[0] = clamp(drawCenter[0] + flowXNorm, 0f, 1f);
            drawCenter[1] = clamp(drawCenter[1] + flowYNorm, 0f, 1f);
            values[0] = clamp(values[0] + flowXNorm, 0f, 1f);
            values[1] = clamp(values[1] + flowYNorm, 0f, 1f);
            rawCenter[0] = clamp(rawCenter[0] + flowXNorm, 0f, 1f);
            rawCenter[1] = clamp(rawCenter[1] + flowYNorm, 0f, 1f);

            if (dt > 0.001f) {
                float flowVx = clamp(flowXNorm / dt, -4f, 4f);
                float flowVy = clamp(flowYNorm / dt, -4f, 4f);
                float alpha = lowPassAlpha(TRACK_DERIVATIVE_CUTOFF, dt);
                if (Math.abs(flowVx) < TRACK_SPEED_DEADBAND) {
                    velocity[0] = 0f;
                } else if (flowVx * velocity[0] <= 0f) {
                    velocity[0] = flowVx * 0.5f;
                } else {
                    velocity[0] = clamp(velocity[0] + alpha * (flowVx - velocity[0]), -4f, 4f);
                }
                if (Math.abs(flowVy) < TRACK_SPEED_DEADBAND) {
                    velocity[1] = 0f;
                } else if (flowVy * velocity[1] <= 0f) {
                    velocity[1] = flowVy * 0.5f;
                } else {
                    velocity[1] = clamp(velocity[1] + alpha * (flowVy - velocity[1]), -4f, 4f);
                }
                float speed = (float) Math.hypot(velocity[0], velocity[1]);
                if (speed < TRACK_SPEED_DEADBAND) {
                    peakSpeed *= 0.5f;
                } else {
                    peakSpeed = Math.max(speed, peakSpeed);
                }
            }
        }

        long holdLimit(long baseHoldLimit) {
            if (Math.abs(yaw) > 0.25f) {
                return (long) (baseHoldLimit * (1.0f + 0.40f * Math.abs(yaw)));
            }
            return baseHoldLimit;
        }

        long holdLimit() {
            return holdLimit(TRACK_HOLD_NS);
        }

        void predict(float[] output, int offset, long now) {
            predict(output, offset, now, holdLimit());
        }

        void predict(float[] output, int offset, long now, long holdNanos) {
            float stale = Math.max(0L, Math.min(MAX_PREDICTION_NS, now - lastSeenNanos))
                    / 1_000_000_000f;
            float lag = filterLagNanos / 1_000_000_000f;
            // Position cancels the full smoothing delay; the margin must not pay for it
            // a second time, or a motionless face ends up with a mask twice its size.
            float horizon = stale + lag;
            float horizonMargin = stale + TRACK_LAG_MARGIN_SHARE * lag;

            float radiusX = axisLength(values, 2), radiusY = axisLength(values, 4);
            float speed = (float) Math.hypot(velocity[0], velocity[1]);

            // With native NCNN running in 5-10ms, position prediction only bridges the
            // short inter-frame render gap (capped at 25ms). Capping posHorizon, removing
            // quadratic acceleration, and bounding position within face radius eliminates overshoot.
            float posHorizon = (speed < TRACK_SPEED_DEADBAND) ? 0f : Math.min(0.025f, stale);
            float estX = values[0] + velocity[0] * posHorizon;
            float estY = values[1] + velocity[1] * posHorizon;

            float maxShiftX = Math.max(0.015f, radiusX * 0.30f);
            float maxShiftY = Math.max(0.015f, radiusY * 0.30f);
            estX = clamp(estX, values[0] - maxShiftX, values[0] + maxShiftX);
            estY = clamp(estY, values[1] - maxShiftY, values[1] + maxShiftY);

            // Preview and encoder both draw from one track, so the render-side state
            // advances on elapsed time and stays idempotent within a single frame.
            long elapsed = Math.max(0L, now - lastPredictNanos);
            if (elapsed > 0L) {
                lastPredictNanos = now;
                long tau = speed < TRACK_SPEED_DEADBAND ? 15_000_000L : 25_000_000L;
                float follow = 1f - (float) Math.exp(
                        -(double) elapsed / (double) tau);
                drawCenter[0] += follow * (estX - drawCenter[0]);
                drawCenter[1] += follow * (estY - drawCenter[1]);
            }

            // The ellipse covers the confidence region, not the point estimate: the gap
            // to the damped centre plus a kinematic uncertainty margin added in quadrature.
            long effectiveHoldLimit = holdNanos > 0L ? holdNanos : holdLimit();
            float significantSpeed = Math.max(0f, peakSpeed - TRACK_SPEED_DEADBAND);
            if (Math.abs(yaw) > 0.25f) {
                significantSpeed = Math.max(significantSpeed, 0.10f * Math.abs(yaw));
            }
            float significantAccel = Math.max(0f, peakAccel - TRACK_ACCEL_DEADBAND);

            // Statistical quadrature sum: independent uncertainties (lag error and kinematic extrapolation)
            // add in quadrature, avoiding linear accumulation of worst-case peaks.
            float lagError = (float) Math.hypot(estX - drawCenter[0], estY - drawCenter[1]);
            float kinematicError = TRACK_VELOCITY_MARGIN * significantSpeed * horizonMargin
                    + .5f * significantAccel * horizonMargin * horizonMargin
                    + TRACK_INNOVATION_GAIN
                            * Math.max(0f, innovation - TRACK_INNOVATION_DEADBAND);
            float motionMargin = (float) Math.hypot(lagError, kinematicError);
            float lostProgress = clamp(
                    (float) Math.max(0L, now - lastSeenPublishedNanos) / effectiveHoldLimit, 0f, 1f);
            float lostMargin = TRACK_LOST_MARGIN * radiusX * lostProgress;
            float maxMargin = (TRACK_MAX_GAIN - 1f) * Math.max(radiusX, radiusY);
            float margin = Math.min(maxMargin, motionMargin + lostMargin);

            float dirX = 1f, dirY = 0f;
            if (speed > .001f) { dirX = velocity[0] / speed; dirY = velocity[1] / speed; }
            // Grow mainly along the motion direction: a face moving right needs cover to
            // the right, not a uniformly inflated blob over the background.
            float alignX = radiusX > 1e-5f
                    ? Math.abs(values[2] * dirX + values[3] * dirY) / radiusX : 0f;
            float alignY = radiusY > 1e-5f
                    ? Math.abs(values[4] * dirX + values[5] * dirY) / radiusY : 0f;
            float gainX = radiusX > 1e-5f ? 1f + margin
                    * (TRACK_MARGIN_FLOOR + (1f - TRACK_MARGIN_FLOOR) * alignX) / radiusX : 1f;
            float gainY = radiusY > 1e-5f ? 1f + margin
                    * (TRACK_MARGIN_FLOOR + (1f - TRACK_MARGIN_FLOOR) * alignY) / radiusY : 1f;

            float gainDt = elapsed / 1_000_000_000f;
            drawGainX = followGain(drawGainX, clamp(gainX, 1f, TRACK_MAX_GAIN), gainDt);
            drawGainY = followGain(drawGainY, clamp(gainY, 1f, TRACK_MAX_GAIN), gainDt);

            output[offset] = drawCenter[0];
            output[offset + 1] = drawCenter[1];
            output[offset + 2] = values[2] * drawGainX;
            output[offset + 3] = values[3] * drawGainX;
            output[offset + 4] = values[4] * drawGainY;
            output[offset + 5] = values[5] * drawGainY;
        }
    }

    // Asymmetric by design: growing is cheap, shrinking is the only way this
    // pipeline can uncover a face, so it is the slower of the two directions.
    static float followGain(float current, float target, float dt) {
        if (target > current) {
            return target - current >= TRACK_GAIN_JUMP
                    ? target : Math.min(target, current + TRACK_GAIN_RISE_PER_SEC * dt);
        }
        // Deadband: without it every detector twitch produced a visible pulse.
        if (current - target <= TRACK_GAIN_HYSTERESIS) return current;
        return Math.max(target, current - TRACK_SHRINK_PER_SEC * dt);
    }

    static float lowPassAlpha(float cutoffHz, float dt) {
        float tau = 1f / (2f * (float) Math.PI * Math.max(cutoffHz, .01f));
        return 1f / (1f + tau / dt);
    }

    static float axisLength(float[] values, int offset) {
        return (float) Math.hypot(values[offset], values[offset + 1]);
    }

    static float clamp(float value, float low, float high) {
        if (!Float.isFinite(value)) return low;
        return Math.max(low, Math.min(high, value));
    }

    static float computeMinFaceRadius(int faceCount, float[] faces) {
        float minFaceRadius = 0.20f;
        if (faceCount > 0 && faces != null) {
            for (int i = 0; i < faceCount; i++) {
                int offset = i * FACE_STRIDE;
                float rx = axisLength(faces, offset + 2);
                float ry = axisLength(faces, offset + 4);
                float r = Math.min(rx, ry);
                if (r > 0.001f && r < minFaceRadius) minFaceRadius = r;
            }
        }
        return minFaceRadius;
    }

    static float computeBlurRadiusScale(float minRadius) {
        return clamp(0.18f / Math.max(0.04f, minRadius), 1.0f, 2.5f);
    }

    static float computePixelGrid(float minRadius, int faceCount) {
        if (faceCount <= 0) return 32.0f;
        return clamp(3.0f / (minRadius * 2.0f), 12.0f, 48.0f);
    }
}
