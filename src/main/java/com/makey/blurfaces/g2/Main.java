package com.makey.blurfaces.g2;

import android.graphics.SurfaceTexture;
import android.graphics.PorterDuff;
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

import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult;

import java.io.FileInputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.Future;
import java.util.function.Consumer;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedBridge;
import org.telegram.messenger.AndroidUtilities;
import org.telegram.messenger.ApplicationLoader;
import org.telegram.messenger.R;
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
    private static final int MAX_FACES = 4;
    private static final int FACE_STRIDE = 6;
    private static final long CAPTURE_INTERVAL_NS = 33_333_333L;
    private static final long TRACK_HOLD_NS = 350_000_000L;
    private static final long MAX_TRACK_HOLD_NS = 1_200_000_000L;
    private static final long MAX_PREDICTION_NS = 120_000_000L;
    private static final float TRACK_SMOOTHING = 0.55f;
    private static final long METRICS_LOG_INTERVAL_NS = 5_000_000_000L;
    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;
    // MediaPipe's canonical face oval. PCA makes roll stable while extrema size the mask.
    private static final int[] OVAL = {10,338,297,332,284,251,389,356,454,323,361,288,397,
            365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109};

    private static final List<XC_MethodHook.Unhook> HOOKS = new ArrayList<>();
    private static final Map<Object, CameraState> CAMERA_STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, CameraState>());
    private static final Map<Object, EncoderState> ENCODER_STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, EncoderState>());
    private static final Map<Object, BlurControl> BLUR_CONTROLS =
            Collections.synchronizedMap(new WeakHashMap<Object, BlurControl>());
    private static final Map<Integer, String> SOURCE_BY_SLOT = new ConcurrentHashMap<>();
    private static final Map<Integer, String> SOURCE_BY_TEXTURE = new ConcurrentHashMap<>();
    private static final Map<String, Long> SOURCE_ACTIVE_SINCE = new ConcurrentHashMap<>();
    private static final Map<String, SourceTracks> SOURCE_TRACKS = new ConcurrentHashMap<>();
    private static final AtomicReference<CapturedFrame> LATEST_FRAME = new AtomicReference<>();
    private static final AtomicBoolean DRAIN_SCHEDULED = new AtomicBoolean();
    private static final ArrayBlockingQueue<ByteBuffer> FRAME_POOL = new ArrayBlockingQueue<>(3);
    private static final ConcurrentHashMap<Class<?>, ConcurrentHashMap<String, Field>> FIELD_CACHE =
            new ConcurrentHashMap<>();
    private static final AtomicLong RECONFIGURE_GENERATION = new AtomicLong();

    private static volatile boolean initialized;
    private static volatile boolean acceptingFrames;
    private static volatile boolean blurEnabled = true;
    private static volatile int roundVideoResolutionOverride;
    private static volatile float faceMaskScale = 1.0f;
    private static volatile int maskMode;
    private static volatile String protectionState = "DISABLED";
    private static FaceLandmarker landmarker;
    private static FaceDetector detector;
    private static boolean liteModel;
    private static ExecutorService frameExecutor;
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
            "uniform vec2 uFaceAxisY[" + MAX_FACES + "];uniform int uFaceCount;uniform vec2 uViewport;uniform float uMaskScale;uniform int uMaskMode;" +
            "vec2 buv(){return clamp(gl_FragCoord.xy/uViewport,0.0,1.0);}" +
            "void main(){vec4 src=texture2D(sTexture,vTextureCoord);if(uViewport.x<1.0){gl_FragColor=src;return;}" +
            "if(uFaceCount<0){gl_FragColor=texture2D(sBlurTexture,buv());return;}if(uFaceCount==0){gl_FragColor=src;return;}" +
            "vec2 p=vec2(gl_FragCoord.x/uViewport.x,1.0-gl_FragCoord.y/uViewport.y);float m=0.0;" +
            "for(int i=0;i<" + MAX_FACES + ";++i){if(i>=uFaceCount)break;vec2 d=p-uFaceCenter[i];vec2 x=uFaceAxisX[i],y=uFaceAxisY[i];" +
             "float z=x.x*y.y-x.y*y.x;if(abs(z)<.000001)continue;vec2 l=vec2((d.x*y.y-d.y*y.x)/z,(-d.x*x.y+d.y*x.x)/z);" +
             "float q=sqrt(max(dot(l,l),0.0))/max(uMaskScale,.1);m=max(m,clamp(1.0-smoothstep(.84,1.04,q),0.0,1.0));}if(m<=0.0){gl_FragColor=src;return;}" +
            "vec2 uv=buv();if(uMaskMode==1)uv=(floor(uv*32.0)+.5)/32.0;vec4 protectedColor=uMaskMode==2?vec4(.03,.03,.03,1.0):texture2D(sBlurTexture,uv);gl_FragColor=mix(src,protectedColor,m);}";
    private static final String ENCODER_FS =
            "#extension GL_OES_EGL_image_external : require\nprecision highp float;varying vec2 vTextureCoord;" +
            "uniform samplerExternalOES sTexture;uniform sampler2D sBlurTexture;uniform vec2 preview;uniform vec2 resolution;uniform float alpha;uniform vec2 texelSize;" +
            "uniform vec2 uFaceCenter[" + MAX_FACES + "];uniform vec2 uFaceAxisX[" + MAX_FACES + "];" +
            "uniform vec2 uFaceAxisY[" + MAX_FACES + "];uniform int uFaceCount;uniform vec2 uViewport;uniform float uMaskScale;uniform int uMaskMode;" +
            "vec2 buv(){return clamp(gl_FragCoord.xy/uViewport,0.0,1.0);}" +
            "void main(){vec4 src=texture2D(sTexture,vTextureCoord);if(uViewport.x<1.0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "if(uFaceCount<0){vec4 o=texture2D(sBlurTexture,buv());gl_FragColor=vec4(o.rgb*alpha,alpha);return;}" +
            "if(uFaceCount==0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "vec2 p=vec2(gl_FragCoord.x/uViewport.x,1.0-gl_FragCoord.y/uViewport.y);float m=0.0;" +
            "for(int i=0;i<" + MAX_FACES + ";++i){if(i>=uFaceCount)break;vec2 d=p-uFaceCenter[i];vec2 x=uFaceAxisX[i],y=uFaceAxisY[i];" +
             "float z=x.x*y.y-x.y*y.x;if(abs(z)<.000001)continue;vec2 l=vec2((d.x*y.y-d.y*y.x)/z,(-d.x*x.y+d.y*x.x)/z);" +
             "float q=sqrt(max(dot(l,l),0.0))/max(uMaskScale,.1);m=max(m,clamp(1.0-smoothstep(.84,1.04,q),0.0,1.0));}if(m<=0.0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "vec2 uv=buv();if(uMaskMode==1)uv=(floor(uv*32.0)+.5)/32.0;vec4 protectedColor=uMaskMode==2?vec4(.03,.03,.03,1.0):texture2D(sBlurTexture,uv);vec4 o=mix(src,protectedColor,m);gl_FragColor=vec4(o.rgb*alpha,alpha);}";

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
                + ", dropped=" + framesDropped + ", inferenceMs=" + average;
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
            float confidence = parseDetectionConfidence(confidenceValue);
            boolean useGpu = !"cpu".equals(processorValue);
            liteModel = !"precise".equals(modelValue);
            startFrameExecutor();
            Future<?> created = frameExecutor.submit(() -> {
                try {
                    if (liteModel) detector = createDetector(modelPath, confidence, useGpu);
                    else landmarker = createLandmarker(modelPath, confidence, useGpu);
                } catch (Exception error) {
                    throw new IllegalStateException("Selected model initialization failed", error);
                }
            });
            created.get(15, TimeUnit.SECONDS);
            hookRoundVideoResolution();
            hookCameraControls();
            hookSurfaceUpdates();
            hookCameraRenderer();
            hookEncoderRenderer();
            acceptingFrames = true;
            initialized = true;
            protectionState = "ACTIVE";
            emit("MediaPipe " + (liteModel ? modelValue + "-range Face Detector" : "Face Landmarker")
                    + " 0.10.29 armed with "
                    + (useGpu ? "GPU" : "CPU") + " + VIDEO on one worker");
        } catch (Throwable error) {
            protectionState = "FAILED";
            emit("Face Landmarker unavailable; host camera left untouched", error);
            onUnload();
            throw new IllegalStateException("Face Landmarker initialization failed", error);
        }
    }

    public static boolean reconfigure(String modelPath, String confidenceValue,
                                      String processorValue, String modelValue,
                                      String generationValue) {
        if (!initialized || frameExecutor == null) throw new IllegalStateException("Runtime is not active");
        final long requestGeneration;
        try { requestGeneration = Long.parseLong(generationValue); }
        catch (Throwable ignored) { throw new IllegalArgumentException("Invalid reconfigure generation"); }
        for (;;) {
            long previous = RECONFIGURE_GENERATION.get();
            if (requestGeneration < previous) return false;
            if (RECONFIGURE_GENERATION.compareAndSet(previous, requestGeneration)) break;
        }
        float confidence = parseDetectionConfidence(confidenceValue);
        boolean useGpu = !"cpu".equals(processorValue);
        boolean nextLite = !"precise".equals(modelValue);
        ExecutorService executor = frameExecutor;
        Future<EngineCandidate> created = executor.submit(() -> {
            if (requestGeneration != RECONFIGURE_GENERATION.get()) return null;
            return createEngineCandidate(modelPath, confidence, useGpu, nextLite);
        });
        EngineCandidate candidate = null;
        try {
            candidate = created.get(15, TimeUnit.SECONDS);
            if (candidate == null) return false;
            synchronized (Main.class) {
                if (!initialized || requestGeneration != RECONFIGURE_GENERATION.get()) {
                    candidate.close();
                    return false;
                }
                acceptingFrames = false;
                CapturedFrame queued = LATEST_FRAME.getAndSet(null);
                if (queued != null) FRAME_POOL.offer(queued.rgba);
                FaceLandmarker oldLandmarker = landmarker;
                FaceDetector oldDetector = detector;
                landmarker = candidate.landmarker;
                detector = candidate.detector;
                candidate.committed = true;
                liteModel = nextLite;
                SOURCE_TRACKS.clear();
                acceptingFrames = true;
                executor.execute(() -> closeEngines(oldLandmarker, oldDetector));
            }
            emit("Active model switched to " + (nextLite ? modelValue + "-range Face Detector" : "Face Landmarker"));
            return true;
        } catch (Throwable error) {
            if (candidate != null) candidate.close();
            if (initialized) acceptingFrames = true;
            emit("Model switch failed", error);
            throw new IllegalStateException("Model switch failed", error);
        }
    }

    public static boolean switchModel(String modelPath, String confidenceValue,
                                      String processorValue, String modelValue) {
        return reconfigure(modelPath, confidenceValue, processorValue, modelValue,
                Long.toString(RECONFIGURE_GENERATION.incrementAndGet()));
    }

    private static EngineCandidate createEngineCandidate(String modelPath, float confidence,
                                                          boolean useGpu, boolean lite) throws Exception {
        return lite ? new EngineCandidate(null, createDetector(modelPath, confidence, useGpu))
                : new EngineCandidate(createLandmarker(modelPath, confidence, useGpu), null);
    }

    private static void closeEngines(FaceLandmarker oldLandmarker, FaceDetector oldDetector) {
        try { if (oldLandmarker != null) oldLandmarker.close(); }
        catch (Throwable error) { emit("Old FaceLandmarker close failed", error); }
        try { if (oldDetector != null) oldDetector.close(); }
        catch (Throwable error) { emit("Old FaceDetector close failed", error); }
    }

    private static float parseDetectionConfidence(String value) {
        int parsed;
        try { parsed = Integer.parseInt(value); }
        catch (Throwable ignored) { parsed = 60; }
        if (parsed != 40 && parsed != 50 && parsed != 60) parsed = 60;
        return parsed / 100.0f;
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

    private static FaceLandmarker createLandmarker(String modelPath, float confidence,
                                                    boolean useGpu) throws Exception {
        try (FileInputStream stream = new FileInputStream(modelPath);
             FileChannel channel = stream.getChannel()) {
            ByteBuffer model = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            BaseOptions base = BaseOptions.builder().setModelAssetBuffer(model)
                    .setDelegate(useGpu ? Delegate.GPU : Delegate.CPU).build();
            FaceLandmarker.FaceLandmarkerOptions options = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(base)
                     .setRunningMode(RunningMode.VIDEO)
                    .setNumFaces(MAX_FACES)
                     .setMinFaceDetectionConfidence(confidence)
                     .setMinFacePresenceConfidence(confidence)
                     .setMinTrackingConfidence(0.50f)
                    .build();
            // The selected delegate is explicit; never switch modes silently.
            return FaceLandmarker.createFromOptions(ApplicationLoader.applicationContext, options);
        }
    }

    private static FaceDetector createDetector(String modelPath, float confidence,
                                                boolean useGpu) throws Exception {
        try (FileInputStream stream = new FileInputStream(modelPath);
             FileChannel channel = stream.getChannel()) {
            ByteBuffer model = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            BaseOptions base = BaseOptions.builder().setModelAssetBuffer(model)
                    .setDelegate(useGpu ? Delegate.GPU : Delegate.CPU).build();
            FaceDetector.FaceDetectorOptions options = FaceDetector.FaceDetectorOptions.builder()
                    .setBaseOptions(base).setRunningMode(RunningMode.VIDEO)
                    .setMinDetectionConfidence(confidence).setMinSuppressionThreshold(.30f).build();
            return FaceDetector.createFromOptions(ApplicationLoader.applicationContext, options);
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
            Theme.ResourcesProvider resourcesProvider =
                    (Theme.ResourcesProvider) field(cameraView.getClass(), "resourcesProvider").get(cameraView);
            BlurControl control = new BlurControl(
                    (FrameLayout) cameraView, zoomSlider, resourcesProvider);
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
        protectionState = enabled ? (initialized ? "ACTIVE" : "STARTING") : "DISABLED";
        CapturedFrame queued = LATEST_FRAME.getAndSet(null);
        if (queued != null) FRAME_POOL.offer(queued.rgba);
        SOURCE_TRACKS.clear();
        SOURCE_ACTIVE_SINCE.clear();
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
            state.faceCount = geometry == null ? (hasFreshResult(source, now) ? 0 : -1) : geometry.count;
            if (geometry != null) System.arraycopy(geometry.faces, 0, state.faces, 0, geometry.count * FACE_STRIDE);
            uploadGeometry(state.program, state.faceCount, state.faces, state.center, state.axisX,
                    state.axisY, state.viewport, state.blurSampler, state.geometryScratch);
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
            restorePreviewFields(thread, state);
            emit("Preview draw left untouched", error);
        }
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
        if (!state.swapActive) { restorePreviewBlur(state); return; }
        try {
            Class<?> type = thread.getClass();
            field(type, "drawProgram").setInt(thread, state.savedProgram);
            field(type, "positionHandle").setInt(thread, state.savedPosition);
            field(type, "textureHandle").setInt(thread, state.savedTexture);
            field(type, "vertexMatrixHandle").setInt(thread, state.savedMvp);
            field(type, "textureMatrixHandle").setInt(thread, state.savedSt);
        } catch (Throwable error) { emit("Preview host field restore failed", error); }
        finally { state.swapActive = false; restorePreviewBlur(state); }
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
            float[] st = new float[16]; surface.getTransformMatrix(st);
            float[] tex = new float[8]; hostTexture.position(0); hostTexture.get(tex);
            if (!DRAIN_SCHEDULED.get() && now - state.lastCaptureNanos >= state.captureIntervalNanos)
                frameBuffer = FRAME_POOL.poll();
            if (frameBuffer != null && state.readPixels == null) state.readPixels = ByteBuffer.allocateDirect(
                    CleanFrameTap.SIZE * CleanFrameTap.SIZE * 4).order(ByteOrder.nativeOrder());
            state.blurTexture = state.tap.renderBlur(textures[slot], mvp, st, tex,
                    frameBuffer == null ? null : state.readPixels);
            if (state.blurTexture == 0)
                throw new IllegalStateException("multi-pass preview blur failed: " + state.tap.lastError());
            if (!state.pipelineLogged) {
                state.pipelineLogged = true;
                emit("Preview multi-pass Gaussian ready size=" + CleanFrameTap.SIZE + "x" + CleanFrameTap.SIZE);
            }
            bindPreviewBlur(state);
            String source = sourceKey(thread, slot, surface);
            activateSource(state, source, now);
            SOURCE_BY_SLOT.put(slot, source);
            SOURCE_BY_TEXTURE.put(textures[slot], source);
            if (frameBuffer != null) {
                framesCaptured++;
                frameBuffer.clear();
                copyVerticallyCorrected(state.readPixels, frameBuffer, CleanFrameTap.SIZE, CleanFrameTap.SIZE);
                frameBuffer.position(0);
                CapturedFrame next = new CapturedFrame(frameBuffer, now, nextTimestampMs(now), source);
                frameBuffer = null;
                CapturedFrame old = LATEST_FRAME.getAndSet(next);
                if (old != null) { framesDropped++; FRAME_POOL.offer(old.rgba); }
                state.lastCaptureNanos = now;
                state.captureIntervalNanos = Math.max(CAPTURE_INTERVAL_NS,
                        Math.min(100_000_000L, state.captureIntervalNanos - 2_000_000L));
                scheduleDrain();
            }
        } catch (Throwable error) {
            if (frameBuffer != null) FRAME_POOL.offer(frameBuffer);
            state.captureIntervalNanos = Math.min(100_000_000L,
                    Math.max(CAPTURE_INTERVAL_NS, state.captureIntervalNanos + 10_000_000L));
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
            Thread thread = new Thread(r, "BlurFacesMediaPipeInput"); thread.setDaemon(true); return thread;
        });
    }

    private static void scheduleDrain() {
        ExecutorService executor = frameExecutor;
        if (executor == null || executor.isShutdown() || !DRAIN_SCHEDULED.compareAndSet(false, true)) return;
        try { executor.execute(Main::submitLatestFrame); }
        catch (Throwable ignored) { DRAIN_SCHEDULED.set(false); }
    }

    private static void submitLatestFrame() {
        CapturedFrame frame = LATEST_FRAME.getAndSet(null);
        if (frame == null) { DRAIN_SCHEDULED.set(false); return; }
        MPImage image = null;
        try {
            long inferenceStart = System.nanoTime();
            frame.rgba.position(0);
            image = new ByteBufferImageBuilder(frame.rgba, CleanFrameTap.SIZE, CleanFrameTap.SIZE,
                    MPImage.IMAGE_FORMAT_RGBA).build();
            FaceGeometry detections;
            if (liteModel) {
                FaceDetectorResult result = detector.detectForVideo(image, frame.timestampMs);
                detections = boxesToGeometry(result.detections(), frame.captureNanos, frame.sourceKey);
            } else {
                FaceLandmarkerResult result = landmarker.detectForVideo(image, frame.timestampMs);
                detections = meshToGeometry(result.faceLandmarks(), frame.captureNanos, frame.sourceKey);
            }
            if (isSourceActive(frame.sourceKey)) {
                updateTracks(frame.sourceKey, detections);
                framesProcessed++;
                inferenceNanos += System.nanoTime() - inferenceStart;
                maybeLogMetrics();
            } else emit("Dropped late MediaPipe result from retired source=" + frame.sourceKey);
        } catch (Throwable error) {
            emit("MediaPipe VIDEO frame failed", error);
        } finally {
            if (image != null) image.close();
            FRAME_POOL.offer(frame.rgba);
            DRAIN_SCHEDULED.set(false);
            if (acceptingFrames && LATEST_FRAME.get() != null) scheduleDrain();
        }
    }

    private static FaceGeometry boxesToGeometry(List<Detection> faces, long captureNanos,
                                                 String source) {
        float[] output = new float[MAX_FACES * FACE_STRIDE];
        int count = 0;
        for (Detection face : faces) {
            if (count >= MAX_FACES) break;
            android.graphics.RectF box = face.boundingBox();
            float cx = box.centerX() / CleanFrameTap.SIZE;
            float cy = box.centerY() / CleanFrameTap.SIZE;
            // BlazeFace boxes are intentionally expanded: unlike the landmark
            // path they have no dense contour, and slower devices need movement
            // margin between detector results.
            float halfWidth = box.width() * 1.50f / (2f * CleanFrameTap.SIZE);
            float halfHeight = box.height() * 1.65f / (2f * CleanFrameTap.SIZE);
            float ux = 1f, uy = 0f;
            if (face.keypoints().isPresent() && face.keypoints().get().size() >= 2) {
                NormalizedKeypoint left = face.keypoints().get().get(0);
                NormalizedKeypoint right = face.keypoints().get().get(1);
                float dx = right.x() - left.x(), dy = right.y() - left.y();
                float length = (float) Math.hypot(dx, dy);
                if (length > .001f) { ux = dx / length; uy = dy / length; }
            }
            int offset = count * FACE_STRIDE;
            output[offset] = cx; output[offset + 1] = cy;
            output[offset + 2] = ux * halfWidth; output[offset + 3] = uy * halfWidth;
            output[offset + 4] = -uy * halfHeight; output[offset + 5] = ux * halfHeight;
            count++;
        }
        return new FaceGeometry(output, count, captureNanos, source);
    }

    private static FaceGeometry meshToGeometry(List<List<NormalizedLandmark>> faces,
                                                long captureNanos, String source) {
        float[] output = new float[MAX_FACES * FACE_STRIDE];
        int count = 0;
        for (List<NormalizedLandmark> mesh : faces) {
            if (count == MAX_FACES || mesh.size() <= 454) break;
            if (ovalPcaAffine(mesh, output, count * FACE_STRIDE)) count++;
        }
        return new FaceGeometry(output, count, captureNanos, source);
    }

    private static boolean ovalPcaAffine(List<NormalizedLandmark> mesh, float[] out, int offset) {
        float cx = 0f, cy = 0f;
        for (int index : OVAL) { NormalizedLandmark p = mesh.get(index); cx += p.x(); cy += p.y(); }
        cx /= OVAL.length; cy /= OVAL.length;
        float xx = 0f, xy = 0f, yy = 0f;
        for (int index : OVAL) {
            NormalizedLandmark p = mesh.get(index); float x = p.x() - cx, y = p.y() - cy;
            if (!Float.isFinite(x) || !Float.isFinite(y)) return false;
            xx += x * x; xy += x * y; yy += y * y;
        }
        float angle = 0.5f * (float) Math.atan2(2f * xy, xx - yy);
        float ux = (float) Math.cos(angle), uy = (float) Math.sin(angle);
        float vx = -uy, vy = ux, minU = Float.MAX_VALUE, maxU = -Float.MAX_VALUE;
        float minV = Float.MAX_VALUE, maxV = -Float.MAX_VALUE;
        for (int index : OVAL) {
            NormalizedLandmark p = mesh.get(index); float dx = p.x() - cx, dy = p.y() - cy;
            float u = dx * ux + dy * uy, v = dx * vx + dy * vy;
            minU = Math.min(minU, u); maxU = Math.max(maxU, u);
            minV = Math.min(minV, v); maxV = Math.max(maxV, v);
        }
        float rawU = (maxU - minU) * .5f, rawV = (maxV - minV) * .5f;
        if (!Float.isFinite(cx) || !Float.isFinite(cy) || rawU < .008f || rawV < .008f) return false;
        // Expand beyond the mesh oval and add a small edge allowance. If the
        // mesh is tiny, scale both axes together so its aspect ratio is retained.
        float radiusU = (maxU - minU) * .78f + .004f;
        float radiusV = (maxV - minV) * .78f + .004f;
        float safeScale = Math.max(1f, .022f / Math.min(radiusU, radiusV));
        radiusU *= safeScale; radiusV *= safeScale;
        cx += ux * (maxU + minU) * .5f + vx * (maxV + minV) * .5f;
        cy += uy * (maxU + minU) * .5f + vy * (maxV + minV) * .5f;
        out[offset] = cx; out[offset + 1] = cy;
        out[offset + 2] = ux * radiusU; out[offset + 3] = uy * radiusU;
        out[offset + 4] = vx * radiusV; out[offset + 5] = vy * radiusV;
        return true;
    }

    private static FaceGeometry geometryFor(String source, long now) {
        if (source == null || source.isEmpty()) return null;
        Long activeSince = SOURCE_ACTIVE_SINCE.get(source);
        if (activeSince == null) return null;
        SourceTracks tracks = SOURCE_TRACKS.get(source);
        if (tracks == null || tracks.lastResultNanos < activeSince) return null;
        if (!tracks.hasFreshPublication(now)) protectionState = "DEGRADED";
        else if (initialized && blurEnabled) protectionState = "ACTIVE";
        return tracks.geometryAt(now, source);
    }

    private static boolean hasFreshResult(String source, long now) {
        if (source == null || source.isEmpty()) return false;
        Long activeSince = SOURCE_ACTIVE_SINCE.get(source);
        SourceTracks tracks = SOURCE_TRACKS.get(source);
        return activeSince != null && tracks != null && tracks.lastResultNanos >= activeSince
                && tracks.hasFreshPublication(now);
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
            // preview/resolution are part of the host interface but this shader
            // does not consume them, so GLES may legally optimize them to -1.
            // Host glUniform calls with -1 are defined no-ops.
            state.ready = state.position >= 0 && state.texture >= 0 && state.mvp >= 0 && state.st >= 0
                    && state.alpha >= 0
                    && state.center >= 0 && state.axisX >= 0 && state.axisY >= 0 && state.count >= 0
                    && state.viewport >= 0 && state.blurSampler >= 0;
            if (!state.ready) throw new IllegalStateException("encoder shader interface incomplete");
            emit("Encoder blur shader ready size=" + width + "x" + height);
        } catch (Throwable error) { state.ready = false; emit("Encoder shader setup failed", error); }
    }

    private static void beforeEncoderDraw(Object renderer, Object snapshot) {
        EncoderState state = encoderState(renderer);
        if (!acceptingFrames || !blurEnabled || !state.ready || snapshot == null) return;
        try {
            int slot = field(snapshot.getClass(), "surfaceIndex").getInt(snapshot);
            int textureId = field(snapshot.getClass(), "textureId").getInt(snapshot);
            float[] mvp = (float[]) field(snapshot.getClass(), "mvpMatrix").get(snapshot);
            float[] st = (float[]) field(snapshot.getClass(), "stMatrix").get(snapshot);
            float[] tex = (float[]) field(snapshot.getClass(), "textureCoords").get(snapshot);
            state.blurTexture = state.tap.renderBlur(textureId, mvp, st, tex, null);
            if (state.blurTexture == 0)
                throw new IllegalStateException("multi-pass encoder blur failed: " + state.tap.lastError());
            bindEncoderBlur(state);
            String source = SOURCE_BY_TEXTURE.get(textureId);
            if (source == null) source = SOURCE_BY_SLOT.get(slot);
            long now = System.nanoTime();
            FaceGeometry geometry = geometryFor(source, now);
            state.faceCount = geometry == null ? (hasFreshResult(source, now) ? 0 : -1) : geometry.count;
            if (encoderTransitionActive(renderer)) state.faceCount = -1;
            if (geometry != null) System.arraycopy(geometry.faces, 0, state.faces, 0, geometry.count * FACE_STRIDE);
            uploadGeometry(state.program, state.faceCount, state.faces, state.center, state.axisX,
                    state.axisY, state.viewport, state.blurSampler, state.geometryScratch);
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
            restoreEncoderFields(renderer, state);
            emit("Encoder draw left untouched", error);
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

    private static boolean encoderTransitionActive(Object renderer) {
        try {
            Object outer = field(renderer.getClass(), "this$0").get(renderer);
            int[] oldTextures = (int[]) field(outer.getClass(), "oldCameraTexture").get(outer);
            return oldTextures != null && oldTextures.length > 0 && oldTextures[0] != 0;
        } catch (Throwable ignored) { return false; }
    }

    private static void restoreEncoderFields(Object renderer, EncoderState state) {
        if (!state.swapActive) { restoreEncoderBlur(state); return; }
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
        finally { state.swapActive = false; restoreEncoderBlur(state); }
    }

    private static void uploadGeometry(int program, int count, float[] faces,
                                        int center, int axisX, int axisY, int viewportLocation,
                                        int blurSamplerLocation, float[] scratch) {
        int[] prior = new int[1], viewport = new int[4];
        GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, prior, 0);
        GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
        try {
            GLES20.glUseProgram(program); GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uFaceCount"), count);
            GLES20.glUniform2f(viewportLocation, viewport[2], viewport[3]);
            GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "uMaskScale"), faceMaskScale);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uMaskMode"), maskMode);
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

    private static String sourceKey(Object thread, int slot, SurfaceTexture surface) {
        try {
            int[] generations = (int[]) field(thread.getClass(), "surfaceGeneration").get(thread);
            return slot + ":" + generations[slot] + ":" + System.identityHashCode(surface);
        } catch (Throwable ignored) { return ""; }
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

    private static void activateSource(CameraState state, String source, long now) {
        if (source == null || source.isEmpty() || source.equals(state.activeSource)) return;
        state.activeSource = source;
        SOURCE_ACTIVE_SINCE.put(source, now);
        emit("Camera source changed; full-frame privacy blur active until fresh detection source=" + source);
    }

    private static void updateTracks(String source, FaceGeometry detections) {
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
            restorePreviewFields(owner, state); state.tap.release();
            if (state.program != 0) GLES20.glDeleteProgram(state.program);
        }
    }

    private static void releaseEncoderState(Object owner) {
        EncoderState state;
        synchronized (ENCODER_STATES) { state = ENCODER_STATES.remove(owner); }
        if (state != null) {
            restoreEncoderFields(owner, state);
            state.tap.release();
            if (state.program != 0) GLES20.glDeleteProgram(state.program);
        }
    }

    public static synchronized boolean onUnload() {
        boolean clean = true;
        acceptingFrames = false;
        protectionState = "STOPPING";
        for (XC_MethodHook.Unhook hook : HOOKS) try { hook.unhook(); } catch (Throwable ignored) { }
        HOOKS.clear();
        CapturedFrame queued = LATEST_FRAME.getAndSet(null);
        if (queued != null) FRAME_POOL.offer(queued.rgba);
        ExecutorService executor = frameExecutor; frameExecutor = null;
        FaceLandmarker active = landmarker;
        FaceDetector activeDetector = detector;
        if (active != null && executor != null && !executor.isShutdown()) {
            try { executor.submit(active::close).get(3, TimeUnit.SECONDS); }
            catch (Throwable error) { clean = false; emit("FaceLandmarker worker close failed", error); }
        }
        if (activeDetector != null && executor != null && !executor.isShutdown()) {
            try { executor.submit(activeDetector::close).get(3, TimeUnit.SECONDS); }
            catch (Throwable error) { clean = false; emit("FaceDetector worker close failed", error); }
        }
        clean &= stopExecutor(executor, "MediaPipe input");
        landmarker = null;
        detector = null;
        DRAIN_SCHEDULED.set(false); SOURCE_TRACKS.clear(); SOURCE_ACTIVE_SINCE.clear();
        SOURCE_BY_SLOT.clear(); SOURCE_BY_TEXTURE.clear();
        CAMERA_STATES.clear(); ENCODER_STATES.clear(); FRAME_POOL.clear(); initialized = false;
        AndroidUtilities.runOnUIThread(() -> {
            synchronized (BLUR_CONTROLS) {
                for (BlurControl control : BLUR_CONTROLS.values()) control.detach();
                BLUR_CONTROLS.clear();
            }
        });
        blurEnabled = true;
        logger = null;
        FIELD_CACHE.clear();
        protectionState = "DISABLED";
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
        int program, position, texture, mvp, st, center, axisX, axisY, count, viewport, blurSampler, faceCount;
        int savedProgram, savedPosition, savedTexture, savedMvp, savedSt, blurTexture, savedBlurBinding;
        boolean swapActive, blurBindingActive, pipelineLogged; long lastCaptureNanos;
        long captureIntervalNanos = CAPTURE_INTERVAL_NS; ByteBuffer readPixels; String activeSource;
    }

    private static final class BlurControl {
        final FrameLayout host;
        final View zoomSlider;
        final FrameLayout pill;
        final View selector;
        final Theme.ResourcesProvider resourcesProvider;
        final ImageView blurButton;
        final ImageView clearButton;
        final ViewTreeObserver.OnPreDrawListener positionListener = () -> {
            updatePosition();
            return true;
        };

        BlurControl(FrameLayout host, View zoomSlider, Theme.ResourcesProvider resourcesProvider) {
            this.host = host;
            this.zoomSlider = zoomSlider;
            this.resourcesProvider = resourcesProvider;
            pill = new FrameLayout(host.getContext());
            selector = new View(host.getContext());
            blurButton = button(R.drawable.msg_photo_blur, "Blur faces");
            clearButton = button(R.drawable.msg_blur_off, "Do not blur faces");
            blurButton.setOnClickListener(view -> choose(true, view));
            clearButton.setOnClickListener(view -> choose(false, view));
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
            // The zoom view reserves 16 dp below its visible 48 dp pill.
            float top = zoomSlider.getY() + zoomSlider.getHeight() - AndroidUtilities.dp(8);
            float maximum = Math.max(0, host.getHeight() - pill.getHeight() - AndroidUtilities.dp(8));
            pill.setTranslationY(Math.min(top, maximum));
            pill.setAlpha(zoomSlider.getAlpha());
            pill.setVisibility(host.getVisibility() == View.VISIBLE ? View.VISIBLE : View.INVISIBLE);
        }

        void choose(boolean enabled, View source) {
            source.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP);
            setBlurEnabled(enabled);
        }

        void update() {
            applyTheme();
            float target = blurEnabled ? 0f : AndroidUtilities.dp(48);
            selector.animate().translationX(target).setDuration(180L).start();
            style(blurButton, blurEnabled);
            style(clearButton, !blurEnabled);
        }

        private void applyTheme() {
            int panel = Theme.getColor(Theme.key_chat_messagePanelBackground, resourcesProvider);
            int selected = Theme.getColor(Theme.key_featuredStickers_addButton, resourcesProvider);
            pill.setBackground(Theme.createRoundRectDrawable(AndroidUtilities.dp(24), panel));
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
        int center, axisX, axisY, count, viewport, blurSampler, faceCount, blurTexture, savedBlurBinding;
        int savedProgram, savedPosition, savedTexture, savedPreview, savedResolution, savedAlpha;
        int savedMvp, savedSt, savedTexel;
        boolean ready, swapActive, activeLogged, blurBindingActive, pipelineLogged;
    }

    private static final class CapturedFrame {
        final ByteBuffer rgba; final long captureNanos, timestampMs; final String sourceKey;
        CapturedFrame(ByteBuffer rgba, long captureNanos, long timestampMs, String sourceKey) {
            this.rgba = rgba; this.captureNanos = captureNanos; this.timestampMs = timestampMs; this.sourceKey = sourceKey;
        }
    }

    private static final class EngineCandidate {
        final FaceLandmarker landmarker;
        final FaceDetector detector;
        boolean committed;
        EngineCandidate(FaceLandmarker landmarker, FaceDetector detector) {
            this.landmarker = landmarker;
            this.detector = detector;
        }
        void close() { if (!committed) closeEngines(landmarker, detector); }
    }

    private static final class FaceGeometry {
        final float[] faces; final int count; final long captureNanos; final String sourceKey;
        FaceGeometry(float[] faces, int count, long captureNanos, String sourceKey) {
            this.faces = faces; this.count = count; this.captureNanos = captureNanos; this.sourceKey = sourceKey;
        }
    }

    private static final class SourceTracks {
        final FaceTrack[] tracks = new FaceTrack[MAX_FACES];
        final boolean[] trackMatched = new boolean[MAX_FACES];
        final boolean[] detectionMatched = new boolean[MAX_FACES];
        long lastResultNanos, lastPublishedNanos, adaptiveHoldNanos = TRACK_HOLD_NS;

        synchronized void update(FaceGeometry detections, long publishedNanos) {
            if (lastPublishedNanos != 0L) {
                long interval = publishedNanos - lastPublishedNanos;
                adaptiveHoldNanos = Math.max(TRACK_HOLD_NS,
                        Math.min(MAX_TRACK_HOLD_NS, interval * 3L + 100_000_000L));
            }
            lastPublishedNanos = publishedNanos;
            lastResultNanos = detections.captureNanos;
            java.util.Arrays.fill(trackMatched, false);
            java.util.Arrays.fill(detectionMatched, false);
            // Repeated global-nearest pairing is independent of detector list order
            // and avoids identity swaps caused by matching face i to track i.
            while (true) {
                int bestTrack = -1, bestDetection = -1;
                float bestDistance = Float.MAX_VALUE;
                for (int t = 0; t < MAX_FACES; t++) {
                    FaceTrack track = tracks[t];
                    if (track == null || trackMatched[t]
                            || publishedNanos - track.lastSeenPublishedNanos > adaptiveHoldNanos) continue;
                    for (int d = 0; d < detections.count; d++) {
                        if (detectionMatched[d]) continue;
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
                tracks[bestTrack].observe(detections.faces, bestDetection * FACE_STRIDE,
                        detections.captureNanos, publishedNanos);
                trackMatched[bestTrack] = true;
                detectionMatched[bestDetection] = true;
            }
            for (int d = 0; d < detections.count; d++) if (!detectionMatched[d]) {
                int slot = replacementSlot(publishedNanos);
                if (slot >= 0) {
                    tracks[slot] = new FaceTrack(detections.faces, d * FACE_STRIDE,
                            detections.captureNanos, publishedNanos);
                    trackMatched[slot] = true;
                }
            }
            for (int t = 0; t < MAX_FACES; t++) if (tracks[t] != null
                    && publishedNanos - tracks[t].lastSeenPublishedNanos > adaptiveHoldNanos) tracks[t] = null;
        }

        private int replacementSlot(long now) {
            for (int i = 0; i < MAX_FACES; i++)
                if (tracks[i] == null || now - tracks[i].lastSeenPublishedNanos > adaptiveHoldNanos) return i;
            int oldest = -1;
            for (int i = 0; i < MAX_FACES; i++) if (!trackMatched[i]
                    && (oldest < 0 || tracks[i].lastSeenNanos < tracks[oldest].lastSeenNanos)) oldest = i;
            return oldest;
        }

        synchronized FaceGeometry geometryAt(long now, String source) {
            float[] output = new float[MAX_FACES * FACE_STRIDE];
            int count = 0;
            for (int i = 0; i < MAX_FACES; i++) {
                FaceTrack track = tracks[i];
                if (track == null || now - track.lastSeenPublishedNanos > adaptiveHoldNanos) continue;
                track.predict(output, count * FACE_STRIDE, now);
                count++;
            }
            return count == 0 ? null : new FaceGeometry(output, count, lastResultNanos, source);
        }

        synchronized boolean hasFreshPublication(long now) {
            return lastPublishedNanos != 0L && now - lastPublishedNanos <= adaptiveHoldNanos;
        }
    }

    private static final class FaceTrack {
        final float[] values = new float[FACE_STRIDE];
        final float[] velocity = new float[FACE_STRIDE];
        long lastSeenNanos, lastSeenPublishedNanos;

        FaceTrack(float[] detection, int offset, long now, long publishedNanos) {
            System.arraycopy(detection, offset, values, 0, FACE_STRIDE);
            lastSeenNanos = now; lastSeenPublishedNanos = publishedNanos;
        }

        void observe(float[] detection, int offset, long now, long publishedNanos) {
            float dt = Math.max(.001f, (now - lastSeenNanos) / 1_000_000_000f);
            float sign = detection[offset + 2] * values[2] + detection[offset + 3] * values[3] < 0f
                    ? -1f : 1f;
            for (int i = 0; i < FACE_STRIDE; i++) {
                float measured = detection[offset + i] * (i >= 2 ? sign : 1f);
                float previous = values[i];
                float filtered = previous + TRACK_SMOOTHING * (measured - previous);
                float measuredVelocity = (filtered - previous) / dt;
                velocity[i] = clamp(velocity[i] * .5f + measuredVelocity * .5f, -2f, 2f);
                values[i] = filtered;
            }
            lastSeenNanos = now; lastSeenPublishedNanos = publishedNanos;
        }

        void predict(float[] output, int offset, long now) {
            float dt = Math.max(0f, Math.min(MAX_PREDICTION_NS, now - lastSeenNanos)) / 1_000_000_000f;
            output[offset] = values[0] + velocity[0] * dt;
            output[offset + 1] = values[1] + velocity[1] * dt;
            System.arraycopy(values, 2, output, offset + 2, FACE_STRIDE - 2);
        }
    }

    private static float axisLength(float[] values, int offset) {
        return (float) Math.hypot(values[offset], values[offset + 1]);
    }

    private static float clamp(float value, float low, float high) {
        return Math.max(low, Math.min(high, value));
    }
}
