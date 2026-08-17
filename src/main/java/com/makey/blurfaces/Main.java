package com.makey.blurfaces;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.GLES20;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;

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
import java.util.function.Consumer;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedBridge;
import org.telegram.messenger.ApplicationLoader;

/** Round-video face blur whose sole geometry source is MediaPipe Face Landmarker. */
public final class Main {
    private static final String TAG = "BlurFaces";
    private static final String CAMERA_GL_THREAD =
            "org.telegram.ui.Components.InstantCameraView$CameraGLThread";
    private static final String ENCODER_RENDERER =
            "org.telegram.ui.Components.InstantCameraView$EncoderRenderer";
    private static final String FRAME_SNAPSHOT =
            "com.exteragram.messenger.camera.RoundVideoEncoder$FrameSnapshot";
    private static final int MAX_FACES = 4;
    private static final int FACE_STRIDE = 6;
    private static final long CAPTURE_INTERVAL_NS = 33_333_333L;
    private static final long STALE_HOLD_NS = 350_000_000L;
    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;
    // MediaPipe's canonical face oval. PCA makes roll stable while extrema size the mask.
    private static final int[] OVAL = {10,338,297,332,284,251,389,356,454,323,361,288,397,
            365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109};

    private static final List<XC_MethodHook.Unhook> HOOKS = new ArrayList<>();
    private static final Map<Object, CameraState> CAMERA_STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, CameraState>());
    private static final Map<Object, EncoderState> ENCODER_STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, EncoderState>());
    private static final Map<Integer, String> SOURCE_BY_SLOT = new ConcurrentHashMap<>();
    private static final Map<Integer, String> SOURCE_BY_TEXTURE = new ConcurrentHashMap<>();
    private static final Map<String, Long> SOURCE_ACTIVE_SINCE = new ConcurrentHashMap<>();
    private static final Map<String, FaceGeometry> RESULTS = new ConcurrentHashMap<>();
    private static final Map<String, FaceGeometry> HELD_RESULTS = new ConcurrentHashMap<>();
    private static final AtomicReference<CapturedFrame> LATEST_FRAME = new AtomicReference<>();
    private static final AtomicReference<SubmittedFrame> IN_FLIGHT = new AtomicReference<>();
    private static final AtomicBoolean DRAIN_SCHEDULED = new AtomicBoolean();
    private static final ArrayBlockingQueue<ByteBuffer> FRAME_POOL = new ArrayBlockingQueue<>(3);

    private static volatile boolean initialized;
    private static volatile boolean acceptingFrames;
    private static FaceLandmarker landmarker;
    private static ExecutorService frameExecutor;
    private static long lastTimestampMs;
    private static Consumer<String> logger;

    private static final String VS =
            "uniform mat4 uMVPMatrix;uniform mat4 uSTMatrix;attribute vec4 aPosition;" +
            "attribute vec4 aTextureCoord;varying vec2 vTextureCoord;void main(){" +
            "gl_Position=uMVPMatrix*aPosition;vTextureCoord=(uSTMatrix*aTextureCoord).xy;}";
    private static final String FS =
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float;" +
            "varying vec2 vTextureCoord;uniform samplerExternalOES sTexture;uniform sampler2D sBlurTexture;" +
            "uniform vec2 uFaceCenter[" + MAX_FACES + "];uniform vec2 uFaceAxisX[" + MAX_FACES + "];" +
            "uniform vec2 uFaceAxisY[" + MAX_FACES + "];uniform int uFaceCount;uniform vec2 uViewport;" +
            "vec2 buv(){return clamp(gl_FragCoord.xy/uViewport,0.0,1.0);}" +
            "void main(){vec4 src=texture2D(sTexture,vTextureCoord);if(uViewport.x<1.0){gl_FragColor=src;return;}" +
            "if(uFaceCount<0){gl_FragColor=texture2D(sBlurTexture,buv());return;}if(uFaceCount==0){gl_FragColor=src;return;}" +
            "vec2 p=vec2(gl_FragCoord.x/uViewport.x,1.0-gl_FragCoord.y/uViewport.y);float m=0.0;" +
            "for(int i=0;i<" + MAX_FACES + ";++i){if(i>=uFaceCount)break;vec2 d=p-uFaceCenter[i];vec2 x=uFaceAxisX[i],y=uFaceAxisY[i];" +
            "float z=x.x*y.y-x.y*y.x;if(abs(z)<.000001)continue;vec2 l=vec2((d.x*y.y-d.y*y.x)/z,(-d.x*x.y+d.y*x.x)/z);" +
            "m=max(m,1.0-smoothstep(.82,1.0,length(l)));}if(m<=0.0){gl_FragColor=src;return;}" +
            "gl_FragColor=mix(src,texture2D(sBlurTexture,buv()),m);}";
    private static final String ENCODER_FS =
            "#extension GL_OES_EGL_image_external : require\nprecision highp float;varying vec2 vTextureCoord;" +
            "uniform samplerExternalOES sTexture;uniform sampler2D sBlurTexture;uniform vec2 preview;uniform vec2 resolution;uniform float alpha;uniform vec2 texelSize;" +
            "uniform vec2 uFaceCenter[" + MAX_FACES + "];uniform vec2 uFaceAxisX[" + MAX_FACES + "];" +
            "uniform vec2 uFaceAxisY[" + MAX_FACES + "];uniform int uFaceCount;uniform vec2 uViewport;" +
            "vec2 buv(){return clamp(gl_FragCoord.xy/uViewport,0.0,1.0);}" +
            "void main(){vec4 src=texture2D(sTexture,vTextureCoord);if(uViewport.x<1.0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "if(uFaceCount<0){vec4 o=texture2D(sBlurTexture,buv());gl_FragColor=vec4(o.rgb*alpha,alpha);return;}" +
            "if(uFaceCount==0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "vec2 p=vec2(gl_FragCoord.x/uViewport.x,1.0-gl_FragCoord.y/uViewport.y);float m=0.0;" +
            "for(int i=0;i<" + MAX_FACES + ";++i){if(i>=uFaceCount)break;vec2 d=p-uFaceCenter[i];vec2 x=uFaceAxisX[i],y=uFaceAxisY[i];" +
            "float z=x.x*y.y-x.y*y.x;if(abs(z)<.000001)continue;vec2 l=vec2((d.x*y.y-d.y*y.x)/z,(-d.x*x.y+d.y*x.x)/z);" +
            "m=max(m,1.0-smoothstep(.82,1.0,length(l)));}if(m<=0.0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}" +
            "vec4 o=mix(src,texture2D(sBlurTexture,buv()),m);gl_FragColor=vec4(o.rgb*alpha,alpha);}";

    private Main() { }
    public static void setLogger(Consumer<String> value) { logger = value; }
    private static void emit(String message) {
        Log.i(TAG, message);
        if (logger != null) try { logger.accept("[BlurFaces] " + message); } catch (Throwable ignored) { }
    }
    private static void emit(String message, Throwable error) {
        Log.e(TAG, message, error);
        if (logger != null) try { logger.accept("[BlurFaces] " + message + ": " + error); } catch (Throwable ignored) { }
    }

    public static synchronized void initAndStart(String modelPath) {
        if (initialized) return;
        try {
            landmarker = createGpuLandmarker(modelPath);
            startFrameExecutor();
            hookSurfaceUpdates();
            hookCameraRenderer();
            hookEncoderRenderer();
            acceptingFrames = true;
            initialized = true;
            emit("MediaPipe Face Landmarker 0.10.29 armed with GPU + LIVE_STREAM");
        } catch (Throwable error) {
            emit("GPU Face Landmarker unavailable; host camera left untouched", error);
            onUnload();
            throw new IllegalStateException("GPU Face Landmarker initialization failed", error);
        }
    }

    private static FaceLandmarker createGpuLandmarker(String modelPath) throws Exception {
        try (FileInputStream stream = new FileInputStream(modelPath);
             FileChannel channel = stream.getChannel()) {
            ByteBuffer model = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            BaseOptions base = BaseOptions.builder().setModelAssetBuffer(model)
                    .setDelegate(Delegate.GPU).build();
            FaceLandmarker.FaceLandmarkerOptions options = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(base)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumFaces(MAX_FACES)
                    .setMinFaceDetectionConfidence(0.60f)
                    .setMinFacePresenceConfidence(0.60f)
                    .setMinTrackingConfidence(0.60f)
                    .setResultListener(Main::onLandmarkerResult)
                    .setErrorListener(Main::onLandmarkerError)
                    .build();
            // No CPU retry is permitted: failure here occurs before any host hook.
            return FaceLandmarker.createFromOptions(ApplicationLoader.applicationContext, options);
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
        if (!acceptingFrames) return;
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
                    state.axisY, state.viewport, state.blurSampler);
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
        int[] active = new int[1], binding = new int[1];
        GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, active, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        if (!state.blurBindingActive) {
            GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, binding, 0);
            state.savedBlurBinding = binding[0];
            state.blurBindingActive = true;
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.blurTexture);
        GLES20.glActiveTexture(active[0]);
    }

    private static void restorePreviewBlur(CameraState state) {
        if (!state.blurBindingActive) return;
        int[] active = new int[1]; GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, active, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.savedBlurBinding);
        GLES20.glActiveTexture(active[0]);
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
        if (!acceptingFrames || EGL14.eglGetCurrentContext() == EGL14.EGL_NO_CONTEXT) return;
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
            if (now - state.lastCaptureNanos >= CAPTURE_INTERVAL_NS) frameBuffer = FRAME_POOL.poll();
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
                frameBuffer.clear();
                copyVerticallyCorrected(state.readPixels, frameBuffer, CleanFrameTap.SIZE, CleanFrameTap.SIZE);
                frameBuffer.position(0);
                CapturedFrame next = new CapturedFrame(frameBuffer, now, nextTimestampMs(now), source);
                frameBuffer = null;
                CapturedFrame old = LATEST_FRAME.getAndSet(next);
                if (old != null) FRAME_POOL.offer(old.rgba);
                state.lastCaptureNanos = now;
                scheduleDrain();
            }
        } catch (Throwable error) {
            if (frameBuffer != null) FRAME_POOL.offer(frameBuffer);
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
        if (executor == null || executor.isShutdown() || IN_FLIGHT.get() != null
                || !DRAIN_SCHEDULED.compareAndSet(false, true)) return;
        try { executor.execute(Main::submitLatestFrame); }
        catch (Throwable ignored) { DRAIN_SCHEDULED.set(false); }
    }

    private static void submitLatestFrame() {
        DRAIN_SCHEDULED.set(false);
        if (!acceptingFrames || IN_FLIGHT.get() != null) return;
        CapturedFrame frame = LATEST_FRAME.getAndSet(null);
        if (frame == null) return;
        Bitmap bitmap = null;
        MPImage image = null;
        boolean bufferReturned = false;
        try {
            bitmap = Bitmap.createBitmap(CleanFrameTap.SIZE, CleanFrameTap.SIZE, Bitmap.Config.ARGB_8888);
            frame.rgba.position(0); bitmap.copyPixelsFromBuffer(frame.rgba);
            image = new BitmapImageBuilder(bitmap).build();
            FRAME_POOL.offer(frame.rgba);
            bufferReturned = true;
            SubmittedFrame submitted = new SubmittedFrame(frame.captureNanos, frame.timestampMs,
                    frame.sourceKey, bitmap, image);
            if (!IN_FLIGHT.compareAndSet(null, submitted)) throw new IllegalStateException("in-flight race");
            landmarker.detectAsync(image, frame.timestampMs);
        } catch (Throwable error) {
            if (!bufferReturned) FRAME_POOL.offer(frame.rgba);
            if (image != null) image.close();
            if (bitmap != null) bitmap.recycle();
            IN_FLIGHT.set(null);
            emit("MediaPipe submission failed", error);
            scheduleDrain();
        }
    }

    private static void onLandmarkerResult(FaceLandmarkerResult result, MPImage input) {
        SubmittedFrame submitted = IN_FLIGHT.getAndSet(null);
        try {
            if (submitted == null || result.timestampMs() != submitted.timestampMs) return;
            if (!isSourceActive(submitted.sourceKey)) {
                emit("Dropped late MediaPipe result from retired source=" + submitted.sourceKey);
                return;
            }
            FaceGeometry geometry = meshToGeometry(result.faceLandmarks(), submitted.captureNanos, submitted.sourceKey);
            RESULTS.put(submitted.sourceKey, geometry);
            if (geometry.count > 0) HELD_RESULTS.put(submitted.sourceKey, geometry);
        } catch (Throwable error) { emit("MediaPipe result rejected", error); }
        finally {
            if (input != null) input.close();
            if (submitted != null) {
                if (submitted.image != input) submitted.image.close();
                submitted.bitmap.recycle();
            }
            scheduleDrain();
        }
    }

    private static void onLandmarkerError(RuntimeException error) {
        SubmittedFrame submitted = IN_FLIGHT.getAndSet(null);
        if (submitted != null) {
            submitted.image.close();
            submitted.bitmap.recycle();
        }
        emit("MediaPipe LIVE_STREAM error", error);
        scheduleDrain();
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
        // Keep the detected oval inside the fully blurred core; feather only
        // outside the face so contour pixels never blend back to the original.
        float radiusU = (maxU - minU) * 0.68f, radiusV = (maxV - minV) * 0.68f;
        if (!Float.isFinite(cx) || !Float.isFinite(cy) || radiusU < .025f || radiusV < .025f) return false;
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
        FaceGeometry current = RESULTS.get(source);
        if (current != null && current.captureNanos >= activeSince && current.count > 0
                && now - current.captureNanos <= STALE_HOLD_NS) return current;
        FaceGeometry held = HELD_RESULTS.get(source);
        return held != null && held.captureNanos >= activeSince && now - held.captureNanos <= STALE_HOLD_NS
                ? held : null;
    }

    private static boolean hasFreshResult(String source, long now) {
        if (source == null || source.isEmpty()) return false;
        Long activeSince = SOURCE_ACTIVE_SINCE.get(source);
        FaceGeometry current = RESULTS.get(source);
        return activeSince != null && current != null && current.captureNanos >= activeSince
                && now - current.captureNanos <= STALE_HOLD_NS;
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
        if (!acceptingFrames || !state.ready || snapshot == null) return;
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
                    state.axisY, state.viewport, state.blurSampler);
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
        int[] active = new int[1], binding = new int[1];
        GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, active, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        if (!state.blurBindingActive) {
            GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, binding, 0);
            state.savedBlurBinding = binding[0];
            state.blurBindingActive = true;
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.blurTexture);
        GLES20.glActiveTexture(active[0]);
    }

    private static void restoreEncoderBlur(EncoderState state) {
        if (!state.blurBindingActive) return;
        int[] active = new int[1]; GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, active, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, state.savedBlurBinding);
        GLES20.glActiveTexture(active[0]);
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
                                        int blurSamplerLocation) {
        int[] prior = new int[1], viewport = new int[4];
        GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, prior, 0);
        GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
        try {
            GLES20.glUseProgram(program); GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uFaceCount"), count);
            GLES20.glUniform2f(viewportLocation, viewport[2], viewport[3]);
            GLES20.glUniform1i(blurSamplerLocation, 1);
            if (count > 0) {
                float[] c = new float[count * 2], x = new float[count * 2], y = new float[count * 2];
                for (int i = 0; i < count; i++) {
                    int src = i * FACE_STRIDE, dst = i * 2;
                    c[dst] = faces[src]; c[dst + 1] = faces[src + 1];
                    x[dst] = faces[src + 2]; x[dst + 1] = faces[src + 3];
                    y[dst] = faces[src + 4]; y[dst + 1] = faces[src + 5];
                }
                GLES20.glUniform2fv(center, count, c, 0); GLES20.glUniform2fv(axisX, count, x, 0);
                GLES20.glUniform2fv(axisY, count, y, 0);
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
        for (Class<?> current = type; current != null; current = current.getSuperclass()) {
            try { Field field = current.getDeclaredField(name); field.setAccessible(true); return field; }
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

    public static synchronized void onUnload() {
        acceptingFrames = false;
        for (XC_MethodHook.Unhook hook : HOOKS) try { hook.unhook(); } catch (Throwable ignored) { }
        HOOKS.clear();
        CapturedFrame queued = LATEST_FRAME.getAndSet(null);
        if (queued != null) FRAME_POOL.offer(queued.rgba);
        SubmittedFrame submitted = IN_FLIGHT.getAndSet(null);
        if (submitted != null) { submitted.image.close(); submitted.bitmap.recycle(); }
        ExecutorService executor = frameExecutor; frameExecutor = null;
        stopExecutor(executor, "MediaPipe input");
        FaceLandmarker active = landmarker; landmarker = null;
        if (active != null) try { active.close(); } catch (Throwable error) { emit("FaceLandmarker close failed", error); }
        DRAIN_SCHEDULED.set(false); RESULTS.clear(); HELD_RESULTS.clear(); SOURCE_ACTIVE_SINCE.clear();
        SOURCE_BY_SLOT.clear(); SOURCE_BY_TEXTURE.clear();
        CAMERA_STATES.clear(); ENCODER_STATES.clear(); FRAME_POOL.clear(); initialized = false;
    }

    private static void stopExecutor(ExecutorService executor, String name) {
        if (executor == null) return;
        executor.shutdown();
        try {
            if (!executor.awaitTermination(3, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                if (!executor.awaitTermination(1, TimeUnit.SECONDS)) emit(name + " executor did not stop");
            }
        } catch (InterruptedException error) { executor.shutdownNow(); Thread.currentThread().interrupt(); }
    }

    private static final class CameraState {
        final CleanFrameTap tap = new CleanFrameTap();
        final float[] faces = new float[MAX_FACES * FACE_STRIDE];
        int program, position, texture, mvp, st, center, axisX, axisY, count, viewport, blurSampler, faceCount;
        int savedProgram, savedPosition, savedTexture, savedMvp, savedSt, blurTexture, savedBlurBinding;
        boolean swapActive, blurBindingActive, pipelineLogged; long lastCaptureNanos; ByteBuffer readPixels; String activeSource;
    }

    private static final class EncoderState {
        final CleanFrameTap tap = new CleanFrameTap();
        final float[] faces = new float[MAX_FACES * FACE_STRIDE];
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

    private static final class SubmittedFrame {
        final long captureNanos, timestampMs; final String sourceKey; final Bitmap bitmap; final MPImage image;
        SubmittedFrame(long captureNanos, long timestampMs, String sourceKey, Bitmap bitmap, MPImage image) {
            this.captureNanos = captureNanos; this.timestampMs = timestampMs; this.sourceKey = sourceKey;
            this.bitmap = bitmap; this.image = image;
        }
    }

    private static final class FaceGeometry {
        final float[] faces; final int count; final long captureNanos; final String sourceKey;
        FaceGeometry(float[] faces, int count, long captureNanos, String sourceKey) {
            this.faces = faces; this.count = count; this.captureNanos = captureNanos; this.sourceKey = sourceKey;
        }
    }
}
