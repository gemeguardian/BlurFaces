package com.makey.blurfaces;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.GLES20;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedBridge;
import org.telegram.messenger.ApplicationLoader;

/** Preview-only, scoped CameraGLThread blur. No GLES symbols are hooked. */
public final class Main {
    private static final String TAG = "BlurFaces";
    private static final String CAMERA_GL_THREAD =
            "org.telegram.ui.Components.InstantCameraView$CameraGLThread";
    private static final String ENCODER_RENDERER =
            "org.telegram.ui.Components.InstantCameraView$EncoderRenderer";
    // RoundVideoEncoder is packaged in the host's camera module; resolve the
    // snapshot through the renderer's method signature when a split classloader
    // hides it from Main's loader.
    private static final String FRAME_SNAPSHOT =
            "com.exteragram.messenger.camera.RoundVideoEncoder$FrameSnapshot";
    private static final int MAX_FACES = 10;
    private static final int FACE_ANCHOR_STRIDE = 14;
    private static final int FACE_AFFINE_STRIDE = 6; // center.xy, axisX.xy, axisY.xy
    // Mesh is an optional effect payload. It never replaces the five-point
    // privacy root, because a detailed tracker can be less stable during fast motion.
    private static final int MESH_STRIDE = 3;
    private static final int MESH_POINTS = 478;
    private static final int MESH_FLOATS = MESH_POINTS * MESH_STRIDE;
    private static final long MAX_MESH_AGE_NS = 140_000_000L;
    // Capture runs at 30 Hz, so this schedules SCRFD at roughly 15 Hz while
    // LK handles the intervening frames. 4 would make reacquisition only 7.5 Hz.
    private static final int DETECT_EVERY = 2;
    // Readback is a GPU/CPU barrier. Keep it off the hot render path except at
    // this bounded cadence; 320x320 remains the coordinate system sent to JNI.
    private static final long CAPTURE_INTERVAL_NS = 33_333_333L; // 30 Hz latest-frame lane
    // A camera switch must be decided at the first render boundary. This path
    // deliberately waits for one SCRFD pass rather than displaying an unclassified
    // camera frame or replacing the entire circle with an unrelated effect.
    private static final long MAX_RESULT_AGE_NS = 180_000_000L;
    // The detector cannot identify the back of a head. Keep the latest reliable
    // head mask briefly through that transition instead of using the full-frame
    // emergency cover; after this bound, return to the untouched camera image.
    private static final long LOST_HEAD_HOLD_NS = 420_000_000L;
    // Prediction covers the measured worker/readback handoff, but remains short
    // enough that a lost face cannot be carried across a whole gesture.
    private static final float MAX_PREDICT_SECONDS = 0.20f;
    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;
    private static final List<XC_MethodHook.Unhook> HOOKS = new ArrayList<>();
    private static final Map<Object, CameraState> STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, CameraState>());
    private static final Map<Object, EncoderState> ENCODER_STATES =
            Collections.synchronizedMap(new WeakHashMap<Object, EncoderState>());
    private static volatile boolean initialized;
    private static volatile boolean acceptingFrames;
    private static Consumer<String> logger;
    // The worker owns every NativeBridge process/read call. The GL thread only
    // replaces this single slot, so a slow inference can never build a backlog.
    private static final AtomicReference<CapturedFrame> LATEST_FRAME = new AtomicReference<>();
    private static final AtomicReference<FaceResult> LATEST_RESULT = new AtomicReference<>();
    // A camera switch forces the first captured frame through SCRFD immediately.
    // This is only a detector-cadence gate: the renderer must never turn it into
    // a whole-circle effect while waiting for the result.
    private static final AtomicReference<String> PENDING_CAMERA_SOURCE = new AtomicReference<>();
    // FrameSnapshot.cameraId is NOT the CameraGLThread.onDraw cameraId on every
    // host path: with both cameras open the host writes surfaceIndex into the
    // snapshot. Bind encoder frames by their surface slot instead. Generation and
    // SurfaceTexture identity keep a reused slot from accepting an old result.
    private static final Map<Integer, String> ACTIVE_SOURCE_BY_SURFACE = new ConcurrentHashMap<>();
    private static final AtomicReference<MeshResult> LATEST_MESH = new AtomicReference<>();
    private static final AtomicReference<MeshFrame> LATEST_MESH_FRAME = new AtomicReference<>();
    private static final AtomicBoolean WORKER_SCHEDULED = new AtomicBoolean();
    private static final AtomicBoolean MESH_WORKER_SCHEDULED = new AtomicBoolean();

    private static final ArrayBlockingQueue<ByteBuffer> FRAME_POOL = new ArrayBlockingQueue<>(3);
    private static final ArrayBlockingQueue<ByteBuffer> MESH_FRAME_POOL = new ArrayBlockingQueue<>(2);
    // Root and mesh must never share a worker. The privacy root is latency-critical;
    // mesh can skip frames, but cannot delay the next detector/tracker handoff.
    private static ExecutorService worker, meshWorker;
    private static long workerFrames, workerTotalNs;
    private static FaceResult workerPrevious;
    private static int workerMisses;
    // Created from an immutable verified .task file; mesh-worker-only use.
    private static FaceLandmarker meshLandmarker;
    private static String meshModelPath;
    private static volatile boolean meshRuntimeLoaded, meshAvailable;

    // Exact host names in InstantCameraView.CameraGLThread.
    private static final String VS =
            "uniform mat4 uMVPMatrix; uniform mat4 uSTMatrix;\n" +
            "attribute vec4 aPosition; attribute vec4 aTextureCoord; varying vec2 vTextureCoord;\n" +
            "void main() { gl_Position = uMVPMatrix * aPosition; " +
            "vTextureCoord = (uSTMatrix * aTextureCoord).xy; }\n";
    private static final String FS =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float; varying vec2 vTextureCoord; uniform samplerExternalOES sTexture;\n" +
            "uniform vec2 uFaceCenter[" + MAX_FACES + "]; uniform vec2 uFaceAxisX[" + MAX_FACES + "]; uniform vec2 uFaceAxisY[" + MAX_FACES + "]; uniform int uFaceCount; uniform int uEmergencyCover; uniform vec2 uViewport;\n" +
            "vec2 clampUv(vec2 uv) { return clamp(uv, 0.0, 1.0); }\n" +
            "vec4 privacyBlur(vec2 uv) { vec2 r = vec2(26.0) / max(uViewport, vec2(1.0));\n" +
            " vec4 b = texture2D(sTexture, clampUv(uv)) * 0.16;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2( r.x, 0.0))) * 0.105;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2(-r.x, 0.0))) * 0.105;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2(0.0,  r.y))) * 0.105;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2(0.0, -r.y))) * 0.105;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2( r.x,  r.y))) * 0.08;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2(-r.x,  r.y))) * 0.08;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2( r.x, -r.y))) * 0.08;\n" +
            " b += texture2D(sTexture, clampUv(uv + vec2(-r.x, -r.y))) * 0.08; return b; }\n" +
            "vec4 pixelate(vec2 uv) { vec2 c = vec2(1.0 / 24.0); vec2 p = (floor(uv / c) + .5) * c; return texture2D(sTexture, clampUv(p)); }\n" +
            "vec4 privacy(vec2 uv) { return mix(privacyBlur(uv), pixelate(uv), 0.72); }\n" +
            "void main() { vec4 src = texture2D(sTexture, vTextureCoord);\n" +
            " if (uEmergencyCover != 0) { gl_FragColor=privacy(vTextureCoord); return; }\n" +
            " if (uFaceCount <= 0 || uViewport.x < 1.0 || uViewport.y < 1.0) { gl_FragColor = src; return; }\n" +
            " vec2 p = vec2(gl_FragCoord.x / uViewport.x, 1.0 - gl_FragCoord.y / uViewport.y); float cover = 0.0;\n" +
            " for (int i=0; i<" + MAX_FACES + "; ++i) { if (i >= uFaceCount) break; vec2 q=p-uFaceCenter[i]; vec2 ax=uFaceAxisX[i]; vec2 ay=uFaceAxisY[i]; float det=ax.x*ay.y-ax.y*ay.x; if(abs(det)<.000001) continue; vec2 local=vec2((q.x*ay.y-q.y*ay.x)/det,(-q.x*ax.y+q.y*ax.x)/det); cover=max(cover,1.0-smoothstep(.84,1.0,length(local))); }\n" +
            " gl_FragColor=mix(src,privacy(vTextureCoord),cover); }\n";

    // Encoder shader keeps every uniform used by EncoderRenderer's cached host
    // locations. The host performs two draws (old camera during flip, then the
    // current camera), so the same shader protects both without a second draw.
    private static final String ENCODER_FS =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision highp float; varying vec2 vTextureCoord; uniform samplerExternalOES sTexture;\n" +
            "uniform vec2 preview; uniform vec2 resolution; uniform float alpha; uniform vec2 texelSize;\n" +
            "uniform vec2 uFaceCenter[" + MAX_FACES + "]; uniform vec2 uFaceAxisX[" + MAX_FACES + "]; uniform vec2 uFaceAxisY[" + MAX_FACES + "]; uniform int uFaceCount; uniform int uEmergencyCover; uniform vec2 uViewport;\n" +
            "vec2 clampUv2(vec2 uv) { return clamp(uv, 0.0, 1.0); }\n" +
            // The encoder output is larger than the 320px detector tap. The old
            // radius was tied to output pixels, so it became proportionally tiny
            // in the uploaded video. Use the source texel size and a wider kernel.
            // The encoded frame is the final privacy boundary. Use a radius and
            // block size that remain visible after Telegram's video re-encode;
            // the previous kernel softened pixels but left eyes and mouth readable.
            "vec4 privacyBlur2(vec2 uv) { vec2 r=max(texelSize*128.0,vec2(0.035));\n" +
            " vec4 b=texture2D(sTexture,clampUv2(uv))*0.10;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2( r.x,0.0)))*0.09;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(-r.x,0.0)))*0.09;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(0.0, r.y)))*0.09;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(0.0,-r.y)))*0.09;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2( r.x, r.y)))*0.08;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(-r.x, r.y)))*0.08;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2( r.x,-r.y)))*0.08;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(-r.x,-r.y)))*0.08;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(2.0*r.x,0.0)))*0.055;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(-2.0*r.x,0.0)))*0.055;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(0.0,2.0*r.y)))*0.055;\n" +
            " b+=texture2D(sTexture,clampUv2(uv+vec2(0.0,-2.0*r.y)))*0.055; return b; }\n" +
            "vec4 pixelate2(vec2 uv) { vec2 frameSize=max(preview,vec2(1.0)); vec2 outSize=max(resolution,vec2(1.0)); vec2 cell=max(1.0/max(frameSize,outSize),max(texelSize*72.0,vec2(0.045))); vec2 p=(floor(uv/cell)+.5)*cell; return texture2D(sTexture,clampUv2(p)); }\n" +
            "vec4 privacy2(vec2 uv) { return mix(privacyBlur2(uv),pixelate2(uv),0.78); }\n" +
            "void main(){ vec4 src=texture2D(sTexture,vTextureCoord);\n" +
            "if(uEmergencyCover!=0){gl_FragColor=vec4(privacy2(vTextureCoord).rgb*alpha,alpha);return;}\n" +
            "if(uFaceCount<=0||uViewport.x<1.0||uViewport.y<1.0){gl_FragColor=vec4(src.rgb*alpha,alpha);return;}\n" +
            "vec2 p=vec2(gl_FragCoord.x/uViewport.x,1.0-gl_FragCoord.y/uViewport.y);float cover=0.0;\n" +
            "for(int i=0;i<" + MAX_FACES + ";++i){if(i>=uFaceCount)break;vec2 q=p-uFaceCenter[i];vec2 ax=uFaceAxisX[i];vec2 ay=uFaceAxisY[i];float det=ax.x*ay.y-ax.y*ay.x;if(abs(det)<.000001)continue;vec2 local=vec2((q.x*ay.y-q.y*ay.x)/det,(-q.x*ax.y+q.y*ax.x)/det);cover=max(cover,1.0-smoothstep(.58,1.0,length(local)));}\n" +
            "vec4 outColor=mix(src,privacy2(vTextureCoord),cover);gl_FragColor=vec4(outColor.rgb*alpha,alpha);}\n";

    private Main() { }
    public static void setLogger(Consumer<String> value) { logger = value; }
    private static void emit(String s) { Log.i(TAG, s); if (logger != null) try { logger.accept("[BlurFaces] " + s); } catch (Throwable ignored) { } }
    private static void emit(String s, Throwable t) { Log.e(TAG, s, t); if (logger != null) try { logger.accept("[BlurFaces] " + s + ": " + t); } catch (Throwable ignored) { } }

    public static synchronized void initAndStart(String soPath, String modelParam, String modelBin,
                                                    String meshTaskPath, String meshNativePath) {
        if (initialized) return;
        try {
            // Native initialization deliberately precedes hook registration. A failed
            // model leaves Telegram entirely unhooked instead of a pass-through build.
            System.load(soPath);
            int nativeResult = NativeBridge.init(modelParam, modelBin);
            if (nativeResult != 0) throw new IllegalStateException("NativeBridge.init=" + nativeResult);
            // The MediaPipe JNI payload is optional. A bad/missing dense runtime is
            // recorded and disables only effects; it can never stop the privacy root.
            // FaceLandmarker performs System.loadLibrary("mediapipe_tasks_vision_jni")
            // from this DEX loader. Its native search path is the verified mesh
            // cache directory passed by DexLoader, so do not preload it through
            // Main's loader under a different ownership record.
            meshRuntimeLoaded = meshNativePath != null && !meshNativePath.isEmpty();
            if (!meshRuntimeLoaded) emit("Dense mesh JNI unavailable; stable root continues");
            meshModelPath = meshTaskPath;
            startWorker();
            acceptingFrames = true;
            hookCameraGLThread();
            hookEncoderRenderer();
            initialized = true;
            emit("Live blur armed: stable 5-point privacy root + optional dense mesh effect layer");
        } catch (Throwable t) {
            emit("Preview blur unavailable; host camera left untouched", t);
            onUnload();
        }
    }

    private static void hookCameraGLThread() throws Exception {
        Class<?> type = Class.forName(CAMERA_GL_THREAD, false, Main.class.getClassLoader());
        Method draw = type.getDeclaredMethod("onDraw", Integer.class, boolean.class, boolean.class);
        draw.setAccessible(true);
        HOOKS.add(XposedBridge.hookMethod(draw, new XC_MethodHook() {
        // The first callback has no current camera EGL context, so installation
        // is bootstrapped after that host draw. Every later callback starts with
        // CameraGLThread's context still current; arm the replacement BEFORE the
        // host executes glUseProgram/draw/swap. Merely mutating fields after swap
        // never proves that the visible draw consumed our program.
        @Override public void beforeHookedMethod(MethodHookParam p) {
            CameraState s = stateFor(p.thisObject);
            if (s.installed && EGL14.eglGetCurrentContext() != EGL14.EGL_NO_CONTEXT) {
                prepareNextPreviewDraw(p.thisObject, (Integer) p.args[0]);
                if (!s.drawArmedLogged) {
                    s.drawArmedLogged = true;
                    emit("Preview replacement armed before visible host draw EGL=" + s.context
                            + " program=" + s.program);
                }
            }
        }
        @Override public void afterHookedMethod(MethodHookParam p) {
            afterDraw(p.thisObject, (Integer) p.args[0]);
        }
        }));
        emit("Hook registered: CameraGLThread.onDraw(Integer, boolean, boolean), preview scope only");
    }

    /**
     * The encoder has a different EGL context. We only replace its own shader
     * after the host creates it and never carry preview GL names into it.
     * Failure is optional: preview remains available and the host encoder is
     * left untouched.
     */
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
                    encoderSurfaceCreated(p.thisObject, (Integer) p.args[0], (Integer) p.args[1]);
                }
            }));
            HOOKS.add(XposedBridge.hookMethod(draw, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam p) {
                    beforeEncoderDraw(p.thisObject, p.args[1]);
                }
            }));
            HOOKS.add(XposedBridge.hookMethod(destroyed, new XC_MethodHook() {
                @Override public void beforeHookedMethod(MethodHookParam p) {
                    encoderSurfaceDestroyed(p.thisObject);
                }
            }));
            emit("Encoder hooks registered: onEncoderSurfaceCreated/onDrawEncoderFrame/onEncoderSurfaceDestroyed");
        } catch (Throwable t) {
            emit("Encoder blur unavailable; preview remains active and host encoder is untouched", t);
        }
    }

    private static void encoderSurfaceCreated(Object renderer, int width, int height) {
        EncoderState e = encoderStateFor(renderer);
        try {
            if (e.program != 0) GLES20.glDeleteProgram(e.program);
            e.program = 0; e.hostCaptured = false; e.ready = false;
            e.width = width; e.height = height;
            e.context = String.valueOf(EGL14.eglGetCurrentContext());
            e.program = createEncoderProgram();
            if (e.program == 0) throw new IllegalStateException("encoder shader program failed");
            e.position = GLES20.glGetAttribLocation(e.program, "aPosition");
            e.texture = GLES20.glGetAttribLocation(e.program, "aTextureCoord");
            e.mvp = GLES20.glGetUniformLocation(e.program, "uMVPMatrix");
            e.st = GLES20.glGetUniformLocation(e.program, "uSTMatrix");
            e.preview = GLES20.glGetUniformLocation(e.program, "preview");
            e.resolution = GLES20.glGetUniformLocation(e.program, "resolution");
            e.alpha = GLES20.glGetUniformLocation(e.program, "alpha");
            e.texel = GLES20.glGetUniformLocation(e.program, "texelSize");
            e.faceCenter = GLES20.glGetUniformLocation(e.program, "uFaceCenter");
            e.faceAxisX = GLES20.glGetUniformLocation(e.program, "uFaceAxisX");
            e.faceAxisY = GLES20.glGetUniformLocation(e.program, "uFaceAxisY");
            e.count = GLES20.glGetUniformLocation(e.program, "uFaceCount");
            e.cover = GLES20.glGetUniformLocation(e.program, "uEmergencyCover");
            e.viewport = GLES20.glGetUniformLocation(e.program, "uViewport");
            if (e.position < 0 || e.texture < 0 || e.mvp < 0 || e.st < 0 || e.preview < 0 || e.resolution < 0 || e.alpha < 0 || e.texel < 0 || e.faceCenter < 0 || e.faceAxisX < 0 || e.faceAxisY < 0 || e.count < 0 || e.cover < 0 || e.viewport < 0) {
                throw new IllegalStateException("encoder shader interface incomplete");
            }
            e.ready = true;
            emit("Encoder shader ready EGL=" + e.context + " size=" + width + "x" + height);
        } catch (Throwable t) {
            e.ready = false;
            restoreEncoderInterface(renderer, e);
            emit("Encoder shader setup failed; original encoder draw retained", t);
        }
    }

    private static void beforeEncoderDraw(Object renderer, Object snapshot) {
        EncoderState e = encoderStateFor(renderer);
        if (!e.ready || snapshot == null) return;
        try {
            Class<?> c = snapshot.getClass();
            int surfaceIndex = field(c, "surfaceIndex").getInt(snapshot);
            int textureId = field(c, "textureId").getInt(snapshot);
            if (textureId <= 0) return;
            if (!e.hostCaptured) {
                e.hostProgram = field(renderer.getClass(), "drawProgram").getInt(renderer);
                e.hostPosition = field(renderer.getClass(), "positionHandle").getInt(renderer);
                e.hostTexture = field(renderer.getClass(), "textureHandle").getInt(renderer);
                e.hostPreview = field(renderer.getClass(), "previewSizeHandle").getInt(renderer);
                e.hostResolution = field(renderer.getClass(), "resolutionHandle").getInt(renderer);
                e.hostAlpha = field(renderer.getClass(), "alphaHandle").getInt(renderer);
                e.hostMvp = field(renderer.getClass(), "vertexMatrixHandle").getInt(renderer);
                e.hostSt = field(renderer.getClass(), "textureMatrixHandle").getInt(renderer);
                e.hostTexel = field(renderer.getClass(), "texelSizeHandle").getInt(renderer);
                e.hostCaptured = true;
            }
            // Program-local handles are swapped atomically; host's method still
            // performs the sole draw and the encoder's own EGL context owns e.program.
            field(renderer.getClass(), "drawProgram").setInt(renderer, e.program);
            field(renderer.getClass(), "positionHandle").setInt(renderer, e.position);
            field(renderer.getClass(), "textureHandle").setInt(renderer, e.texture);
            field(renderer.getClass(), "previewSizeHandle").setInt(renderer, e.preview);
            field(renderer.getClass(), "resolutionHandle").setInt(renderer, e.resolution);
            field(renderer.getClass(), "alphaHandle").setInt(renderer, e.alpha);
            field(renderer.getClass(), "vertexMatrixHandle").setInt(renderer, e.mvp);
            field(renderer.getClass(), "textureMatrixHandle").setInt(renderer, e.st);
            field(renderer.getClass(), "texelSizeHandle").setInt(renderer, e.texel);
            FaceResult result = LATEST_RESULT.get();
            long now = System.nanoTime();
            String logicalSource = ACTIVE_SOURCE_BY_SURFACE.get(surfaceIndex);
            if (logicalSource == null) {
                e.faceCount = 0;
                e.emergencyCover = false;
                uploadEncoderPrivacyUniforms(e);
                return;
            }
            // A back-of-head turn is an expected detector miss, not evidence that
            // the whole camera image needs obscuring. Keep the last head-local mask
            // briefly; after it expires, draw the untouched frame rather than a
            // full-circle emergency blur.
            FaceResult held = heldHeadFor(logicalSource, now);
            // A detector-confirmed empty scene is newer evidence than any held
            // geometry. The old ordering replaced that zero with LAST_RELIABLE_HEAD,
            // so the encoder kept pixelating an empty area after the face left.
            boolean currentConfirmedZero = result != null
                    && logicalSource.equals(result.sourceKey)
                    && result.count == 0 && result.authoritative;
            if (!currentConfirmedZero
                    && (result == null || result.count == 0) && held != null) result = held;
            e.faceCount = 0;
            boolean confirmedZero = result != null && result.count == 0 && result.authoritative;
            boolean fresh = result != null && now - result.captureNanos <= LOST_HEAD_HOLD_NS;
            // The encoder has its own EGL context. A preview result is tagged with
            // The result is tagged with preview EGL, while the encoder can own
            // another EGL context. Compare only the logical camera identity
            // (surface index + OES texture) and keep EGL for resource ownership.
            boolean sameEncoderSource = result != null && logicalSource.equals(result.sourceKey);
            // The switch still forces an immediate SCRFD pass, but video remains
            // pass-through during that one-result gap. Whole-circle blur is never
            // an acceptable substitute for a head-local mask.
            boolean awaitingSwitchScan = logicalSource.equals(PENDING_CAMERA_SOURCE.get());
            e.emergencyCover = false;

            if (sameEncoderSource && fresh && !confirmedZero) {
                // Detector/track recovery has no reliable face geometry yet. Keep
                // the head-local hold above when available; otherwise leave the
                // original encoder frame visible instead of covering the circle.
                e.emergencyCover = false;
                e.faceCount = result.count;
                float ahead = Math.min(MAX_PREDICT_SECONDS, Math.max(0.0f, (now - result.captureNanos) / 1_000_000_000.0f));
                for (int i = 0; i < result.count; i++) {
                    int o = i * FACE_AFFINE_STRIDE;
                    // Prediction operates on a complete face-aligned basis: center,
                    // scale and roll move together instead of translating a box.
                    for (int j = 0; j < FACE_AFFINE_STRIDE; ++j) e.facesData[o + j] = result.faces[o + j] + result.velocity[o + j] * ahead;
                }
            }
            uploadEncoderPrivacyUniforms(e);
            if (!e.captureLogged) { e.captureLogged = true; emit("Encoder frame protection active EGL=" + e.context); }
        } catch (Throwable t) {
            restoreEncoderInterface(renderer, e);
            e.ready = false;
            emit("Encoder shader disabled; original encoder draw restored", t);
        }
    }

    private static void uploadEncoderPrivacyUniforms(EncoderState e) {
        int[] viewport = new int[4]; GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
        int[] previousProgram = new int[1];
        GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, previousProgram, 0);
        GLES20.glUseProgram(e.program);
        GLES20.glUniform1i(e.count, e.faceCount);
        GLES20.glUniform1i(e.cover, e.emergencyCover ? 1 : 0);
        GLES20.glUniform2f(e.viewport, viewport[2] > 0 ? viewport[2] : e.width, viewport[3] > 0 ? viewport[3] : e.height);
        uploadFaceBasis(e.faceCount, e.facesData, e.faceCenter, e.faceAxisX, e.faceAxisY);
        GLES20.glUseProgram(previousProgram[0]);
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static int createEncoderProgram() {
        int vertex = compile(GLES20.GL_VERTEX_SHADER, VS);
        int fragment = compile(GLES20.GL_FRAGMENT_SHADER, ENCODER_FS);
        if (vertex == 0 || fragment == 0) {
            if (vertex != 0) GLES20.glDeleteShader(vertex);
            if (fragment != 0) GLES20.glDeleteShader(fragment);
            return 0;
        }
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertex); GLES20.glAttachShader(program, fragment); GLES20.glLinkProgram(program);
        GLES20.glDeleteShader(vertex); GLES20.glDeleteShader(fragment);
        int[] linked = new int[1]; GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) { emit("encoder shader link: " + GLES20.glGetProgramInfoLog(program)); GLES20.glDeleteProgram(program); return 0; }
        return program;
    }

    private static void encoderSurfaceDestroyed(Object renderer) {
        EncoderState e = encoderStateFor(renderer);
        restoreEncoderInterface(renderer, e);
        e.ready = false;
    }

    private static void restoreEncoderInterface(Object renderer, EncoderState e) {
        try {
            if (e.hostCaptured) {
                field(renderer.getClass(), "drawProgram").setInt(renderer, e.hostProgram);
                field(renderer.getClass(), "positionHandle").setInt(renderer, e.hostPosition);
                field(renderer.getClass(), "textureHandle").setInt(renderer, e.hostTexture);
                field(renderer.getClass(), "previewSizeHandle").setInt(renderer, e.hostPreview);
                field(renderer.getClass(), "resolutionHandle").setInt(renderer, e.hostResolution);
                field(renderer.getClass(), "alphaHandle").setInt(renderer, e.hostAlpha);
                field(renderer.getClass(), "vertexMatrixHandle").setInt(renderer, e.hostMvp);
                field(renderer.getClass(), "textureMatrixHandle").setInt(renderer, e.hostSt);
                field(renderer.getClass(), "texelSizeHandle").setInt(renderer, e.hostTexel);
            }
            if (e.program != 0) GLES20.glDeleteProgram(e.program);
        } catch (Throwable ignored) { }
        e.program = 0; e.hostCaptured = false;
    }

    /**
     * Prepare the next host preview draw while CameraGLThread's EGL context is
     * current. The just-finished frame is necessarily pass-through; subsequent
     * frames use the installed replacement and the latest uploaded uniforms.
     */
    private static void prepareNextPreviewDraw(Object thread, int cameraId) {
        CameraState s = stateFor(thread);
        try {
            if (s.disabled) return;
            if (!s.installed) install(thread, s);
            if (s.installed) {
                FaceResult result = LATEST_RESULT.get();
                long now = System.nanoTime();
                String source = currentSourceKey(thread);
                if (!source.isEmpty() && !source.equals(s.sourceKey)) {
                    Object outer = field(thread.getClass(), "this$0").get(thread);
                    int sourceIndex = field(outer.getClass(), "surfaceIndex").getInt(outer);
                    invalidateForSourceSwitch(sourceIndex, source);
                    s.sourceKey = source;
                    s.faceCount = 0;
                    // beforeDraw precedes SurfaceTexture.updateTexImage(), so its
                    // matrix still belongs to the prior camera frame. Mark the
                    // transition here and scan in the after-hook, when the host has
                    // committed the matching texture/matrix pair.
                    PENDING_CAMERA_SOURCE.set(source);
                    s.emergencyCover = false;
                }
                // Prefer a recent geometry-bearing result while SCRFD reports a
                // zero during a profile/back-of-head turn. A detector zero alone
                // has no safe region to blur, while the held result remains local
                // to the last seen head and expires in under half a second.
                FaceResult held = heldHeadFor(source, now);
                boolean currentConfirmedZero = result != null
                        && source.equals(result.sourceKey)
                        && result.count == 0 && result.authoritative;
                if (!currentConfirmedZero
                        && (result == null || !source.equals(result.sourceKey) || result.count == 0)
                        && held != null) result = held;
                boolean sameSource = result != null && source.equals(result.sourceKey);
                boolean fresh = sameSource && now - result.captureNanos <= LOST_HEAD_HOLD_NS;
                // PENDING_CAMERA_SOURCE controls detector cadence only. In 1.8.2 it
                // was also used as a render gate, so a valid source-bound face could
                // be published while preview was still forced to faceCount=0. The
                // worker's source ownership check already rejects retired results.
                if (sameSource && result.count == 0 && result.authoritative) {
                    s.faceCount = 0;
                    s.emergencyCover = false;
                } else if (fresh) {
                    float ahead = Math.min(MAX_PREDICT_SECONDS,
                            Math.max(0.0f, (now - result.captureNanos) / 1_000_000_000.0f));
                    s.faceCount = result.count;
                    // An unconfirmed zero is most often a yaw/profile miss. The
                    // stale branch below keeps a head-local mask when possible;
                    // it must never turn this transient state into a full-circle
                    // blur of the entire preview.
                    s.emergencyCover = false;
                    for (int i = 0; i < result.count; i++) {
                        int o = i * FACE_AFFINE_STRIDE;
                        // Keep the face-aligned basis in source space. Clamping
                        // would shear the ellipse at the viewport edge and make
                        // a mask that appears to peel off a departing face.
                        for (int j = 0; j < FACE_AFFINE_STRIDE; ++j) s.faces[o + j] = result.faces[o + j] + result.velocity[o + j] * ahead;
                    }
                } else {
                    // A full profile/back-of-head turn is a normal detector miss.
                    // Keep a recent head-local mask for the short transition, then
                    // show the original preview; never escalate this into a full
                    // circular blur that hides the whole camera image.
                    FaceResult fallbackHead = heldHeadFor(source, now);
                    if (fallbackHead != null) {
                        s.faceCount = fallbackHead.count;
                        s.emergencyCover = false;
                        for (int i = 0; i < fallbackHead.count; i++) {
                            int o = i * FACE_AFFINE_STRIDE;
                            for (int j = 0; j < FACE_AFFINE_STRIDE; ++j) s.faces[o + j] = fallbackHead.faces[o + j];
                        }
                    } else {
                        s.faceCount = 0;
                        s.emergencyCover = false;
                    }
                }
                uploadFaceUniforms(s);
                if (s.faceCount > 0 && PREVIEW_MASK_LOGGED.add(source)) {
                    emit("preview mask submitted source=" + source + " faces=" + s.faceCount
                            + " center=" + s.faces[0] + "," + s.faces[1]);
                }
            }
        } catch (Throwable t) {
            // If installation failed after changing fields, restore the exact host
            // program/locations before returning. A new CameraGLThread can retry.
            restoreHostInterface(thread, s);
            disable(s, "preview preparation failed", t);
        }
    }

    /** After host swap: throttled raw OES capture only. No JNI/CPU work on GL thread. */
    private static void afterDraw(Object thread, int cameraId) {
        CameraState s = stateFor(thread);
        if (!acceptingFrames || s.disabled) return;
        // Bootstrap installation after the first draw, when the camera EGL context
        // is guaranteed current. Later uniform updates happen in beforeHookedMethod,
        // before the host's visible glUseProgram/draw/swap sequence.
        if (!s.installed) prepareNextPreviewDraw(thread, cameraId);
        if (!s.installed || s.disabled) return;
        long now = System.nanoTime();
        if (now - s.lastCaptureNanos < CAPTURE_INTERVAL_NS) return;
        try {
            Object outer = field(thread.getClass(), "this$0").get(thread);
            int index = field(outer.getClass(), "surfaceIndex").getInt(outer);
            int[] textures = (int[]) field(outer.getClass(), "cameraTexture").get(outer);
            float[] mvp = (float[]) field(outer.getClass(), "mMVPMatrix").get(outer);
            float[] st = (float[]) field(outer.getClass(), "mSTMatrix").get(outer);
            FloatBuffer hostTex = ((FloatBuffer) field(outer.getClass(), "textureBuffer").get(outer)).duplicate();
            if (index < 0 || textures == null || index >= textures.length || textures[index] <= 0
                    || mvp == null || mvp.length != 16 || st == null || st.length != 16)
                throw new IllegalStateException("verified host source unavailable");
            float[] texCoords = new float[8]; hostTex.position(0); hostTex.get(texCoords);
            if (s.pixels == null) s.pixels = ByteBuffer.allocateDirect(CleanFrameTap.SIZE * CleanFrameTap.SIZE * 4).order(ByteOrder.nativeOrder());
            long captureStart = System.nanoTime();
            if (!s.tap.capture(textures[index], mvp, st, texCoords, s.pixels)) throw new IllegalStateException("raw OES FBO capture failed");
            ByteBuffer immutable = FRAME_POOL.poll();
            if (immutable == null) {
                s.droppedFrames++;
                return;
            }
            immutable.clear();
            s.pixels.position(0); s.pixels.limit(s.pixels.capacity()); immutable.put(s.pixels); immutable.position(0);
            long sequence = ++s.captureSequence;
            String source = sourceKey(thread, index);
            if (source.isEmpty() || !source.equals(s.sourceKey)) {
                FRAME_POOL.offer(immutable);
                throw new IllegalStateException("captured source changed during draw");
            }
            s.sourceKey = source;
            // The first captured frame after a switch must run SCRFD immediately,
            // rather than waiting for the normal every-second-frame cadence.
            boolean switchScan = source.equals(PENDING_CAMERA_SOURCE.get());
            CapturedFrame replacement = new CapturedFrame(immutable, now, source,
                    switchScan || (sequence % DETECT_EVERY) == 1);
            CapturedFrame old = LATEST_FRAME.getAndSet(replacement);
            if (old != null) FRAME_POOL.offer(old.rgba);
            scheduleWorker();
            // Best-effort low-priority copy for effects. If its two-slot pool is
            // busy, skip it; the root's latest-frame handoff remains untouched.
            ByteBuffer meshPixels = MESH_FRAME_POOL.poll();
            if (meshPixels != null) {
                meshPixels.clear(); s.pixels.position(0); s.pixels.limit(s.pixels.capacity());
                meshPixels.put(s.pixels); meshPixels.position(0);
                MeshFrame oldMesh = LATEST_MESH_FRAME.getAndSet(new MeshFrame(meshPixels, now, source));
                if (oldMesh != null) MESH_FRAME_POOL.offer(oldMesh.rgba);
                scheduleMeshWorker();
            }
            s.lastCaptureNanos = now;
            s.captureCount++;
            s.captureTotalNs += System.nanoTime() - captureStart;
            if ((s.captureCount % 120) == 0) emit("capture diag egl=" + s.context + " n=" + s.captureCount +
                    " avgReadbackMs=" + (s.captureTotalNs / s.captureCount / 1_000_000L));
            if (!s.captureLogged) { s.captureLogged = true; emit("Preview-only raw FBO capture active for EGL=" + s.context + "; detector coordinates are top-left preview UV"); }
        } catch (Throwable t) {
            // A capture/detector fault must not leave a replacement program or
            // stale mask active on the next frame. Restore the host interface;
            // the host keeps rendering while this camera's tap stays disabled.
            restoreHostInterface(thread, s);
            disable(s, "capture/detection disabled; original host preview restored", t);
        }
    }

    private static void install(Object thread, CameraState s) throws Exception {
        s.attempted = true;
        if (EGL14.eglGetCurrentContext() == EGL14.EGL_NO_CONTEXT) throw new IllegalStateException("no current EGL context");
        Field drawProgram = field(thread.getClass(), "drawProgram");
        Field positionHandle = field(thread.getClass(), "positionHandle");
        Field textureHandle = field(thread.getClass(), "textureHandle");
        Field vertexMatrixHandle = field(thread.getClass(), "vertexMatrixHandle");
        Field textureMatrixHandle = field(thread.getClass(), "textureMatrixHandle");
        int original = drawProgram.getInt(thread); if (original == 0) throw new IllegalStateException("host drawProgram is zero");
        // Snapshot host fields before any mutation: this is the fail-open path if
        // a later location lookup or reflection write throws.
        s.hostProgram = original;
        s.hostPosition = positionHandle.getInt(thread);
        s.hostTexture = textureHandle.getInt(thread);
        s.hostVertexMatrix = vertexMatrixHandle.getInt(thread);
        s.hostTextureMatrix = textureMatrixHandle.getInt(thread);
        int replacement = createProgram(); if (replacement == 0) throw new IllegalStateException("replacement program failed");
        int pos = GLES20.glGetAttribLocation(replacement, "aPosition");
        int tex = GLES20.glGetAttribLocation(replacement, "aTextureCoord");
        int mvp = GLES20.glGetUniformLocation(replacement, "uMVPMatrix");
        int stm = GLES20.glGetUniformLocation(replacement, "uSTMatrix");
        s.faceCenterUniform = GLES20.glGetUniformLocation(replacement, "uFaceCenter");
        s.faceAxisXUniform = GLES20.glGetUniformLocation(replacement, "uFaceAxisX");
        s.faceAxisYUniform = GLES20.glGetUniformLocation(replacement, "uFaceAxisY");
        s.faceCountUniform = GLES20.glGetUniformLocation(replacement, "uFaceCount");
        s.emergencyUniform = GLES20.glGetUniformLocation(replacement, "uEmergencyCover");
        s.viewportUniform = GLES20.glGetUniformLocation(replacement, "uViewport");
        if (pos < 0 || tex < 0 || mvp < 0 || stm < 0 || s.faceCenterUniform < 0 || s.faceAxisXUniform < 0 || s.faceAxisYUniform < 0 || s.faceCountUniform < 0 || s.emergencyUniform < 0 || s.viewportUniform < 0) {
            GLES20.glDeleteProgram(replacement); throw new IllegalStateException("replacement interface incomplete");
        }
        drawProgram.setInt(thread, replacement);
        positionHandle.setInt(thread, pos);
        textureHandle.setInt(thread, tex);
        vertexMatrixHandle.setInt(thread, mvp);
        textureMatrixHandle.setInt(thread, stm);
        s.program = replacement; s.context = String.valueOf(EGL14.eglGetCurrentContext()); s.installed = true;
        emit("Scoped preview shader installed EGL=" + s.context + " hostProgram=" + original + " replacement=" + replacement);
    }

    /** Temporarily bind only our replacement to set uniforms for the imminent host draw. */
    private static void uploadFaceBasis(int count, float[] data, int centerUniform, int axisXUniform, int axisYUniform) {
        if (count <= 0) return;
        float[] centers = new float[MAX_FACES * 2], axisX = new float[MAX_FACES * 2], axisY = new float[MAX_FACES * 2];
        for (int i = 0; i < count; ++i) {
            int src = i * FACE_AFFINE_STRIDE, dst = i * 2;
            centers[dst] = data[src]; centers[dst + 1] = data[src + 1];
            axisX[dst] = data[src + 2]; axisX[dst + 1] = data[src + 3];
            axisY[dst] = data[src + 4]; axisY[dst + 1] = data[src + 5];
        }
        GLES20.glUniform2fv(centerUniform, count, centers, 0);
        GLES20.glUniform2fv(axisXUniform, count, axisX, 0);
        GLES20.glUniform2fv(axisYUniform, count, axisY, 0);
    }

    private static void uploadFaceUniforms(CameraState s) {
        int[] prior = new int[1], viewport = new int[4];
        GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, prior, 0);
        GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
        try {
            GLES20.glUseProgram(s.program);
            GLES20.glUniform1i(s.faceCountUniform, s.faceCount);
            GLES20.glUniform1i(s.emergencyUniform, s.emergencyCover ? 1 : 0);
            uploadFaceBasis(s.faceCount, s.faces, s.faceCenterUniform, s.faceAxisXUniform, s.faceAxisYUniform);
            GLES20.glUniform2f(s.viewportUniform, viewport[2], viewport[3]);
        } finally { GLES20.glUseProgram(prior[0]); }
    }

    private static void restoreHostInterface(Object thread, CameraState s) {
        if (s.hostProgram == 0) return;
        try {
            field(thread.getClass(), "drawProgram").setInt(thread, s.hostProgram);
            field(thread.getClass(), "positionHandle").setInt(thread, s.hostPosition);
            field(thread.getClass(), "textureHandle").setInt(thread, s.hostTexture);
            field(thread.getClass(), "vertexMatrixHandle").setInt(thread, s.hostVertexMatrix);
            field(thread.getClass(), "textureMatrixHandle").setInt(thread, s.hostTextureMatrix);
            if (s.program != 0) GLES20.glDeleteProgram(s.program);
            s.program = 0;
            s.installed = false;
        } catch (Throwable ignored) { }
    }

    private static int createProgram() {
        int v = compile(GLES20.GL_VERTEX_SHADER, VS), f = compile(GLES20.GL_FRAGMENT_SHADER, FS);
        if (v == 0 || f == 0) { if (v != 0) GLES20.glDeleteShader(v); if (f != 0) GLES20.glDeleteShader(f); return 0; }
        int p = GLES20.glCreateProgram(); GLES20.glAttachShader(p, v); GLES20.glAttachShader(p, f); GLES20.glLinkProgram(p);
        GLES20.glDeleteShader(v); GLES20.glDeleteShader(f); int[] linked = new int[1]; GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) { emit("preview shader link: " + GLES20.glGetProgramInfoLog(p)); GLES20.glDeleteProgram(p); return 0; }
        return p;
    }
    private static int compile(int type, String text) {
        int s = GLES20.glCreateShader(type); if (s == 0) return 0; GLES20.glShaderSource(s, text); GLES20.glCompileShader(s);
        int[] ok = new int[1]; GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0);
        if (ok[0] == 0) { emit("preview shader compile: " + GLES20.glGetShaderInfoLog(s)); GLES20.glDeleteShader(s); return 0; } return s;
    }
    private static void disable(CameraState s, String why, Throwable t) { if (!s.disabled) { s.disabled = true; s.faceCount = 0; emit(why, t); } }
    private static CameraState stateFor(Object o) { synchronized (STATES) { CameraState s = STATES.get(o); if (s == null) { s = new CameraState(); STATES.put(o, s); } return s; } }
    private static Field field(Class<?> c, String n) throws NoSuchFieldException { for (Class<?> t = c; t != null; t = t.getSuperclass()) try { Field f = t.getDeclaredField(n); f.setAccessible(true); return f; } catch (NoSuchFieldException ignored) { } throw new NoSuchFieldException(n); }
    private static String sourceKey(Object thread, int index) {
        try {
            int[] generations = (int[]) field(thread.getClass(), "surfaceGeneration").get(thread);
            SurfaceTexture[] surfaces = (SurfaceTexture[]) field(thread.getClass(), "cameraSurface").get(thread);
            if (generations == null || surfaces == null || index < 0 || index >= generations.length
                    || index >= surfaces.length || surfaces[index] == null) return "";
            // Identity hash is process-local and only distinguishes a recreated
            // SurfaceTexture instance. No GL texture name crosses EGL contexts.
            return index + ":" + generations[index] + ":"
                    + System.identityHashCode(surfaces[index]);
        } catch (Throwable ignored) { return ""; }
    }
    private static String currentSourceKey(Object thread) {
        try {
            Object outer = field(thread.getClass(), "this$0").get(thread);
            int index = field(outer.getClass(), "surfaceIndex").getInt(outer);
            return sourceKey(thread, index);
        } catch (Throwable ignored) { return ""; }
    }

    private static void invalidateForSourceSwitch(int surfaceIndex, String source) {
        ACTIVE_SOURCE_BY_SURFACE.clear();
        ACTIVE_SOURCE_BY_SURFACE.put(surfaceIndex, source);
        PENDING_CAMERA_SOURCE.set(source);
        LATEST_RESULT.set(null);
        LAST_RELIABLE_HEAD.set(null);
        LATEST_MESH.set(null);
        workerPrevious = null;
        CapturedFrame pending = LATEST_FRAME.getAndSet(null);
        if (pending != null) FRAME_POOL.offer(pending.rgba);
        MeshFrame meshPending = LATEST_MESH_FRAME.getAndSet(null);
        if (meshPending != null) MESH_FRAME_POOL.offer(meshPending.rgba);
        // The first frame below is forced through full SCRFD. That detector pass
        // replaces every native tracker under the same worker/mutex, so a separate
        // reset JNI call is unnecessary and cannot become a new crash boundary.
        emit("source epoch changed surface=" + surfaceIndex + " key=" + source + "; stale state cleared");
    }

    public static synchronized void onUnload() {
        acceptingFrames = false;
        for (XC_MethodHook.Unhook h : HOOKS) try { h.unhook(); } catch (Throwable ignored) { }
        HOOKS.clear();
        CapturedFrame pending = LATEST_FRAME.getAndSet(null);
        if (pending != null) FRAME_POOL.offer(pending.rgba);
        LATEST_RESULT.set(null); LAST_RELIABLE_HEAD.set(null); PENDING_CAMERA_SOURCE.set(null); LATEST_MESH.set(null); workerPrevious = null;
        ROOT_ACTIVE_LOGGED.clear();
        PREVIEW_MASK_LOGGED.clear();
        ACTIVE_SOURCE_BY_SURFACE.clear();
        MeshFrame meshPending = LATEST_MESH_FRAME.getAndSet(null);
        if (meshPending != null) MESH_FRAME_POOL.offer(meshPending.rgba);
        meshRuntimeLoaded = false; meshAvailable = false; meshModelPath = null; meshLandmarker = null;
        boolean stopped = stopWorker() && stopMeshWorker();
        STATES.clear();
        ENCODER_STATES.clear();
        // cleanup shares the native mutex with process. Only call it after the
        // single worker has exited; on timeout, leaking native state is safer
        // than racing an in-flight JNI invocation during plugin unload.
        if (stopped) try { NativeBridge.cleanup(); } catch (Throwable ignored) { }
        else emit("worker did not stop before unload; native cleanup skipped for safety");
        initialized = false;
    }
    private static final class CameraState {
        final CleanFrameTap tap = new CleanFrameTap(); final float[] faces = new float[MAX_FACES * FACE_AFFINE_STRIDE];
        boolean attempted, installed, disabled, captureLogged, drawArmedLogged;
        int program, faceCenterUniform, faceAxisXUniform, faceAxisYUniform, faceCountUniform, emergencyUniform, viewportUniform, faceCount;
        int hostProgram, hostPosition, hostTexture, hostVertexMatrix, hostTextureMatrix;
        String context, sourceKey = ""; boolean emergencyCover; ByteBuffer pixels;
        long lastCaptureNanos, captureCount, captureTotalNs, droppedFrames, captureSequence;
    }

    private static final class EncoderState {
        int width, height, program;
        int position, texture, mvp, st, preview, resolution, alpha, texel, faceCenter, faceAxisX, faceAxisY, count, cover, viewport;
        int hostProgram, hostPosition, hostTexture, hostPreview, hostResolution, hostAlpha, hostMvp, hostSt, hostTexel, faceCount;
        boolean ready, hostCaptured, emergencyCover, captureLogged;
        String context;
        final float[] facesData = new float[MAX_FACES * FACE_AFFINE_STRIDE];
    }

    private static EncoderState encoderStateFor(Object renderer) {
        synchronized (ENCODER_STATES) {
            EncoderState state = ENCODER_STATES.get(renderer);
            if (state == null) { state = new EncoderState(); ENCODER_STATES.put(renderer, state); }
            return state;
        }
    }

    private static final class CapturedFrame {
        final ByteBuffer rgba; final long captureNanos; final String sourceKey; final boolean detect;
        CapturedFrame(ByteBuffer rgba, long captureNanos, String sourceKey, boolean detect) {
            this.rgba = rgba; this.captureNanos = captureNanos; this.sourceKey = sourceKey; this.detect = detect;
        }
    }
    private static final class MeshFrame {
        final ByteBuffer rgba; final long captureNanos; final String sourceKey;
        MeshFrame(ByteBuffer rgba, long captureNanos, String sourceKey) {
            this.rgba = rgba; this.captureNanos = captureNanos; this.sourceKey = sourceKey;
        }
    }
    private static final class MeshResult {
        // points: 478 normalized xyz landmarks. pose: 4x4 facial transform,
        // blendshapes: the model's named expression coefficients.
        final float[] points, pose, blendshapes; final long captureNanos; final String sourceKey;
        MeshResult(float[] points, float[] pose, float[] blendshapes, long captureNanos, String sourceKey) {
            this.points = points; this.pose = pose; this.blendshapes = blendshapes;
            this.captureNanos = captureNanos; this.sourceKey = sourceKey;
        }
    }
    private static final class FaceResult {
        final float[] faces, velocity;
        // Never derive privacy axes, count, or emergency state from mesh.
        final int count; final long captureNanos; final String sourceKey; final boolean authoritative;
        FaceResult(float[] faces, float[] velocity, int count, long captureNanos,
                   String sourceKey, boolean authoritative) {
            this.faces = faces; this.velocity = velocity;
            this.count = count; this.captureNanos = captureNanos; this.sourceKey = sourceKey; this.authoritative = authoritative;
        }
    }
    // Last actual face geometry is intentionally separate from LATEST_RESULT:
    // a back-of-head detector miss must not erase the mask in the same frame.
    private static final AtomicReference<FaceResult> LAST_RELIABLE_HEAD = new AtomicReference<>();
    private static final Set<String> ROOT_ACTIVE_LOGGED = ConcurrentHashMap.newKeySet();
    private static final Set<String> PREVIEW_MASK_LOGGED = ConcurrentHashMap.newKeySet();

    private static FaceResult heldHeadFor(String sourceKey, long now) {
        FaceResult held = LAST_RELIABLE_HEAD.get();
        if (held == null || held.count <= 0 || !sourceKey.equals(held.sourceKey)
                || now - held.captureNanos > LOST_HEAD_HOLD_NS) return null;
        return held;
    }

    /** Latest dense mesh for future effects. Privacy shaders intentionally never read it. */
    private static float[] meshForEffects(String sourceKey, long now) {
        MeshResult result = LATEST_MESH.get();
        if (result == null || !sourceKey.equals(result.sourceKey)
                || now - result.captureNanos > MAX_MESH_AGE_NS) return null;
        return result.points;
    }

    // One canonical face basis is shared by preview and encoder. Eyes set roll
    // and horizontal scale; nose/mouth set vertical scale and center. The result
    // is intentionally rejected if its geometry is degenerate, keeping the
    // emergency cover active instead of rendering a guessed surface mask.
    private static boolean anchorsToAffine(float[] anchors, int source, float[] affine, int target) {
        float leX = anchors[source + 4], leY = anchors[source + 5];
        float reX = anchors[source + 6], reY = anchors[source + 7];
        float noseX = anchors[source + 8], noseY = anchors[source + 9];
        float lmX = anchors[source + 10], lmY = anchors[source + 11];
        float rmX = anchors[source + 12], rmY = anchors[source + 13];
        float eyeX = reX - leX, eyeY = reY - leY;
        float eyeDistance = (float) Math.hypot(eyeX, eyeY);
        float mouthX = (lmX + rmX) * 0.5f, mouthY = (lmY + rmY) * 0.5f;
        float vertical = (float) Math.hypot(mouthX - (leX + reX) * 0.5f, mouthY - (leY + reY) * 0.5f);
        if (!Float.isFinite(eyeDistance) || !Float.isFinite(vertical) || eyeDistance < 0.012f || vertical < 0.018f) return false;
        // Center is weighted toward nose, matching the visual face surface rather
        // than the detector box. Perpendicular eye axis makes the ellipse roll.
        // A vector-weighted center preserves similarity transforms exactly: it
        // rotates and zooms with the anchors rather than blending X/Y separately.
        float centerX = noseX * 0.52f + (leX + reX) * 0.12f + mouthX * 0.24f;
        float centerY = noseY * 0.52f + (leY + reY) * 0.12f + mouthY * 0.24f;
        float ux = eyeX / eyeDistance, uy = eyeY / eyeDistance;
        // Five landmarks sit inside the face. The previous profile fix inflated
        // the ellipse symmetrically, which covered a whole head even in ordinary
        // poses. Instead, retain a face-sized radius and move that ellipse only
        // toward the detector-box side containing the extra cheek/ear contour.
        float boxX = anchors[source], boxY = anchors[source + 1];
        float boxW = anchors[source + 2], boxH = anchors[source + 3];
        if (!Float.isFinite(boxX) || !Float.isFinite(boxY) || !Float.isFinite(boxW) || !Float.isFinite(boxH)
                || boxW < 0.025f || boxH < 0.025f) return false;
        float boxCenterX = boxX + boxW * 0.5f, boxCenterY = boxY + boxH * 0.5f;
        // Project the detector-centre offset onto the eye axis. At yaw this points
        // toward the wider side of the detected head; in frontal pose it is nearly
        // zero, so the mask stays compact instead of widening on both cheeks.
        float lateral = (boxCenterX - centerX) * ux + (boxCenterY - centerY) * uy;
        float lateralLimit = boxW * 0.20f;
        lateral = Math.max(-lateralLimit, Math.min(lateralLimit, lateral));
        float rx = Math.max(eyeDistance * 1.34f, boxW * 0.62f);
        float ry = Math.max(vertical * 1.56f, boxH * 0.62f);
        centerX += ux * lateral + (-uy) * (vertical * 0.16f);
        centerY += uy * lateral + ux * (vertical * 0.16f);
        if (!Float.isFinite(centerX) || !Float.isFinite(centerY) || !Float.isFinite(rx) || !Float.isFinite(ry)) return false;
        affine[target] = centerX; affine[target + 1] = centerY;
        affine[target + 2] = ux * rx; affine[target + 3] = uy * rx;
        affine[target + 4] = -uy * ry; affine[target + 5] = ux * ry;
        return true;
    }

    private static synchronized void startWorker() {
        if (worker != null && !worker.isShutdown()) return;
        WORKER_SCHEDULED.set(false); MESH_WORKER_SCHEDULED.set(false);
        workerFrames = workerTotalNs = 0; workerPrevious = null;
        meshLandmarker = null; meshAvailable = false;
        FRAME_POOL.clear(); MESH_FRAME_POOL.clear();
        for (int i = 0; i < 3; i++) FRAME_POOL.offer(ByteBuffer.allocateDirect(CleanFrameTap.SIZE * CleanFrameTap.SIZE * 4).order(ByteOrder.nativeOrder()));
        for (int i = 0; i < 2; i++) MESH_FRAME_POOL.offer(ByteBuffer.allocateDirect(CleanFrameTap.SIZE * CleanFrameTap.SIZE * 4).order(ByteOrder.nativeOrder()));
        worker = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "BlurFacesRootWorker"); t.setDaemon(true); return t;
        });
        meshWorker = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "BlurFacesMeshWorker"); t.setDaemon(true); return t;
        });
    }
    private static void scheduleWorker() {
        ExecutorService w = worker;
        if (w == null || w.isShutdown() || !WORKER_SCHEDULED.compareAndSet(false, true)) return;
        try { w.execute(Main::drainLatestFrame); }
        catch (Throwable ignored) { WORKER_SCHEDULED.set(false); }
    }
    private static void scheduleMeshWorker() {
        ExecutorService w = meshWorker;
        if (w == null || w.isShutdown() || !MESH_WORKER_SCHEDULED.compareAndSet(false, true)) return;
        try { w.execute(Main::drainLatestMeshFrame); }
        catch (Throwable ignored) { MESH_WORKER_SCHEDULED.set(false); }
    }
    private static FaceLandmarker createMeshLandmarker() {
        if (!meshRuntimeLoaded || meshModelPath == null || meshModelPath.isEmpty()) return null;
        try {
            // The task file lives in app-private cache, not APK assets. File-channel
            // mmap keeps the model readable after the loader returns without copying it.
            FileInputStream stream = new FileInputStream(meshModelPath);
            FileChannel channel = stream.getChannel();
            ByteBuffer taskBytes = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            BaseOptions base = BaseOptions.builder().setModelAssetBuffer(taskBytes)
                    .setDelegate(Delegate.CPU).build();
            FaceLandmarker.FaceLandmarkerOptions options = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(base).setRunningMode(RunningMode.IMAGE).setNumFaces(1)
                    .setMinFaceDetectionConfidence(0.70f).setMinFacePresenceConfidence(0.70f)
                    .setMinTrackingConfidence(0.70f).setOutputFaceBlendshapes(true)
                    .setOutputFacialTransformationMatrixes(true).build();
            FaceLandmarker landmarker = FaceLandmarker.createFromOptions(
                    ApplicationLoader.applicationContext, options);
            try { channel.close(); stream.close(); } catch (IOException ignored) { }
            meshAvailable = true;
            emit("Dense mesh initialized: 478 landmarks; stable 5-point root still owns privacy");
            return landmarker;
        } catch (Throwable t) {
            meshAvailable = false;
            emit("Dense mesh unavailable; stable 5-point privacy path continues", t);
            return null;
        }
    }

    private static MeshResult detectMesh(ByteBuffer rgba, long captureNanos, String sourceKey) {
        if (meshLandmarker == null) meshLandmarker = createMeshLandmarker();
        if (meshLandmarker == null) return null;
        try {
            ByteBuffer pixels = rgba.duplicate(); pixels.position(0);
            Bitmap bitmap = Bitmap.createBitmap(CleanFrameTap.SIZE, CleanFrameTap.SIZE, Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(pixels);
            MPImage image = new BitmapImageBuilder(bitmap).build();
            FaceLandmarkerResult result = meshLandmarker.detect(image);
            if (result.faceLandmarks().isEmpty()) return null;
            List<NormalizedLandmark> points = result.faceLandmarks().get(0);
            if (points.size() != MESH_POINTS) {
                emit("Dense mesh rejected: expected " + MESH_POINTS + " points, got " + points.size());
                return null;
            }
            float[] mesh = new float[MESH_FLOATS];
            for (int i = 0; i < MESH_POINTS; ++i) {
                NormalizedLandmark p = points.get(i);
                float x = p.x(), y = p.y(), z = p.z();
                if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) return null;
                int o = i * MESH_STRIDE; mesh[o] = x; mesh[o + 1] = y; mesh[o + 2] = z;
            }
            float[] pose = null;
            if (result.facialTransformationMatrixes().isPresent()
                    && !result.facialTransformationMatrixes().get().isEmpty()) {
                float[] candidate = result.facialTransformationMatrixes().get().get(0);
                if (candidate != null && candidate.length == 16) {
                    for (float value : candidate) if (!Float.isFinite(value)) return null;
                    pose = candidate.clone();
                }
            }
            float[] blendshapes = null;
            if (result.faceBlendshapes().isPresent() && !result.faceBlendshapes().get().isEmpty()) {
                List<Category> categories = result.faceBlendshapes().get().get(0);
                blendshapes = new float[categories.size()];
                for (int i = 0; i < categories.size(); ++i) {
                    float score = categories.get(i).score();
                    if (!Float.isFinite(score)) return null;
                    blendshapes[i] = score;
                }
            }
            return new MeshResult(mesh, pose, blendshapes, captureNanos, sourceKey);
        } catch (Throwable t) {
            // Mesh failure cannot alter privacy result state: it is purely optional.
            meshAvailable = false;
            emit("Dense mesh frame rejected; stable 5-point privacy path continues", t);
            return null;
        }
    }

    private static void drainLatestMeshFrame() {
        try {
            MeshFrame frame;
            while (acceptingFrames && (frame = LATEST_MESH_FRAME.getAndSet(null)) != null) {
                try {
                    MeshResult result = detectMesh(frame.rgba, frame.captureNanos, frame.sourceKey);
                    if (result != null) LATEST_MESH.set(result);
                } finally {
                    MESH_FRAME_POOL.offer(frame.rgba);
                }
            }
        } catch (Throwable t) { emit("dense mesh worker disabled; stable root continues", t); }
        finally {
            MESH_WORKER_SCHEDULED.set(false);
            if (acceptingFrames && LATEST_MESH_FRAME.get() != null) scheduleMeshWorker();
        }
    }

    private static void drainLatestFrame() {
        try {
            CapturedFrame frame;
            while (acceptingFrames && (frame = LATEST_FRAME.getAndSet(null)) != null) {
                try {
                    long started = System.nanoTime();
                    int result = NativeBridge.process(frame.rgba, CleanFrameTap.SIZE, CleanFrameTap.SIZE, frame.detect);
                    if (result != 0) { emit("native process failed=" + result); continue; }
                    // Native may run SCRFD even on a non-scheduled frame when
                    // there are no active trackers. The Java cadence flag alone
                    // cannot tell that apart from a tracking miss.
                    boolean authoritative = NativeBridge.wasLastProcessDetection();
                    int nativeCount = Math.max(0, Math.min(MAX_FACES, NativeBridge.getLastFaceCount()));
                    float[] anchors = new float[MAX_FACES * FACE_ANCHOR_STRIDE];
                    if (nativeCount > 0) NativeBridge.getFaceAnchors(anchors);
                    float[] faces = new float[MAX_FACES * FACE_AFFINE_STRIDE];
                    int count = 0;
                    for (int i = 0; i < nativeCount; ++i) {
                        if (!anchorsToAffine(anchors, i * FACE_ANCHOR_STRIDE, faces, count * FACE_AFFINE_STRIDE)) {
                            // A claimed face without usable facial geometry is an
                            // unknown result, never an authoritative zero-face.
                            authoritative = false;
                            count = 0;
                            break;
                        }
                        ++count;
                    }
                    float[] velocity = new float[MAX_FACES * FACE_AFFINE_STRIDE];
                    if (workerPrevious != null && workerPrevious.count == count
                            && frame.sourceKey.equals(workerPrevious.sourceKey)) {
                        float dt = Math.max(0.001f, (frame.captureNanos - workerPrevious.captureNanos) / 1_000_000_000.0f);
                        for (int i = 0; i < count; i++) for (int j = 0; j < FACE_AFFINE_STRIDE; j++) {
                            int o = i * FACE_AFFINE_STRIDE + j;
                            float v = (faces[o] - workerPrevious.faces[o]) / dt;
                            velocity[o] = Math.max(-4.0f, Math.min(4.0f, v));
                        }
                    }
                    // Root publication never waits for mesh. A separate best-effort
                    // mesh worker may lag or miss; it cannot change count/authority.
                    FaceResult next = new FaceResult(faces, velocity, count,
                            frame.captureNanos, frame.sourceKey, authoritative);
                    // A switch can invalidate this job while JNI is running. Never
                    // republish it merely because a slot/index has since been reused.
                    if (!ACTIVE_SOURCE_BY_SURFACE.containsValue(frame.sourceKey)) {
                        emit("dropping late result from retired source=" + frame.sourceKey);
                        continue;
                    }
                    workerPrevious = next;
                    LATEST_RESULT.set(next);
                    if (count > 0) LAST_RELIABLE_HEAD.set(next);
                    else if (authoritative) LAST_RELIABLE_HEAD.set(null);
                    // Log the first usable root result for each logical source.
                    // Shader installation proves only that GLES mutation succeeded;
                    // this proves that detector geometry reached both renderers.
                    if (count > 0 && ROOT_ACTIVE_LOGGED.add(frame.sourceKey)) {
                        emit("live root active source=" + frame.sourceKey + " faces=" + count);
                    }
                    // Only a scheduled full detector pass may release the camera
                    // switch gate. Tracker-only frames and a prior camera's result
                    // cannot accidentally make a newly visible face pass through.
                    if (frame.detect && frame.sourceKey.equals(PENDING_CAMERA_SOURCE.get())) {
                        PENDING_CAMERA_SOURCE.compareAndSet(frame.sourceKey, null);
                        emit("camera switch scan complete source=" + frame.sourceKey + " faces=" + count);
                    }
                    workerFrames++; workerTotalNs += System.nanoTime() - started;
                    if ((workerFrames % 20) == 0) emit("worker diag n=" + workerFrames + " avgProcessMs=" +
                            (workerTotalNs / workerFrames / 1_000_000L) + " latestAgeMs=" +
                            ((System.nanoTime() - frame.captureNanos) / 1_000_000L));
                } finally {
                    FRAME_POOL.offer(frame.rgba);
                }
            }
        } catch (Throwable t) { emit("worker disabled", t); }
        finally {
            WORKER_SCHEDULED.set(false);
            // A frame may have arrived between the loop's null read and the flag
            // clear. Schedule exactly one more drain, still with latest-only policy.
            if (acceptingFrames && LATEST_FRAME.get() != null) scheduleWorker();
        }
    }
    private static synchronized boolean stopWorker() {
        ExecutorService w = worker; worker = null; WORKER_SCHEDULED.set(false);
        return stopExecutor(w, "root");
    }
    private static synchronized boolean stopMeshWorker() {
        ExecutorService w = meshWorker; meshWorker = null; MESH_WORKER_SCHEDULED.set(false);
        return stopExecutor(w, "mesh");
    }
    private static boolean stopExecutor(ExecutorService w, String name) {
        if (w == null) return true;
        w.shutdown();
        try { if (!w.awaitTermination(3, TimeUnit.SECONDS)) { w.shutdownNow(); return w.awaitTermination(1, TimeUnit.SECONDS); } }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); return false; }
        return true;
    }
}
