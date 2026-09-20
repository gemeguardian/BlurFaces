package com.makey.blurfaces.g2;

import android.opengl.GLES20;
import android.opengl.GLES30;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/** Context-local OES downsample and ping-pong Gaussian blur pipeline. */
final class CleanFrameTap {
    static final int SIZE = 192;
    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;
    private static final int GL_TEXTURE_BINDING_EXTERNAL_OES = 0x8D67;
    private static final int GL_VERTEX_ATTRIB_ARRAY_ENABLED = 0x8622;
    private static final int PBO_COUNT = 2;

    private static final String OES_VS =
            "attribute vec2 aPos;attribute vec2 aTex;uniform mat4 uMVPMatrix;uniform mat4 uSTMatrix;" +
            "varying vec2 vTex;void main(){gl_Position=uMVPMatrix*vec4(aPos,0.0,1.0);" +
            "vTex=(uSTMatrix*vec4(aTex,0.0,1.0)).xy;}";
    private static final String OES_FS =
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float;" +
            "varying vec2 vTex;uniform samplerExternalOES sTexture;" +
            "void main(){gl_FragColor=texture2D(sTexture,vTex);}";
    private static final String BLUR_VS =
            "attribute vec2 aPos;attribute vec2 aTex;varying vec2 vTex;" +
            "void main(){gl_Position=vec4(aPos,0.0,1.0);vTex=aTex;}";
    private static final String BLUR_FS =
            "precision mediump float;varying vec2 vTex;uniform sampler2D sTexture;uniform vec2 uStep;" +
            "vec2 c(vec2 p){return clamp(p,vec2(.0015),vec2(.9985));}" +
            "void main(){vec4 o=texture2D(sTexture,c(vTex))*.22702703;" +
            "o+=(texture2D(sTexture,c(vTex-uStep*1.38461538))+texture2D(sTexture,c(vTex+uStep*1.38461538)))*.31621622;" +
            "o+=(texture2D(sTexture,c(vTex-uStep*3.23076923))+texture2D(sTexture,c(vTex+uStep*3.23076923)))*.07027027;" +
            "gl_FragColor=o;}";

    private int oesProgram, blurProgram, vbo;
    private final int[] textures = new int[2];
    private final int[] fbos = new int[2];
    private final int[] pbos = new int[PBO_COUNT];
    private int pboIndex = 0;
    private boolean pboInitialized = false;
    private boolean pboPrimed = false;
    private boolean pboSupported = true;
    private int oesPos, oesTex, oesMvp, oesSt, oesSampler;
    private int blurPos, blurTex, blurSampler, blurStep;
    private boolean ready, failed;
    private String lastError = "not started";
    private static final float[] FULL_QUAD_TEX = {0f,0f, 1f,0f, 0f,1f, 1f,1f};
    private final float[] quad = new float[16];
    private final FloatBuffer quadBuffer = ByteBuffer.allocateDirect(16 * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer();

    String lastError() { return lastError; }

    static float clampBlurRadiusScale(float blurRadiusScale) {
        if (!Float.isFinite(blurRadiusScale) || blurRadiusScale < 1.0f) {
            return 1.0f;
        } else if (blurRadiusScale > 2.5f) {
            return 2.5f;
        }
        return blurRadiusScale;
    }

    /** Returns the final RGBA blur texture, or zero if this frame could not be rendered. */
    int renderBlur(int oesTexture, float[] mvpMatrix, float[] stMatrix,
                   float[] hostTexCoords, ByteBuffer cleanReadback) {
        return renderBlur(oesTexture, mvpMatrix, stMatrix, hostTexCoords, cleanReadback, 1.0f);
    }

    int renderBlur(int oesTexture, float[] mvpMatrix, float[] stMatrix,
                   float[] hostTexCoords, ByteBuffer cleanReadback, float blurRadiusScale) {
        if (oesTexture <= 0 || stMatrix == null || stMatrix.length != 16
                || mvpMatrix == null || mvpMatrix.length != 16
                || hostTexCoords == null || hostTexCoords.length != 8
                || (cleanReadback != null && cleanReadback.capacity() < SIZE * SIZE * 4)) {
            lastError = "invalid frame input";
            return 0;
        }
        blurRadiusScale = clampBlurRadiusScale(blurRadiusScale);
        if (failed) {
            release();
        }
        GLState saved = new GLState();
        try {
            drainErrors();
            saved.capture();
            // Some hosts leave a non-fatal error behind after their own draw.
            // Only errors produced after this boundary belong to this pipeline.
            drainErrors();
            if (!ensureCreated()) return 0;
            drainErrors();
            GLES20.glViewport(0, 0, SIZE, SIZE);
            GLES20.glDisable(GLES20.GL_BLEND);
            GLES20.glDisable(GLES20.GL_DEPTH_TEST);
            GLES20.glDisable(GLES20.GL_SCISSOR_TEST);
            GLES20.glDisable(GLES20.GL_CULL_FACE);

            uploadQuad(hostTexCoords);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbos[0]);
            GLES20.glUseProgram(oesProgram);
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo);
            GLES20.glVertexAttribPointer(oesPos, 2, GLES20.GL_FLOAT, false, 16, 0);
            GLES20.glVertexAttribPointer(oesTex, 2, GLES20.GL_FLOAT, false, 16, 8);
            GLES20.glEnableVertexAttribArray(oesPos);
            GLES20.glEnableVertexAttribArray(oesTex);
            GLES20.glUniformMatrix4fv(oesMvp, 1, false, mvpMatrix, 0);
            GLES20.glUniformMatrix4fv(oesSt, 1, false, stMatrix, 0);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, oesTexture);
            GLES20.glUniform1i(oesSampler, 0);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
            if (cleanReadback != null) {
                try {
                    readPixelsAsync(cleanReadback);
                } catch (Throwable readError) {
                    lastError = "readPixels error=" + readError;
                } finally {
                    drainErrors();
                }
            }

            uploadQuad(FULL_QUAD_TEX);
            float step = (4.5f * blurRadiusScale) / SIZE;
            drawBlur(textures[0], fbos[1], step, 0f);
            drawBlur(textures[1], fbos[0], 0f, step);
            drawBlur(textures[0], fbos[1], step, 0f);
            drawBlur(textures[1], fbos[0], 0f, step);
            drawBlur(textures[0], fbos[1], step, 0f);
            drawBlur(textures[1], fbos[0], 0f, step);
            int error = GLES20.glGetError();
            if (error != GLES20.GL_NO_ERROR) {
                lastError = "render GL error=0x" + Integer.toHexString(error);
                drainErrors();
                return 0;
            }
            lastError = "ok";
            return textures[0];
        } catch (Throwable error) {
            lastError = "render exception=" + error;
            return 0;
        } finally {
            saved.restore();
        }
    }

    private void readPixelsAsync(ByteBuffer cleanReadback) {
        cleanReadback.position(0);
        final int bytes = SIZE * SIZE * 4;
        if (pboSupported) {
            try {
                if (!pboInitialized) {
                    GLES30.glGenBuffers(PBO_COUNT, pbos, 0);
                    for (int i = 0; i < PBO_COUNT; i++) {
                        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbos[i]);
                        GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, bytes, null, GLES30.GL_STREAM_READ);
                    }
                    GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
                    pboInitialized = (pbos[0] != 0 && pbos[1] != 0);
                    pboIndex = 0;
                    pboPrimed = false;
                }
                if (pboInitialized) {
                    int nextIndex = (pboIndex + 1) % PBO_COUNT;
                    // Trigger asynchronous readback into currently bound PBO
                    GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbos[pboIndex]);
                    GLES30.glReadPixels(0, 0, SIZE, SIZE, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0);

                    if (pboPrimed) {
                        // Read from the other PBO (which finished transfer from previous frame)
                        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbos[nextIndex]);
                        ByteBuffer mapped = (ByteBuffer) GLES30.glMapBufferRange(
                                GLES30.GL_PIXEL_PACK_BUFFER, 0, bytes, GLES30.GL_MAP_READ_BIT);
                        if (mapped != null) {
                            try {
                                cleanReadback.put(mapped);
                                cleanReadback.position(0);
                            } finally {
                                GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER);
                            }
                        } else {
                            // If mapping wasn't ready yet or failed, direct fallback
                            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
                            GLES20.glReadPixels(0, 0, SIZE, SIZE, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, cleanReadback);
                        }
                    } else {
                        // On first frame, read directly while the PBO pipeline warms up
                        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
                        GLES20.glReadPixels(0, 0, SIZE, SIZE, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, cleanReadback);
                        pboPrimed = true;
                    }
                    GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
                    pboIndex = nextIndex;
                    return;
                }
            } catch (Throwable t) {
                pboSupported = false;
                try {
                    GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
                } catch (Throwable ignored) {}
            }
        }
        // Direct fallback if GLES30 / PBO is not supported
        try {
            GLES20.glReadPixels(0, 0, SIZE, SIZE, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, cleanReadback);
        } catch (Throwable error) {
            lastError = "direct readPixels error=" + error;
        }
    }

    private void drawBlur(int sourceTexture, int destinationFbo, float x, float y) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, destinationFbo);
        GLES20.glUseProgram(blurProgram);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo);
        GLES20.glVertexAttribPointer(blurPos, 2, GLES20.GL_FLOAT, false, 16, 0);
        GLES20.glVertexAttribPointer(blurTex, 2, GLES20.GL_FLOAT, false, 16, 8);
        GLES20.glEnableVertexAttribArray(blurPos);
        GLES20.glEnableVertexAttribArray(blurTex);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sourceTexture);
        GLES20.glUniform1i(blurSampler, 0);
        GLES20.glUniform2f(blurStep, x, y);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    }

    private boolean ensureCreated() {
        if (ready) return true;
        try {
            oesProgram = createProgram(OES_VS, OES_FS);
            blurProgram = createProgram(BLUR_VS, BLUR_FS);
            if (oesProgram == 0 || blurProgram == 0) return fail("shader program creation failed");
            oesPos = GLES20.glGetAttribLocation(oesProgram, "aPos");
            oesTex = GLES20.glGetAttribLocation(oesProgram, "aTex");
            oesMvp = GLES20.glGetUniformLocation(oesProgram, "uMVPMatrix");
            oesSt = GLES20.glGetUniformLocation(oesProgram, "uSTMatrix");
            oesSampler = GLES20.glGetUniformLocation(oesProgram, "sTexture");
            blurPos = GLES20.glGetAttribLocation(blurProgram, "aPos");
            blurTex = GLES20.glGetAttribLocation(blurProgram, "aTex");
            blurSampler = GLES20.glGetUniformLocation(blurProgram, "sTexture");
            blurStep = GLES20.glGetUniformLocation(blurProgram, "uStep");
            if (oesPos < 0 || oesTex < 0 || oesMvp < 0 || oesSt < 0 || oesSampler < 0
                    || blurPos < 0 || blurTex < 0 || blurSampler < 0 || blurStep < 0)
                return fail("shader interface incomplete");

            GLES20.glGenTextures(2, textures, 0);
            GLES20.glGenFramebuffers(2, fbos, 0);
            for (int i = 0; i < 2; i++) {
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[i]);
                GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, SIZE, SIZE, 0,
                        GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbos[i]);
                GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                        GLES20.GL_TEXTURE_2D, textures[i], 0);
                if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
                        != GLES20.GL_FRAMEBUFFER_COMPLETE) return fail("framebuffer incomplete index=" + i);
            }
            int[] id = new int[1];
            GLES20.glGenBuffers(1, id, 0); vbo = id[0];
            ready = vbo != 0 && textures[0] != 0 && textures[1] != 0
                    && fbos[0] != 0 && fbos[1] != 0;
            return ready;
        } catch (Throwable error) { return fail("initialization exception=" + error); }
    }

    private int createProgram(String vertexSource, String fragmentSource) {
        int vs = compile(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fs = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        if (vs == 0 || fs == 0) {
            if (vs != 0) GLES20.glDeleteShader(vs);
            if (fs != 0) GLES20.glDeleteShader(fs);
            return 0;
        }
        int value = GLES20.glCreateProgram();
        GLES20.glAttachShader(value, vs); GLES20.glAttachShader(value, fs);
        GLES20.glLinkProgram(value); GLES20.glDeleteShader(vs); GLES20.glDeleteShader(fs);
        int[] ok = new int[1]; GLES20.glGetProgramiv(value, GLES20.GL_LINK_STATUS, ok, 0);
        if (ok[0] == 0) {
            lastError = "program link=" + GLES20.glGetProgramInfoLog(value);
            GLES20.glDeleteProgram(value); return 0;
        }
        return value;
    }

    private void uploadQuad(float[] t) {
        quad[0] = -1f; quad[1] = -1f; quad[2] = t[0]; quad[3] = t[1];
        quad[4] = 1f; quad[5] = -1f; quad[6] = t[2]; quad[7] = t[3];
        quad[8] = -1f; quad[9] = 1f; quad[10] = t[4]; quad[11] = t[5];
        quad[12] = 1f; quad[13] = 1f; quad[14] = t[6]; quad[15] = t[7];
        quadBuffer.clear(); quadBuffer.put(quad).position(0);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, quad.length * 4, quadBuffer, GLES20.GL_STREAM_DRAW);
    }

    private int compile(int type, String source) {
        int shader = GLES20.glCreateShader(type); if (shader == 0) return 0;
        GLES20.glShaderSource(shader, source); GLES20.glCompileShader(shader);
        int[] ok = new int[1]; GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, ok, 0);
        if (ok[0] == 0) {
            lastError = "shader compile=" + GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader); return 0;
        }
        return shader;
    }

    private boolean fail(String reason) {
        if (lastError == null || "not started".equals(lastError) || "ok".equals(lastError)) lastError = reason;
        else lastError = reason + "; " + lastError;
        failed = true; release(); return false;
    }

    private static void drainErrors() {
        while (GLES20.glGetError() != GLES20.GL_NO_ERROR) { }
    }

    void resetPbo() {
        pboPrimed = false;
        pboIndex = 0;
    }

    void release() {
        try { if (vbo != 0) GLES20.glDeleteBuffers(1, new int[]{vbo}, 0); } catch (Throwable ignored) { }
        try {
            if (pbos[0] != 0 || pbos[1] != 0) {
                GLES20.glDeleteBuffers(PBO_COUNT, pbos, 0);
            }
        } catch (Throwable ignored) { }
        pboInitialized = false;
        pboPrimed = false;
        pboIndex = 0;
        pbos[0] = pbos[1] = 0;
        try { GLES20.glDeleteFramebuffers(2, fbos, 0); } catch (Throwable ignored) { }
        try { GLES20.glDeleteTextures(2, textures, 0); } catch (Throwable ignored) { }
        try { if (oesProgram != 0) GLES20.glDeleteProgram(oesProgram); } catch (Throwable ignored) { }
        try { if (blurProgram != 0) GLES20.glDeleteProgram(blurProgram); } catch (Throwable ignored) { }
        vbo = oesProgram = blurProgram = 0;
        textures[0] = textures[1] = fbos[0] = fbos[1] = 0;
        ready = false;
        failed = false;
    }

    private static final class GLState {
        final int[] active = new int[1], tex2d = new int[1], texOes = new int[1], fbo = new int[1];
        final int[] unit0Tex2d = new int[1], unit0TexOes = new int[1];
        final int[] viewport = new int[4], program = new int[1], array = new int[1], element = new int[1];
        boolean blend, depth, scissor, cull;
        int[] attribEnabled;
        void capture() {
            GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, active, 0);
            GLES20.glGetIntegerv(GLES20.GL_FRAMEBUFFER_BINDING, fbo, 0);
            GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
            GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, program, 0);
            GLES20.glGetIntegerv(GLES20.GL_ARRAY_BUFFER_BINDING, array, 0);
            GLES20.glGetIntegerv(GLES20.GL_ELEMENT_ARRAY_BUFFER_BINDING, element, 0);
            GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, tex2d, 0);
            GLES20.glGetIntegerv(GL_TEXTURE_BINDING_EXTERNAL_OES, texOes, 0);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, unit0Tex2d, 0);
            GLES20.glGetIntegerv(GL_TEXTURE_BINDING_EXTERNAL_OES, unit0TexOes, 0);
            GLES20.glActiveTexture(active[0]);
            blend = GLES20.glIsEnabled(GLES20.GL_BLEND);
            depth = GLES20.glIsEnabled(GLES20.GL_DEPTH_TEST);
            scissor = GLES20.glIsEnabled(GLES20.GL_SCISSOR_TEST);
            cull = GLES20.glIsEnabled(GLES20.GL_CULL_FACE);
            int[] max = new int[1]; GLES20.glGetIntegerv(GLES20.GL_MAX_VERTEX_ATTRIBS, max, 0);
            attribEnabled = new int[Math.max(0, max[0])]; int[] value = new int[1];
            for (int i = 0; i < attribEnabled.length; i++) {
                GLES20.glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, value, 0);
                attribEnabled[i] = value[0];
            }
        }
        void restore() {
            try {
                if (attribEnabled != null) for (int i = 0; i < attribEnabled.length; i++) {
                    if (attribEnabled[i] != 0) GLES20.glEnableVertexAttribArray(i);
                    else GLES20.glDisableVertexAttribArray(i);
                }
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0]);
                GLES20.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                GLES20.glUseProgram(program[0]);
                GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, array[0]);
                GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, element[0]);
                GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, unit0Tex2d[0]);
                GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, unit0TexOes[0]);
                GLES20.glActiveTexture(active[0]);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex2d[0]);
                GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, texOes[0]);
                if (blend) GLES20.glEnable(GLES20.GL_BLEND); else GLES20.glDisable(GLES20.GL_BLEND);
                if (depth) GLES20.glEnable(GLES20.GL_DEPTH_TEST); else GLES20.glDisable(GLES20.GL_DEPTH_TEST);
                if (scissor) GLES20.glEnable(GLES20.GL_SCISSOR_TEST); else GLES20.glDisable(GLES20.GL_SCISSOR_TEST);
                if (cull) GLES20.glEnable(GLES20.GL_CULL_FACE); else GLES20.glDisable(GLES20.GL_CULL_FACE);
            } catch (Throwable ignored) { }
            finally {
                try { GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0); } catch (Throwable ignored) { }
            }
        }
    }
}
