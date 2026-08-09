package com.makey.blurfaces;

import android.opengl.GLES20;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Context-local raw OES tap.  It is called only after CameraGLThread.onDraw has
 * completed its draw/swap boundary, never from a GLES hook.  Every GL name in
 * this object belongs to one CameraGLThread/EGL context.
 */
final class CleanFrameTap {
    static final int SIZE = 320;
    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;
    // GL_TEXTURE_EXTERNAL_OES is a texture *target*; querying its binding needs
    // the distinct GL_TEXTURE_BINDING_EXTERNAL_OES enum. Using the target as a
    // glGetIntegerv pname raises GL_INVALID_ENUM, leaves a latent GL error behind,
    // and makes the post-readback error check reject every otherwise-valid frame.
    private static final int GL_TEXTURE_BINDING_EXTERNAL_OES = 0x8D67;
    private static final int GL_VERTEX_ATTRIB_ARRAY_ENABLED = 0x8622;

    private static final String VS =
            "attribute vec2 aPos;\n" +
            "attribute vec2 aTex;\n" +
            "uniform mat4 uMVPMatrix;\n" +
            "uniform mat4 uSTMatrix;\n" +
            "varying vec2 vTex;\n" +
            "void main() { gl_Position = uMVPMatrix * vec4(aPos, 0.0, 1.0); " +
            "vTex = (uSTMatrix * vec4(aTex, 0.0, 1.0)).xy; }\n";
    private static final String FS =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float; varying vec2 vTex; uniform samplerExternalOES sTexture;\n" +
            "void main() { gl_FragColor = texture2D(sTexture, vTex); }\n";

    private int program, fbo, texture, vbo;
    private int aPos, aTex, uMVPMatrix, uSTMatrix, sTexture;
    private boolean ready, failed;

    boolean capture(int oesTexture, float[] mvpMatrix, float[] stMatrix, float[] hostTexCoords, ByteBuffer dst) {
        if (failed || oesTexture <= 0 || stMatrix == null || stMatrix.length != 16
                || mvpMatrix == null || mvpMatrix.length != 16
                || hostTexCoords == null || hostTexCoords.length != 8 || dst == null
                || dst.capacity() < SIZE * SIZE * 4) return false;
        GLState saved = new GLState();
        try {
            saved.capture();
            if (!ensureCreated()) return false;

            // The host textureBuffer contains its crop/scale coordinates.  Copy
            // those into our VBO, so this FBO is in precisely the same visible
            // preview coordinate system as the host's screen quad.
            uploadQuad(hostTexCoords);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo);
            GLES20.glViewport(0, 0, SIZE, SIZE);
            GLES20.glDisable(GLES20.GL_BLEND);
            GLES20.glDisable(GLES20.GL_DEPTH_TEST);
            GLES20.glDisable(GLES20.GL_SCISSOR_TEST);
            GLES20.glUseProgram(program);
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo);
            GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 16, 0);
            GLES20.glVertexAttribPointer(aTex, 2, GLES20.GL_FLOAT, false, 16, 8);
            GLES20.glEnableVertexAttribArray(aPos);
            GLES20.glEnableVertexAttribArray(aTex);
            GLES20.glUniformMatrix4fv(uMVPMatrix, 1, false, mvpMatrix, 0);
            GLES20.glUniformMatrix4fv(uSTMatrix, 1, false, stMatrix, 0);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, oesTexture);
            GLES20.glUniform1i(sTexture, 0);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
            dst.position(0);
            GLES20.glReadPixels(0, 0, SIZE, SIZE, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, dst);
            return GLES20.glGetError() == GLES20.GL_NO_ERROR;
        } catch (Throwable ignored) {
            return false;
        } finally {
            // restore is intentionally best-effort and includes all state this
            // tap mutates.  Host onDraw has disabled its arrays at this boundary;
            // VBO-backed attrib setup avoids retaining any client-buffer pointer.
            saved.restore();
        }
    }

    private boolean ensureCreated() {
        if (ready) return true;
        try {
            int vs = compile(GLES20.GL_VERTEX_SHADER, VS);
            int fs = compile(GLES20.GL_FRAGMENT_SHADER, FS);
            if (vs == 0 || fs == 0) return fail();
            program = GLES20.glCreateProgram();
            GLES20.glAttachShader(program, vs); GLES20.glAttachShader(program, fs);
            GLES20.glLinkProgram(program);
            GLES20.glDeleteShader(vs); GLES20.glDeleteShader(fs);
            int[] ok = new int[1]; GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, ok, 0);
            if (ok[0] == 0) return fail();
            aPos = GLES20.glGetAttribLocation(program, "aPos");
            aTex = GLES20.glGetAttribLocation(program, "aTex");
            uMVPMatrix = GLES20.glGetUniformLocation(program, "uMVPMatrix");
            uSTMatrix = GLES20.glGetUniformLocation(program, "uSTMatrix");
            sTexture = GLES20.glGetUniformLocation(program, "sTexture");
            if (aPos < 0 || aTex < 0 || uMVPMatrix < 0 || uSTMatrix < 0 || sTexture < 0) return fail();

            int[] id = new int[1];
            GLES20.glGenTextures(1, id, 0); texture = id[0];
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture);
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, SIZE, SIZE, 0,
                    GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glGenFramebuffers(1, id, 0); fbo = id[0];
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo);
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                    GLES20.GL_TEXTURE_2D, texture, 0);
            if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE) return fail();
            GLES20.glGenBuffers(1, id, 0); vbo = id[0];
            ready = vbo != 0 && texture != 0 && fbo != 0;
            return ready;
        } catch (Throwable ignored) { return fail(); }
    }

    private void uploadQuad(float[] t) {
        float[] q = {-1f,-1f,t[0],t[1], 1f,-1f,t[2],t[3], -1f,1f,t[4],t[5], 1f,1f,t[6],t[7]};
        FloatBuffer data = ByteBuffer.allocateDirect(q.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        data.put(q).position(0);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, q.length * 4, data, GLES20.GL_STREAM_DRAW);
    }

    private int compile(int type, String source) {
        int shader = GLES20.glCreateShader(type); if (shader == 0) return 0;
        GLES20.glShaderSource(shader, source); GLES20.glCompileShader(shader);
        int[] ok = new int[1]; GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, ok, 0);
        if (ok[0] == 0) { GLES20.glDeleteShader(shader); return 0; }
        return shader;
    }
    private boolean fail() { failed = true; release(); return false; }

    void release() {
        try { if (vbo != 0) GLES20.glDeleteBuffers(1, new int[]{vbo}, 0); } catch (Throwable ignored) { }
        try { if (fbo != 0) GLES20.glDeleteFramebuffers(1, new int[]{fbo}, 0); } catch (Throwable ignored) { }
        try { if (texture != 0) GLES20.glDeleteTextures(1, new int[]{texture}, 0); } catch (Throwable ignored) { }
        try { if (program != 0) GLES20.glDeleteProgram(program); } catch (Throwable ignored) { }
        vbo = fbo = texture = program = 0; ready = false;
    }

    /** GL ES 2 exposes no portable pointer getter; at this known post-host-draw
     * boundary arrays are disabled. We preserve all enable bits and never leave
     * an enabled attrib pointing into our VBO. */
    private static final class GLState {
        final int[] active = new int[1], tex2d = new int[1], texOes = new int[1], fbo = new int[1];
        // capture both the caller's active unit and unit 0 because capture binds
        // the OES source on unit 0 even when the host left another unit active.
        final int[] unit0Tex2d = new int[1], unit0TexOes = new int[1];
        final int[] viewport = new int[4], program = new int[1], array = new int[1], element = new int[1];
        boolean blend, depth, scissor;
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
            blend = GLES20.glIsEnabled(GLES20.GL_BLEND); depth = GLES20.glIsEnabled(GLES20.GL_DEPTH_TEST);
            scissor = GLES20.glIsEnabled(GLES20.GL_SCISSOR_TEST);
            int[] max = new int[1]; GLES20.glGetIntegerv(GLES20.GL_MAX_VERTEX_ATTRIBS, max, 0);
            attribEnabled = new int[Math.max(0, max[0])]; int[] value = new int[1];
            for (int i = 0; i < attribEnabled.length; i++) { GLES20.glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, value, 0); attribEnabled[i] = value[0]; }
        }
        void restore() {
            try {
                if (attribEnabled != null) for (int i = 0; i < attribEnabled.length; i++) {
                    if (attribEnabled[i] != 0) GLES20.glEnableVertexAttribArray(i); else GLES20.glDisableVertexAttribArray(i);
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
            } catch (Throwable ignored) { }
        }
    }
}
