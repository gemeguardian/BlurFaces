import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
MAIN = (ROOT / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text()
TAP = (ROOT / "src/main/java/com/makey/blurfaces/g2/CleanFrameTap.java").read_text()


class HookContract(unittest.TestCase):
    def test_capture_is_update_tex_image_after_hook(self):
        self.assertIn('SurfaceTexture.class.getDeclaredMethod("updateTexImage")', MAIN)
        self.assertIn("afterHookedMethod", MAIN)
        self.assertIn("afterSurfaceUpdate((SurfaceTexture) param.thisObject)", MAIN)

    def test_surface_filter_uses_exact_identity(self):
        self.assertIn("surfaces[i] == updated", MAIN)
        self.assertIn("surfaces[slot] != surface", MAIN)
        self.assertIn('field(outer.getClass(), "surfaceIndex").getInt(outer)', MAIN)
        self.assertIn("if (slot != activeSlot)", MAIN)
        self.assertNotIn("hookMethod(draw, new XC_MethodHook() {\n            @Override protected void afterHookedMethod(MethodHookParam param) { afterSurfaceUpdate", MAIN)

    def test_renderer_mutations_are_per_draw_and_restored(self):
        self.assertIn("beforePreviewDraw(param.thisObject)", MAIN)
        self.assertIn("afterPreviewDraw(param.thisObject)", MAIN)
        self.assertIn("beforeEncoderDraw(p.thisObject, p.args[1])", MAIN)
        self.assertIn("afterEncoderDraw(p.thisObject)", MAIN)
        self.assertGreaterEqual(MAIN.count("finally { state.swapActive = false; restore"), 2)

    def test_late_results_are_source_bound(self):
        self.assertIn("if (isSourceActive(frame.sourceKey))", MAIN)
        self.assertIn("TRACK_HOLD_NS", MAIN)
        self.assertIn("LATEST_FRAME.getAndSet(next)", MAIN)
        self.assertIn("DRAIN_SCHEDULED.compareAndSet(false, true)", MAIN)

    def test_encoder_prefers_exact_texture_mapping(self):
        self.assertIn("SOURCE_BY_TEXTURE.put(textures[slot], source)", MAIN)
        self.assertIn("SOURCE_BY_TEXTURE.get(textureId)", MAIN)
        self.assertIn("Encoder face protection active", MAIN)

    def test_optimized_encoder_uniforms_are_optional(self):
        ready = MAIN[MAIN.index("state.ready = state.position"):MAIN.index("if (!state.ready)")]
        self.assertNotIn("state.preview >= 0", ready)
        self.assertNotIn("state.resolution >= 0", ready)
        self.assertNotIn("state.texel >= 0", ready)
        self.assertIn("state.alpha >= 0", ready)

    def test_pixel_grid_uniform_cached_and_scale_clamped(self):
        self.assertIn('state.pixelGrid = GLES20.glGetUniformLocation(state.program, "uPixelGrid");', MAIN)
        self.assertIn("state.axisY, state.viewport, state.blurSampler, state.pixelGrid,", MAIN)
        self.assertIn("clampBlurRadiusScale", TAP)

    def test_camera_flip_is_fail_closed_until_fresh_result(self):
        self.assertIn("SOURCE_ACTIVE_SINCE.put(source, now)", MAIN)
        self.assertIn("tracks.lastResultNanos >= activeSince", MAIN)
        self.assertIn("resolveFaceCount(source, now)", MAIN)
        self.assertIn("if(uFaceCount<0){gl_FragColor=texture2D(sBlurTexture,buv());return;}", MAIN)
        self.assertIn("full-frame privacy blur active until fresh detection", MAIN)
        self.assertIn("sawFaceRecently(source, now)", MAIN)
        self.assertIn("switchCameraX", MAIN)
        self.assertIn("isFrontFacing", MAIN)

    def test_privacy_blur_uses_downsampled_separable_gaussian(self):
        self.assertNotIn("floor(p/vec2", MAIN)
        self.assertIn("uniform sampler2D sBlurTexture", MAIN)
        self.assertIn("uniform float uPixelGrid;", MAIN)
        self.assertIn("floor(uv*uPixelGrid)", MAIN)
        self.assertIn("state.tap.renderBlur(textureId, mvp, st, tex, null, blurRadiusScale)", MAIN)
        self.assertIn("state.tap.renderBlur(textures[slot], mvp, st, tex", MAIN)
        self.assertIn("drawBlur(textures[0], fbos[1], step, 0f)", TAP)
        self.assertIn("drawBlur(textures[1], fbos[0], 0f, step)", TAP)
        self.assertGreaterEqual(TAP.count("drawBlur("), 7)
        self.assertIn("float step = (4.5f * blurRadiusScale) / SIZE", TAP)
        self.assertIn("1.38461538", TAP)
        self.assertIn("3.23076923", TAP)
        self.assertGreaterEqual(MAIN.count("if(m<=0.0)"), 2)
        self.assertIn("if (encoderTransitionActive(renderer)) state.faceCount = -1", MAIN)

    def test_face_contour_stays_inside_blurred_core(self):
        self.assertGreaterEqual(MAIN.count("smoothstep(.84,1.04"), 2)

    def test_tracks_are_source_local_order_independent_and_bounded(self):
        self.assertIn("Map<String, SourceTracks> SOURCE_TRACKS", MAIN)
        self.assertIn("Repeated global-nearest pairing", MAIN)
        self.assertIn("TRACK_MIN_CUTOFF = 1.2f", MAIN)
        self.assertIn("MAX_PREDICTION_NS = 120_000_000L", MAIN)
        self.assertIn("now - track.lastSeenPublishedNanos > holdLimit()", MAIN)
        self.assertIn("MAX_TRACK_HOLD_NS = 1_200_000_000L", MAIN)
        self.assertNotIn("HELD_RESULTS", MAIN)

    def test_two_level_confidence_hysteresis(self):
        self.assertIn("TRACK_GATED_MIN_CONFIDENCE = 0.20f;", MAIN)
        self.assertIn("TRACK_NEW_MIN_CONFIDENCE = 0.20f;", MAIN)
        self.assertIn("score < TRACK_GATED_MIN_CONFIDENCE", MAIN)
        self.assertIn("score < newTrackThreshold", MAIN)
        self.assertIn("Math.max(TRACK_NEW_MIN_CONFIDENCE, configuredConfidence)", MAIN)

    def test_anti_overshoot_prediction_bounded(self):
        self.assertIn("float posHorizon = (speed < TRACK_SPEED_DEADBAND) ? 0f : Math.min(0.025f, stale);", MAIN)
        self.assertIn("float maxShiftX = Math.max(0.015f, radiusX * 0.30f);", MAIN)
        self.assertIn("estX = clamp(estX, values[0] - maxShiftX, values[0] + maxShiftX);", MAIN)
        self.assertNotIn("float estX = values[0] + velocity[0] * horizon + .5f * accel[0]", MAIN)

    def test_fail_closed_contracts_on_error_branches(self):
        self.assertIn("enforcePreviewFailClosed(thread, state)", MAIN)
        self.assertIn("enforceEncoderFailClosed(renderer, state, snapshot)", MAIN)
        self.assertIn("FALLBACK_VS", MAIN)
        self.assertIn("FALLBACK_FS", MAIN)
        self.assertIn("fallbackTexture()", MAIN)
        self.assertTrue('setProtectionState("DEGRADED")' in MAIN or 'protectionState = "DEGRADED"' in MAIN)
        self.assertIn("maskLeakFrames++", MAIN)
        # Ensure GL state and fallback program clean swap contracts
        self.assertIn("GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, prevTex, 0);", MAIN)
        self.assertIn("GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, prevTex[0]);", MAIN)
        self.assertIn("state.blurTexture = 0;", MAIN)
        self.assertIn('field(type, "textureMatrixHandle").setInt(thread, -1);', MAIN)
        self.assertIn('field(type, "textureMatrixHandle").setInt(renderer, -1);', MAIN)
        # Verify CleanFrameTap isolation and proper error returns
        self.assertIn("return 0;", TAP)
        self.assertNotIn("return textures[0] != 0 ? textures[0] : 0;", TAP)
        self.assertIn("GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);", TAP)
        self.assertIn("GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER);", TAP)


if __name__ == "__main__":
    unittest.main()
