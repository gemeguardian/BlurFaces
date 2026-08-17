import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
MAIN = (ROOT / "src/main/java/com/makey/blurfaces/Main.java").read_text()
TAP = (ROOT / "src/main/java/com/makey/blurfaces/CleanFrameTap.java").read_text()


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
        self.assertIn("if (!isSourceActive(submitted.sourceKey))", MAIN)
        self.assertIn("STALE_HOLD_NS", MAIN)
        self.assertIn("LATEST_FRAME.getAndSet(next)", MAIN)
        self.assertIn("IN_FLIGHT.get() != null", MAIN)

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

    def test_camera_flip_is_fail_closed_until_fresh_result(self):
        self.assertIn("SOURCE_ACTIVE_SINCE.put(source, now)", MAIN)
        self.assertIn("current.captureNanos >= activeSince", MAIN)
        self.assertIn("hasFreshResult(source, now) ? 0 : -1", MAIN)
        self.assertIn("if(uFaceCount<0){gl_FragColor=texture2D(sBlurTexture,buv());return;}", MAIN)
        self.assertIn("full-frame privacy blur active until fresh detection", MAIN)

    def test_privacy_blur_uses_downsampled_separable_gaussian(self):
        self.assertNotIn("floor(p/vec2", MAIN)
        self.assertIn("uniform sampler2D sBlurTexture", MAIN)
        self.assertIn("state.tap.renderBlur(textureId, mvp, st, tex, null)", MAIN)
        self.assertIn("state.tap.renderBlur(textures[slot], mvp, st, tex", MAIN)
        self.assertIn("drawBlur(textures[0], fbos[1], step, 0f)", TAP)
        self.assertIn("drawBlur(textures[1], fbos[0], 0f, step)", TAP)
        self.assertGreaterEqual(TAP.count("drawBlur("), 7)
        self.assertIn("float step = 4.5f / SIZE", TAP)
        self.assertIn("1.38461538", TAP)
        self.assertIn("3.23076923", TAP)
        self.assertGreaterEqual(MAIN.count("if(m<=0.0)"), 2)
        self.assertIn("if (encoderTransitionActive(renderer)) state.faceCount = -1", MAIN)

    def test_face_contour_stays_inside_blurred_core(self):
        self.assertIn("* 0.68f, radiusV", MAIN)
        self.assertGreaterEqual(MAIN.count("smoothstep(.82,1.0"), 2)


if __name__ == "__main__":
    unittest.main()
