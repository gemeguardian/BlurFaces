package com.makey.blurfaces.g2;

import android.graphics.Bitmap;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/** Opt-in sampling only. Disk IO and JPEG encoding never execute on camera/inference threads. */
final class DebugCapture {
    static final int CANDIDATES = 64, HEADER = 10, TRACK_OFFSET = HEADER + CANDIDATES * 5;
    static final int SNAPSHOT_FLOATS = TRACK_OFFSET + 4 * 8;
    final DebugCaptureStore store;
    private ExecutorService writer;
    private String bundleId = "unknown";

    DebugCapture(Path directory) { store = new DebugCaptureStore(directory); }

    synchronized void configure(boolean enabled, String bundle) throws IOException {
        if (bundle != null && bundle.matches("[0-9a-f]{64}")) bundleId = bundle;
        store.configure(enabled, System.nanoTime(), System.currentTimeMillis());
        if (enabled && (writer == null || writer.isShutdown())) {
            writer = Executors.newSingleThreadExecutor(r -> {
                Thread thread = new Thread(r, "BlurFacesDebugWriter");
                thread.setDaemon(true);
                return thread;
            });
        }
    }

    synchronized void stop() {
        try { store.configure(false, System.nanoTime(), System.currentTimeMillis()); }
        catch (IOException error) { store.failure(); }
        // Do not interrupt an in-flight atomic write. The generation check cancels it.
        if (writer != null) writer.shutdown();
    }

    synchronized void submit(long token, ByteBuffer rgba, int width, int height,
                             float[] snapshot, int count, float confidence,
                             long captureNanos, long resultNanos, long inferenceNanos, String source,
                             float[] javaTracks, boolean blurEnabled, int maskMode) {
        try {
            if (!store.accepts(token, System.nanoTime()) || writer == null || writer.isShutdown()) {
                store.release();
                return;
            }
            byte[] pixels = new byte[width * height * 4];
            ByteBuffer input = rgba.duplicate();
            input.position(0); input.get(pixels);
            float[] details = snapshot.clone();
            long wallMillis = System.currentTimeMillis();
            String bundle = bundleId;
            writer.execute(() -> {
                try {
                    if (!store.accepts(token, System.nanoTime())) return;
                    byte[] zip = encode(pixels, width, height, details, count, confidence,
                            captureNanos, resultNanos, inferenceNanos, source, wallMillis, bundle,
                            javaTracks, blurEnabled, maskMode);
                    store.write(token, System.nanoTime(), zip);
                } catch (Exception error) {
                    store.failure(token);
                } finally {
                    store.release();
                }
            });
        } catch (Exception error) {
            store.failure(token);
            store.release();
        }
    }

    static byte[] encode(byte[] rgba, int width, int height, float[] snapshot, int count,
                         float confidence, long captureNanos, long resultNanos, long inferenceNanos,
                         String source, long wallMillis, String bundle,
                         float[] javaTracks, boolean blurEnabled, int maskMode) throws Exception {
        if (width <= 0 || height <= 0 || (long) width * height * 4 != rgba.length)
            throw new IllegalArgumentException("Invalid RGBA frame");
        JSONObject json = new JSONObject();
        json.put("schema", 1);
        json.put("bundle_id", bundle);
        json.put("wall_time_ms", wallMillis);
        json.put("source", source);
        json.put("capture_nanos", captureNanos);
        json.put("result_nanos", resultNanos);
        json.put("capture_to_result_ms", (resultNanos - captureNanos) / 1_000_000.0);
        json.put("inference_ms", inferenceNanos / 1_000_000.0);
        json.put("configured_confidence", confidence);
        json.put("width", width); json.put("height", height);
        json.put("native_result", count);
        json.put("reason", count < 0 ? "native_error_or_overflow" : count == 0 ? "no_active_native_tracks" : "native_tracks");
        json.put("scope", "post_filter_post_nms_candidates_native_and_java_tracks_not_final_render_masks");
        json.put("full_frame_saved", true);
        json.put("frame_file", "frame.rgba");
        json.put("frame_format", "RGBA8888_top_left_tightly_packed_before_clahe");
        json.put("blur_enabled", blurEnabled);
        json.put("mask_mode", maskMode);
        json.put("java_track_fields", new JSONArray(new String[]{
                "cx", "cy", "axis_x_x", "axis_x_y", "axis_y_x", "axis_y_y",
                "draw_cx", "draw_cy", "capture_age_ms", "publication_age_ms"}));
        JSONArray javaState = new JSONArray();
        for (int p = 0; p + 10 <= javaTracks.length; p += 10) {
            JSONArray row = new JSONArray();
            for (int j = 0; j < 10; j++) row.put(javaTracks[p + j]);
            javaState.put(row);
        }
        json.put("java_tracks", javaState);
        JSONArray boxes = new JSONArray();
        json.put("boxes", boxes);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(bytes)) {
            zip.putNextEntry(new ZipEntry("frame.rgba"));
            zip.write(rgba);
            zip.closeEntry();
            boolean valid = validSnapshot(snapshot);
            json.put("snapshot_valid", valid);
            if (valid) {
                json.put("detector_status", (int) snapshot[1]);
                json.put("high_threshold", snapshot[2]);
                json.put("low_threshold", snapshot[3]);
                json.put("instant_threshold", snapshot[4]);
                json.put("threshold_luminance_previous", snapshot[5]);
                json.put("frame_luminance", snapshot[6]);
                json.put("candidate_total", (int) snapshot[7]);
                json.put("candidate_saved", (int) snapshot[8]);
                json.put("candidates_truncated", snapshot[7] > snapshot[8]);
                for (int i = 0; i < (int) snapshot[8]; i++) {
                    int p = HEADER + i * 5;
                    JSONObject box = new JSONObject();
                    box.put("kind", "candidate"); box.put("score", snapshot[p+4]);
                    addCrop(zip, boxes, box, "candidate_" + i + ".jpg", rgba, width, height,
                            snapshot[p], snapshot[p+1], snapshot[p+2], snapshot[p+3]);
                }
                for (int i = 0; i < (int) snapshot[9]; i++) {
                    int p = TRACK_OFFSET + i * 8;
                    JSONObject box = new JSONObject();
                    box.put("kind", snapshot[p+1] == 2 ? "predicted_track" : "observed_track");
                    box.put("track_id", (int) snapshot[p]); box.put("state", (int) snapshot[p+1]);
                    box.put("frames_lost", (int) snapshot[p+2]); box.put("score", snapshot[p+3]);
                    addCrop(zip, boxes, box, "track_" + i + ".jpg", rgba, width, height,
                            snapshot[p+4], snapshot[p+5], snapshot[p+6], snapshot[p+7]);
                }
            }
            zip.putNextEntry(new ZipEntry("metadata.json"));
            zip.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
        if (bytes.size() > DebugCaptureStore.MAX_RECORD_BYTES) throw new IOException("Sample too large");
        return bytes.toByteArray();
    }

    static boolean validSnapshot(float[] s) {
        if (s == null || s.length != SNAPSHOT_FLOATS || s[0] != 1) return false;
        for (float value : s) if (!Float.isFinite(value)) return false;
        return s[8] >= 0 && s[8] <= CANDIDATES && s[8] == (int) s[8]
                && s[7] >= s[8] && s[9] >= 0 && s[9] <= 4 && s[9] == (int) s[9];
    }

    private static void addCrop(ZipOutputStream zip, JSONArray boxes, JSONObject box, String name,
                                byte[] rgba, int width, int height,
                                float x1, float y1, float x2, float y2) throws Exception {
        box.put("bbox_xyxy_normalized", new JSONArray(new double[]{x1, y1, x2, y2}));
        int[] crop = cropBounds(x1, y1, x2, y2, width, height);
        if (crop != null) {
            box.put("image", name);
            box.put("crop_xyxy_pixels", new JSONArray(crop));
            int w = crop[2]-crop[0], h = crop[3]-crop[1];
            int[] argb = cropArgb(rgba, width, height, crop);
            Bitmap bitmap = Bitmap.createBitmap(argb, w, h, Bitmap.Config.ARGB_8888);
            try {
                zip.putNextEntry(new ZipEntry(name));
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 90, zip)) throw new IOException("JPEG encode failed");
                zip.closeEntry();
            } finally { bitmap.recycle(); }
        }
        boxes.put(box);
    }

    // 25% context on each side, clipped to the actual detector image. Top-left origin.
    static int[] cropBounds(float x1, float y1, float x2, float y2, int width, int height) {
        if (width <= 0 || height <= 0 || !Float.isFinite(x1) || !Float.isFinite(y1)
                || !Float.isFinite(x2) || !Float.isFinite(y2) || x2 <= x1 || y2 <= y1) return null;
        float dx = (x2-x1)*.25f, dy = (y2-y1)*.25f;
        int left = Math.max(0, Math.min(width, (int) Math.floor((x1-dx)*width)));
        int top = Math.max(0, Math.min(height, (int) Math.floor((y1-dy)*height)));
        int right = Math.max(0, Math.min(width, (int) Math.ceil((x2+dx)*width)));
        int bottom = Math.max(0, Math.min(height, (int) Math.ceil((y2+dy)*height)));
        return right <= left || bottom <= top ? null : new int[]{left, top, right, bottom};
    }

    static int[] cropArgb(byte[] rgba, int width, int height, int[] crop) {
        if (rgba.length != width*height*4 || crop[0] < 0 || crop[1] < 0
                || crop[2] > width || crop[3] > height || crop[2] <= crop[0] || crop[3] <= crop[1])
            throw new IllegalArgumentException("Invalid crop input");
        int[] output = new int[(crop[2]-crop[0])*(crop[3]-crop[1])];
        int n = 0;
        for (int y = crop[1]; y < crop[3]; y++) for (int x = crop[0]; x < crop[2]; x++) {
            int p = (y*width+x)*4;
            output[n++] = 0xff000000 | (rgba[p]&255)<<16 | (rgba[p+1]&255)<<8 | (rgba[p+2]&255);
        }
        return output;
    }
}
