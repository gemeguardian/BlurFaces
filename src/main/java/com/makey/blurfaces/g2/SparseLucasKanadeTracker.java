package com.makey.blurfaces.g2;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * Lightweight sparse Lucas-Kanade optical flow tracker for 192x192 preview frames.
 * Tracks 5-8 points within active face bounding boxes over a 7x7 window with 3 iterations
 * to compensate for camera motion and jerks in real time.
 */
public final class SparseLucasKanadeTracker {
    public static final int WINDOW_SIZE = 7;
    public static final int HALF_WINDOW = 3;
    public static final int ITERATIONS = 3;
    public static final int MIN_VALID_POINTS = 2;
    public static final float MIN_DETERMINANT = 10.0f;
    public static final float MIN_EIGENVALUE = 0.5f;
    public static final float MAX_DISPLACEMENT = 25.0f;

    // Per-tracker working buffers to eliminate heap allocations per frame
    private final float[] ixWin = new float[WINDOW_SIZE * WINDOW_SIZE];
    private final float[] iyWin = new float[WINDOW_SIZE * WINDOW_SIZE];
    private final float[] prevWin = new float[WINDOW_SIZE * WINDOW_SIZE];
    private final float[] pointDx = new float[16];
    private final float[] pointDy = new float[16];

    public static final class FlowResult {
        public final float flowX;
        public final float flowY;
        public final int validPoints;
        public final boolean valid;

        public FlowResult(float flowX, float flowY, int validPoints, boolean valid) {
            this.flowX = flowX;
            this.flowY = flowY;
            this.validPoints = validPoints;
            this.valid = valid;
        }

        public static final FlowResult INVALID = new FlowResult(0f, 0f, 0, false);
    }

    /**
     * Converts a 192x192 RGBA buffer to a grayscale float array.
     * When flipY is true, inverts vertical order so that row 0 corresponds to the top of the image.
     */
    public static void rgbaToGrayscale(ByteBuffer rgba, float[] gray, int width, int height, boolean flipY) {
        if (rgba == null || gray == null || width <= 0 || height <= 0 || gray.length < width * height) return;
        int limit = Math.min(rgba.limit(), width * height * 4);
        for (int y = 0; y < height; y++) {
            int srcY = flipY ? (height - 1 - y) : y;
            int srcRowOffset = srcY * width * 4;
            int dstRowOffset = y * width;
            for (int x = 0; x < width; x++) {
                int idx = srcRowOffset + (x << 2);
                if (idx + 2 >= limit) break;
                int r = rgba.get(idx) & 0xFF;
                int g = rgba.get(idx + 1) & 0xFF;
                int b = rgba.get(idx + 2) & 0xFF;
                gray[dstRowOffset + x] = (r * 77 + g * 150 + b * 29) >> 8;
            }
        }
    }

    /**
     * Converts an RGBA byte array to a grayscale float array.
     */
    public static void rgbaToGrayscale(byte[] rgba, float[] gray, int width, int height, boolean flipY) {
        if (rgba == null || gray == null || width <= 0 || height <= 0 || gray.length < width * height) return;
        int limit = Math.min(rgba.length, width * height * 4);
        for (int y = 0; y < height; y++) {
            int srcY = flipY ? (height - 1 - y) : y;
            int srcRowOffset = srcY * width * 4;
            int dstRowOffset = y * width;
            for (int x = 0; x < width; x++) {
                int idx = srcRowOffset + (x << 2);
                if (idx + 2 >= limit) break;
                int r = rgba[idx] & 0xFF;
                int g = rgba[idx + 1] & 0xFF;
                int b = rgba[idx + 2] & 0xFF;
                gray[dstRowOffset + x] = (r * 77 + g * 150 + b * 29) >> 8;
            }
        }
    }

    /**
     * Selects 7 points around the face center (cx, cy) within +/-0.30*rx and +/-0.30*ry.
     * Coordinates are in pixel space [0..width, 0..height].
     * Returns the number of points written to outX and outY.
     */
    public static int selectTrackPoints(float cxNorm, float cyNorm, float rxNorm, float ryNorm,
                                        int width, int height, float[] outX, float[] outY) {
        if (outX == null || outY == null || outX.length < 8 || outY.length < 8 || width <= 0 || height <= 0) return 0;
        if (Float.isNaN(cxNorm) || Float.isNaN(cyNorm) || Float.isNaN(rxNorm) || Float.isNaN(ryNorm)) return 0;
        float px = cxNorm * width;
        float py = cyNorm * height;
        float prx = Math.max(rxNorm * width, 6.0f);
        float pry = Math.max(ryNorm * height, 6.0f);

        // 7 distinct points inside the face core
        // 0: Center
        outX[0] = px;
        outY[0] = py;
        // 1: Center - 0.30 * rx
        outX[1] = px - 0.30f * prx;
        outY[1] = py;
        // 2: Center + 0.30 * rx
        outX[2] = px + 0.30f * prx;
        outY[2] = py;
        // 3: Center - 0.30 * ry
        outX[3] = px;
        outY[3] = py - 0.30f * pry;
        // 4: Center + 0.30 * ry
        outX[4] = px;
        outY[4] = py + 0.30f * pry;
        // 5: Center - 0.22 * rx, - 0.22 * ry (upper-left)
        outX[5] = px - 0.22f * prx;
        outY[5] = py - 0.22f * pry;
        // 6: Center + 0.22 * rx, - 0.22 * ry (upper-right)
        outX[6] = px + 0.22f * prx;
        outY[6] = py - 0.22f * pry;
        // 7: Center, - 0.15 * ry (nose bridge)
        outX[7] = px;
        outY[7] = py - 0.15f * pry;

        return 8;
    }

    /**
     * Tracks the specified points from prevGray to currGray using sparse Lucas-Kanade.
     * Returns the median displacement (flowX, flowY) in pixels.
     */
    public FlowResult track(float[] prevGray, float[] currGray, int width, int height,
                            float[] ptsX, float[] ptsY, int numPoints) {
        if (prevGray == null || currGray == null || ptsX == null || ptsY == null || numPoints <= 0
                || width < 9 || height < 9
                || prevGray.length < width * height || currGray.length < width * height) {
            return FlowResult.INVALID;
        }

        int validCount = 0;
        int maxPoints = Math.min(numPoints, Math.min(ptsX.length, ptsY.length));

        for (int p = 0; p < maxPoints; p++) {
            float x0 = ptsX[p];
            float y0 = ptsY[p];
            if (Float.isNaN(x0) || Float.isNaN(y0) || Float.isInfinite(x0) || Float.isInfinite(y0)) {
                continue;
            }
            int rx0 = Math.round(x0);
            int ry0 = Math.round(y0);

            // Bounds check for 7x7 window (+/- 3 pixels) plus 1 pixel for central difference
            if (rx0 < 4 || rx0 >= width - 4 || ry0 < 4 || ry0 >= height - 4) {
                continue;
            }

            // Step 1: Spatial gradients & 2x2 structure matrix G over 7x7 window in prevGray
            float gxx = 0f, gyy = 0f, gxy = 0f;
            int winIdx = 0;
            for (int wy = -HALF_WINDOW; wy <= HALF_WINDOW; wy++) {
                int py = ry0 + wy;
                int row = py * width;
                int rowPrev = (py - 1) * width;
                int rowNext = (py + 1) * width;
                for (int wx = -HALF_WINDOW; wx <= HALF_WINDOW; wx++) {
                    int px = rx0 + wx;
                    float ix = 0.5f * (prevGray[row + px + 1] - prevGray[row + px - 1]);
                    float iy = 0.5f * (prevGray[rowNext + px] - prevGray[rowPrev + px]);
                    ixWin[winIdx] = ix;
                    iyWin[winIdx] = iy;
                    prevWin[winIdx] = prevGray[row + px];
                    winIdx++;
                    gxx += ix * ix;
                    gyy += iy * iy;
                    gxy += ix * iy;
                }
            }

            // Step 2: Confidence check: determinant & minimum eigenvalue of G
            float det = gxx * gyy - gxy * gxy;
            float trace = gxx + gyy;
            float diff = gxx - gyy;
            float lambdaMin = 0.5f * (trace - (float) Math.sqrt(Math.max(0f, diff * diff + 4f * gxy * gxy)));

            if (det < MIN_DETERMINANT || lambdaMin < MIN_EIGENVALUE) {
                continue;
            }

            float invDet = 1.0f / det;
            float invGxx = gyy * invDet;
            float invGyy = gxx * invDet;
            float invGxy = -gxy * invDet;

            // Step 3: Lucas-Kanade 3 iterations
            float ux = 0f, uy = 0f;
            boolean pointConverged = true;

            for (int iter = 0; iter < ITERATIONS; iter++) {
                float bx = 0f, by = 0f;
                winIdx = 0;

                for (int wy = -HALF_WINDOW; wy <= HALF_WINDOW; wy++) {
                    float curY = ry0 + wy + uy;
                    if (Float.isNaN(curY) || curY < 0f || curY > height - 1) {
                        pointConverged = false;
                        break;
                    }
                    for (int wx = -HALF_WINDOW; wx <= HALF_WINDOW; wx++) {
                        float curX = rx0 + wx + ux;
                        if (Float.isNaN(curX) || curX < 0f || curX > width - 1) {
                            pointConverged = false;
                            break;
                        }
                        float curVal = sampleBilinear(currGray, curX, curY, width, height);
                        float dI = prevWin[winIdx] - curVal;
                        bx += ixWin[winIdx] * dI;
                        by += iyWin[winIdx] * dI;
                        winIdx++;
                    }
                    if (!pointConverged) break;
                }
                if (!pointConverged) break;

                float dux = invGxx * bx + invGxy * by;
                float duy = invGxy * bx + invGyy * by;
                if (Float.isNaN(dux) || Float.isNaN(duy) || Float.isInfinite(dux) || Float.isInfinite(duy)) {
                    pointConverged = false;
                    break;
                }
                ux += dux;
                uy += duy;

                if (dux * dux + duy * duy < 1e-4f) {
                    break;
                }
            }

            if (!pointConverged) continue;

            float dispSq = ux * ux + uy * uy;
            if (dispSq > MAX_DISPLACEMENT * MAX_DISPLACEMENT || Float.isNaN(dispSq)) {
                continue;
            }

            pointDx[validCount] = ux;
            pointDy[validCount] = uy;
            validCount++;
        }

        if (validCount < MIN_VALID_POINTS) {
            return FlowResult.INVALID;
        }

        Arrays.sort(pointDx, 0, validCount);
        Arrays.sort(pointDy, 0, validCount);
        float medianX = computeMedian(pointDx, validCount);
        float medianY = computeMedian(pointDy, validCount);

        return new FlowResult(medianX, medianY, validCount, true);
    }

    public static float sampleBilinear(float[] img, float x, float y, int width, int height) {
        if (img == null || width <= 0 || height <= 0 || img.length < width * height) return 0f;
        if (Float.isNaN(x) || x <= 0f) {
            x = 0f;
        } else if (x >= width - 1f) {
            x = width - 1f;
        }
        if (Float.isNaN(y) || y <= 0f) {
            y = 0f;
        } else if (y >= height - 1f) {
            y = height - 1f;
        }

        int x0 = (int) x;
        int y0 = (int) y;
        int x1 = Math.min(x0 + 1, width - 1);
        int y1 = Math.min(y0 + 1, height - 1);
        float fx = x - x0;
        float fy = y - y0;

        int row0 = y0 * width;
        int row1 = y1 * width;
        float v00 = img[row0 + x0];
        float v10 = img[row0 + x1];
        float v01 = img[row1 + x0];
        float v11 = img[row1 + x1];

        return (1f - fx) * (1f - fy) * v00 + fx * (1f - fy) * v10 + (1f - fx) * fy * v01 + fx * fy * v11;
    }

    private static float computeMedian(float[] arr, int length) {
        if (length == 0) return 0f;
        int mid = length / 2;
        if ((length & 1) == 1) {
            return arr[mid];
        } else {
            return 0.5f * (arr[mid - 1] + arr[mid]);
        }
    }
}
