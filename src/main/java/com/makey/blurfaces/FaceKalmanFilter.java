package com.makey.blurfaces;

/**
 * 2D Kalman filter for face tracking.
 * State: [x, y, vx, vy] — position and velocity in UV space.
 * Measurement: [x, y] — detected face center.
 */
public final class FaceKalmanFilter {
    // State transition: x += vx*dt, y += vy*dt
    // Process noise (how much we trust prediction vs measurement)
    private static final float PROCESS_NOISE_POS = 0.001f;
    private static final float PROCESS_NOISE_VEL = 0.01f;
    private static final float MEASUREMENT_NOISE = 0.005f;

    // State vector
    private float x, y, vx, vy;
    // Error covariance matrix (4x4, symmetric)
    private float[] P = new float[16];
    private boolean initialized = false;
    private long lastUpdateTime;

    public synchronized void update(float measX, float measY) {
        long now = android.os.SystemClock.elapsedRealtime();
        if (!initialized) {
            x = measX;
            y = measY;
            vx = 0;
            vy = 0;
            // Initial uncertainty
            for (int i = 0; i < 16; i++) P[i] = 0;
            P[0] = P[5] = 1.0f;   // position variance
            P[10] = P[15] = 0.1f; // velocity variance
            initialized = true;
            lastUpdateTime = now;
            return;
        }

        float dt = (now - lastUpdateTime) / 1000.0f;
        if (dt <= 0 || dt > 1.0f) dt = 0.033f; // fallback to 30fps

        // Predict
        float predX = x + vx * dt;
        float predY = y + vy * dt;

        // Predict covariance: P = F*P*F' + Q
        // F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
        float[] newP = new float[16];
        // Simplified: just add process noise to diagonals
        System.arraycopy(P, 0, newP, 0, 16);
        newP[0] += PROCESS_NOISE_POS + dt * dt * P[10];
        newP[5] += PROCESS_NOISE_POS + dt * dt * P[15];
        newP[10] += PROCESS_NOISE_VEL;
        newP[15] += PROCESS_NOISE_VEL;

        // Kalman gain (simplified, diagonal approximation)
        float kPos = newP[0] / (newP[0] + MEASUREMENT_NOISE);
        float kVel = newP[10] / (newP[10] + MEASUREMENT_NOISE * 0.1f);

        // Update
        float innovX = measX - predX;
        float innovY = measY - predY;

        x = predX + kPos * innovX;
        y = predY + kPos * innovY;
        vx = vx + kVel * innovX / Math.max(dt, 0.001f);
        vy = vy + kVel * innovY / Math.max(dt, 0.001f);

        // Clamp velocity to reasonable range (UV units per second)
        float maxVel = 2.0f; // full screen in 0.5s
        vx = Math.max(-maxVel, Math.min(maxVel, vx));
        vy = Math.max(-maxVel, Math.min(maxVel, vy));

        // Update covariance
        newP[0] *= (1 - kPos);
        newP[5] *= (1 - kPos);
        newP[10] *= (1 - kVel);
        newP[15] *= (1 - kVel);
        P = newP;

        lastUpdateTime = now;
    }

    public synchronized float[] predict(float aheadSeconds) {
        if (!initialized) return new float[]{0, 0, 0, 0};
        float predX = x + vx * aheadSeconds;
        float predY = y + vy * aheadSeconds;
        // Clamp to valid UV range
        predX = Math.max(0, Math.min(1, predX));
        predY = Math.max(0, Math.min(1, predY));
        return new float[]{predX, predY, vx, vy};
    }

    public synchronized boolean isInitialized() {
        return initialized;
    }

    public synchronized void reset() {
        initialized = false;
    }
}