package com.makey.blurfaces;

/**
 * One Euro Filter — adaptive low-pass filter for noisy signals.
 * Reduces jitter at low speeds, maintains responsiveness at high speeds.
 */
public final class OneEuroFilter {
    private final float minCutoff;
    private final float beta;
    private final float dCutoff;
    
    private float xPrev;
    private float dxPrev;
    private long tPrev;
    private boolean initialized = false;

    public OneEuroFilter(float minCutoff, float beta, float dCutoff) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
    }

    public synchronized float filter(float x, long timestamp) {
        if (!initialized) {
            xPrev = x;
            dxPrev = 0;
            tPrev = timestamp;
            initialized = true;
            return x;
        }

        float dt = (timestamp - tPrev) / 1000.0f;
        if (dt <= 0) dt = 0.001f;

        // Derivative estimation
        float dx = (x - xPrev) / dt;
        
        // Smooth derivative
        float edx = exponentialSmoothing(dx, dxPrev, alpha(dCutoff, dt));
        
        // Adaptive cutoff based on speed
        float cutoff = minCutoff + beta * Math.abs(edx);
        
        // Smooth signal
        float ex = exponentialSmoothing(x, xPrev, alpha(cutoff, dt));

        xPrev = ex;
        dxPrev = edx;
        tPrev = timestamp;

        return ex;
    }

    private float alpha(float cutoff, float dt) {
        float tau = 1.0f / (2.0f * (float) Math.PI * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }

    private float exponentialSmoothing(float x, float xPrev, float alpha) {
        return alpha * x + (1 - alpha) * xPrev;
    }

    public synchronized void reset() {
        initialized = false;
    }
}