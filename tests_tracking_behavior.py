#!/usr/bin/env python3
"""Behavioral model for order-independent, per-face hold and prediction rules."""
import itertools
import math


def match(tracks, detections):
    unmatched_t = set(range(len(tracks)))
    unmatched_d = set(range(len(detections)))
    pairs = []
    while unmatched_t and unmatched_d:
        distance, ti, di = min(
            ((tracks[t][0] - detections[d][0]) ** 2 +
             (tracks[t][1] - detections[d][1]) ** 2, t, d)
            for t in unmatched_t for d in unmatched_d
        )
        radius = max(tracks[ti][2], 0.045)
        if distance > (radius * 1.6) ** 2:
            break
        pairs.append((ti, detections[di]))
        unmatched_t.remove(ti)
        unmatched_d.remove(di)
    return sorted(pairs)


tracks = [(0.20, 0.30, 0.08), (0.75, 0.32, 0.09), (0.48, 0.70, 0.07)]
detections = [(0.205, 0.302), (0.742, 0.318), (0.485, 0.695)]
expected = match(tracks, detections)
for ordering in itertools.permutations(detections):
    actual = match(tracks, ordering)
    assert [index for index, _ in actual] == [0, 1, 2]
    assert all(math.dist(actual[i][1], expected[i][1]) < 1e-9 for i in range(3))

# Slow inference adapts the hold from result cadence. Publication time prevents
# inference latency itself from consuming the mask lifetime.
hold_ns = max(350_000_000, min(1_200_000_000, 280_000_000 * 3 + 100_000_000))
assert hold_ns == 940_000_000
last_seen = [1_000_000_000, 1_200_000_000]
assert [1_300_000_000 - seen <= hold_ns for seen in last_seen] == [True, True]
assert [2_140_000_001 - seen <= hold_ns for seen in last_seen] == [False, False]

# Prediction is capped even when hold lasts longer.
prediction_ns = min(120_000_000, 300_000_000)
assert prediction_ns == 120_000_000
print("PASS: matching ignores detection order; face hold/expiry and prediction are independent")
