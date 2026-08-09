#!/usr/bin/env python3
"""Deterministic geometry checks for the 5-KPS face-mask basis.

This mirrors Main.anchorsToAffine and validates that its mask basis follows a
translation, uniform zoom and in-plane roll exactly. It is intentionally pure
Python so it runs on the build host without an Android EGL/device runtime.
"""
import math


def affine(points):
    (le_x, le_y), (re_x, re_y), (nose_x, nose_y), (lm_x, lm_y), (rm_x, rm_y) = points
    eye_x, eye_y = re_x - le_x, re_y - le_y
    eye_distance = math.hypot(eye_x, eye_y)
    mouth_x, mouth_y = (lm_x + rm_x) * 0.5, (lm_y + rm_y) * 0.5
    vertical = math.hypot(mouth_x - (le_x + re_x) * 0.5, mouth_y - (le_y + re_y) * 0.5)
    if eye_distance < 0.012 or vertical < 0.018:
        return None
    center_x = nose_x * 0.52 + (le_x + re_x) * 0.12 + mouth_x * 0.24
    center_y = nose_y * 0.52 + (le_y + re_y) * 0.12 + mouth_y * 0.24
    ux, uy = eye_x / eye_distance, eye_y / eye_distance
    rx, ry = eye_distance * 1.32, vertical * 1.62
    # Face-relative jaw offset must rotate with the eye axis.
    center_x += (-uy) * (vertical * 0.20)
    center_y += ux * (vertical * 0.20)
    return (center_x, center_y, ux * rx, uy * rx, -uy * ry, ux * ry)


def transform(points, scale, angle, tx, ty):
    c, s = math.cos(angle), math.sin(angle)
    return [(scale * (c * x - s * y) + tx, scale * (s * x + c * y) + ty) for x, y in points]


def transform_basis(b, scale, angle, tx, ty):
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, ax, ay, bx, by = b
    def r(x, y): return scale * (c * x - s * y), scale * (s * x + c * y)
    cx, cy = r(cx, cy); ax, ay = r(ax, ay); bx, by = r(bx, by)
    return (cx + tx, cy + ty, ax, ay, bx, by)


def close(a, b, tolerance=3e-4):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))

base = [(0.38, 0.42), (0.62, 0.42), (0.50, 0.53), (0.42, 0.65), (0.58, 0.65)]
b0 = affine(base)
assert b0 is not None
for scale, angle, tx, ty in [(1.0, 0.0, 0.10, -0.07), (1.45, 0.0, -0.08, 0.04), (0.78, math.radians(31), 0.13, 0.09), (1.25, math.radians(-48), -0.12, -0.06)]:
    actual = affine(transform(base, scale, angle, tx, ty))
    expected = transform_basis(b0, scale, angle, tx, ty)
    assert close(actual, expected), (actual, expected)
assert affine([(0.5, 0.5)] * 5) is None
print("landmark affine synthetic gate: PASS (translation, zoom, roll, degenerate rejection)")
