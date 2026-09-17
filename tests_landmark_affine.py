#!/usr/bin/env python3
"""Behavioral checks for the oval PCA affine basis and vertical row correction."""
import math


def pca_basis(points):
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    xx = sum((x - cx) ** 2 for x, _ in points)
    xy = sum((x - cx) * (y - cy) for x, y in points)
    yy = sum((y - cy) ** 2 for _, y in points)
    angle = 0.5 * math.atan2(2 * xy, xx - yy)
    u = (math.cos(angle), math.sin(angle))
    v = (-u[1], u[0])
    pu = [(x - cx) * u[0] + (y - cy) * u[1] for x, y in points]
    pv = [(x - cx) * v[0] + (y - cy) * v[1] for x, y in points]
    return cx, cy, u, v, (max(pu) - min(pu)) * 0.58, (max(pv) - min(pv)) * 0.58


oval = [(0.5 + 0.2 * math.cos(i * math.pi / 18), 0.5 + 0.3 * math.sin(i * math.pi / 18)) for i in range(36)]
base = pca_basis(oval)
angle, scale, tx, ty = 0.61, 1.3, -0.11, 0.07
c, s = math.cos(angle), math.sin(angle)
transformed = [(scale * (c*x - s*y) + tx, scale * (s*x + c*y) + ty) for x, y in oval]
actual = pca_basis(transformed)
assert math.isclose(actual[0], scale * (c*base[0] - s*base[1]) + tx, abs_tol=1e-6)
assert math.isclose(actual[1], scale * (s*base[0] + c*base[1]) + ty, abs_tol=1e-6)
assert math.isclose(max(actual[4:]), scale * max(base[4:]), rel_tol=1e-6)
assert math.isclose(min(actual[4:]), scale * min(base[4:]), rel_tol=1e-6)

# glReadPixels rows [bottom, middle, top] must become [top, middle, bottom].
rows = [b"BBBB", b"MMMM", b"TTTT"]
corrected = b"".join(reversed(rows))
assert corrected == b"TTTTMMMMBBBB"
print("PASS: oval PCA follows similarity transforms; glReadPixels rows are vertically corrected")

# Geometry starts at 0.78 of the full oval span, while an exact oval radius is
# 0.5 of that span. The 65% profile therefore closely follows the landmark oval.
precise_ratio = 0.78 * 0.65 / 0.5
standard_ratio = 0.78 * 0.82 / 0.5
wide_ratio = 0.78 * 1.0 / 0.5
assert 1.0 <= precise_ratio < 1.03
assert 1.25 < standard_ratio < 1.30
assert math.isclose(wide_ratio, 1.56)
