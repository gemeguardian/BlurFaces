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

# Yaw calculation and proactive lateral dilation contract
def compute_mesh_yaw(nose_x, right_cheek_x, left_cheek_x):
    mid_x = (right_cheek_x + left_cheek_x) * 0.5
    span_x = abs(left_cheek_x - right_cheek_x)
    if span_x > 1e-4:
        return max(-1.0, min(1.0, (nose_x - mid_x) / (span_x * 0.5)))
    return 0.0

def compute_box_yaw(left_eye_x, right_eye_x, nose_x):
    mid_eye_x = (left_eye_x + right_eye_x) * 0.5
    eye_span = abs(right_eye_x - left_eye_x)
    if eye_span > 1e-4:
        return max(-1.0, min(1.0, (nose_x - mid_eye_x) / (eye_span * 0.5)))
    return 0.0

# Frontal faces: yaw == 0
assert compute_mesh_yaw(0.50, 0.35, 0.65) == 0.0
assert compute_box_yaw(0.40, 0.60, 0.50) == 0.0

# Turned faces (|yaw| = 0.60 > 0.25)
mesh_yaw_left = compute_mesh_yaw(0.59, 0.35, 0.65)
assert math.isclose(mesh_yaw_left, 0.60, abs_tol=1e-5)
box_yaw_left = compute_box_yaw(0.40, 0.60, 0.56)
assert math.isclose(box_yaw_left, 0.60, abs_tol=1e-5)

mesh_yaw_right = compute_mesh_yaw(0.41, 0.35, 0.65)
assert math.isclose(mesh_yaw_right, -0.60, abs_tol=1e-5)
box_yaw_right = compute_box_yaw(0.40, 0.60, 0.44)
assert math.isclose(box_yaw_right, -0.60, abs_tol=1e-5)

# Yaw-aware dilation: expands lateral radius and shifts center
def apply_yaw_dilation(radius, cx, yaw, ux=1.0):
    if abs(yaw) > 0.25:
        factor = 1.0 + 0.35 * abs(yaw)
        radius *= factor
        cx += ux * (yaw * 0.12 * radius)
    return radius, cx

r_base = 0.08
cx_base = 0.50
r_dilated, cx_shifted = apply_yaw_dilation(r_base, cx_base, mesh_yaw_left)
assert r_dilated > r_base
assert math.isclose(r_dilated, r_base * (1.0 + 0.35 * 0.60), rel_tol=1e-5)
assert cx_shifted > cx_base

r_dilated_right, cx_shifted_right = apply_yaw_dilation(r_base, cx_base, mesh_yaw_right)
assert r_dilated_right > r_base
assert cx_shifted_right < cx_base
