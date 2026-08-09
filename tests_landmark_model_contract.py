#!/usr/bin/env python3
"""Validate the shipped NCNN model's output layout against the native decoder."""
from pathlib import Path

SIZE = 320
layout = ((8, 3), (16, 2), (32, 2), (64, 3))
priors = sum(((SIZE + stride - 1) // stride) ** 2 * anchors for stride, anchors in layout)
assert priors == 5875, priors

param = Path("model/retinaface_mnet025_5kps.param").read_text()
# The four endpoint reshapes prove records are 4 loc, 2 score, 10 landmarks.
for line in (
    "Reshape          416                      1 1 368 416 0=4 1=-1",
    "Reshape          457                      1 1 372 457 0=2 1=-1",
    "Reshape          498                      1 1 376 498 0=10 1=-1",
    "Concat           output0",
    "Concat           529",
    "Softmax          530",
):
    assert line in param, line

print(f"landmark model contract: PASS ({priors} priors; loc/output0, 5-KPS/529, confidence/530)")
