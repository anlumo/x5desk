"""YuNet (face_detection_yunet_2023mar.onnx, batch 1) as a Core ML ML Program for macOS: batch 8 (the face pass's
runs, `privacy::MAX`), FP16, the ONNX's input and output names. A build tool, not run by cargo; the package it writes
is checked in (`yunet_b8.mlpackage`, see README).

    python3.12 -m venv venv
    venv/bin/pip install coremltools==9.0 torch==2.15.0 onnx==1.23.2 onnx2torch==1.5.15 numpy
    venv/bin/python -I coreml.py face_detection_yunet_2023mar.onnx yunet_b8.mlpackage

The export's head reshapes fix the batch at 1 (as `build.rs` fixes for burn): their targets and the input's and
outputs' batch become 8 before onnx2torch traces it.
"""

import sys

import coremltools as ct
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx2torch import convert

B = 8
src, out = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
for v in [g.input[0], *g.output]:
    v.type.tensor_type.shape.dim[0].dim_value = B
shapes = {n.input[1] for n in g.node if n.op_type == "Reshape"}
for i, t in enumerate(g.initializer):
    if t.name in shapes:
        a = numpy_helper.to_array(t).copy()
        assert a[0] == 1, (t.name, a)
        a[0] = B
        g.initializer[i].CopyFrom(numpy_helper.from_array(a, t.name))
net = convert(m).eval()
x = torch.rand(B, 3, 640, 640, generator=torch.Generator().manual_seed(1)) * 255
with torch.no_grad():
    tr = torch.jit.trace(net, x)
ml = ct.convert(
    tr,
    inputs=[ct.TensorType(name="input", shape=(B, 3, 640, 640), dtype=np.float32)],
    outputs=[ct.TensorType(name=o.name, dtype=np.float32) for o in g.output],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15,
)
ml.short_description = "YuNet face detection (OpenCV zoo, MIT), batch 8, converted for x5desk"
ml.save(out)
