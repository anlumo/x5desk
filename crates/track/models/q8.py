"""Stores an ONNX model's float weights of two or more dimensions and >= 1024 elements as int8 with a float scale per
first-axis slice (symmetric, largest magnitude to 127), each followed by Cast and Mul back to float32; the other float
weights of 64 elements or more as float16 and a Cast. The graph computes in float32 as before, at a quarter of the
size. A build tool, not run by cargo; what it writes is checked in (tinyclip_39m_image.onnx, tinyclip_39m_text.onnx,
see README).

    python -I q8.py in.onnx out.onnx    (onnx 1.23.2, numpy 2.5.4)
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

m = onnx.load(sys.argv[1])
g = m.graph
new, n8, n16 = [], 0, 0


def store(a, name):
    """Nodes and initializers that rebuild float32 `a` under `name`."""
    global n8, n16
    if a.ndim >= 2 and a.size >= 1024:
        s = np.abs(a).reshape(a.shape[0], -1).max(1) / 127
        s = np.where(s > 0, s, 1).astype(np.float32).reshape((-1,) + (1,) * (a.ndim - 1))
        q = np.clip(np.round(a / s), -127, 127).astype(np.int8)
        g.initializer.extend([numpy_helper.from_array(q, name + "_q8"), numpy_helper.from_array(s, name + "_scale")])
        new.append(helper.make_node("Cast", [name + "_q8"], [name + "_f32"], to=TensorProto.FLOAT, name=name + "_cast"))
        new.append(helper.make_node("Mul", [name + "_f32", name + "_scale"], [name], name=name + "_dq"))
        n8 += a.size
    else:
        g.initializer.append(numpy_helper.from_array(a.astype(np.float16), name + "_f16"))
        new.append(helper.make_node("Cast", [name + "_f16"], [name], to=TensorProto.FLOAT, name=name + "_cast"))
        n16 += a.size


for t in list(g.initializer):
    if t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= 64:
        a = numpy_helper.to_array(t)
        g.initializer.remove(t)
        store(a, t.name)
for n in list(g.node):
    if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
        t = n.attribute[0].t
        if t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= 64:
            g.node.remove(n)
            store(numpy_helper.to_array(t), n.output[0])
# the new nodes go first: graph nodes must be in topological order
for c in reversed(new):
    g.node.insert(0, c)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print(f"{n8} weights to int8, {n16} to float16, {len(new)} nodes")
