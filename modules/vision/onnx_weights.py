"""Store an ONNX model's large weights as fp16, computing in fp32.

Half the file for an image tower, at no measurable cost: every large fp32
initializer is stored as fp16 with a Cast back to fp32 in front of its users,
and both runtimes (OpenVINO, ONNX Runtime) fold the Cast at load, so the
model computes exactly as an fp32 one with rounded weights. The shared frame
encoder ships this way (tools/export_frame_encoder.py), and so does a
fine-tuned action model's own tower (modules/vision/action_models.py).

Needs the ``onnx`` package, which only the steps that write such a file use.
"""
from __future__ import annotations

import numpy as np

MIN_FP16_ELEMENTS = 1024   # smaller tensors (norms, biases) are not worth it


def store_weights_fp16(model):
    """Store every large fp32 initializer as fp16, with a Cast back to fp32 in
    front of its users. Both runtimes fold the Cast at load. Returns
    ``(model, number of tensors converted)``."""
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        weights = numpy_helper.to_array(init)
        half = weights.astype(np.float16)
        if weights.size < MIN_FP16_ELEMENTS or not np.isfinite(half).all():
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(half, name + "__fp16"))
        casts.append(helper.make_node("Cast", [name + "__fp16"], [name],
                                      to=TensorProto.FLOAT, name=name + "__to_fp32"))
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    return model, len(casts)
