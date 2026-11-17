"""Derive a BPU-shaped graph from the packaged Nature TCN ONNX.

Why each change exists (X5 / bayes-e constraints):
  1. 4D NCHW input + (1,K) kernels.  The X5 operator list only accelerates conv2d/conv3d,
     and the 1D form is only listed under CPU constraints.
  2. Length 264 instead of 249.  Dilated conv requires the input W to be divisible by the
     dilation.  With 249 every dilated layer sees 4d*k+1 samples -> not divisible.
     Prepending zero frames keeps the last output mathematically identical, because the
     receptive field is exactly 249 samples and the zeros fall outside it.
  3. Left mirroring + center/scale moved out of the graph.  The raw input tensor mixes gyro
     (per-channel scale 38..123) and accelerometer (1.8..6.0) channels; a single per-tensor
     INT8 activation scale would crush the accelerometer channels.
  4. The 80->2 linear head is cut off.  It is 160 MACs; keeping it on the CPU in FP32 costs
     nothing and preserves output precision at the dequantisation boundary.

Weights are copied unchanged.  Nothing is retrained, quantised or reordered.
"""
from pathlib import Path
import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SRC = Path("/Users/lucas/Documents/Stride-Assist-main/Control/nature_tcn/models/nature_latest_fp32.onnx")
OUT_DIR = Path(__file__).resolve().parents[1] / "models"
DST = OUT_DIR / "nature_bpu.onnx"
PRE = OUT_DIR / "preprocess.json"

PAD_TO = 264          # 264 - 249 = 15 zero frames prepended on the CPU side
HISTORY = 249


def main():
    m = onnx.load(str(SRC))
    g = m.graph
    m_init = list(g.initializer)
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    mirror = init["mirror"].reshape(2, 12)

    # first conv consumes the normalised tensor; that node becomes the graph input
    first_conv = next(n for n in g.node if n.op_type == "Conv")
    norm_tensor = first_conv.input[0]

    # Slice axes 2 -> 3, and the Gather axis 2 -> 3 (the time axis moved to dim 3).
    # The causal crop start must be recomputed: the residual shortcut is cropped to the
    # length the block's convolutions produce, and that length depends on PAD_TO.
    crops, cur = [], PAD_TO
    for b in range(5):
        blk = f"/latest/blocks/blocks.{b}/"
        d = []
        for n in g.node:
            if n.op_type == "Conv" and n.name.startswith(blk) and "downsample" not in n.name:
                d.append(int(helper.get_attribute_value(
                    [a for a in n.attribute if a.name == "dilations"][0])[0]))
        if len(d) != 2:
            raise SystemExit(f"unexpected conv count in block {b}: {d}")
        cur -= 4 * (d[0] + d[1])
        crops.append(-cur)

    axes_fix = {}
    for b, start in enumerate(crops):
        blk = f"/latest/blocks/blocks.{b}/"
        for n in g.node:
            if n.op_type != "Slice" or not n.name.startswith(blk):
                continue
            axes_fix[n.input[3]] = np.array([3], np.int64)
            axes_fix[n.input[1]] = np.array([int(start)], np.int64)

    # graph output: the last ReLU of the deepest block, shape [2,80,1,16]
    last_relu = [n for n in g.node if n.op_type == "Relu"][-1].output[0]

    new_nodes, new_inits = [], []
    for n in g.node:
        if n.op_type in ("Mul", "Sub", "Div"):
            continue                      # preprocessing now happens on the CPU
        if n.op_type == "Gather":          # head is executed on the CPU in FP32
            break
        if n.op_type == "Constant" and n.output[0] in axes_fix:
            new_nodes.append(helper.make_node(
                "Constant", [], [n.output[0]],
                value=helper.make_tensor("v", TensorProto.INT64, [1],
                                         axes_fix[n.output[0]].astype(np.int64))))
            continue
        if n.op_type == "Conv":
            w_name, b_name = n.input[1], (n.input[2] if len(n.input) > 2 else None)
            w = init[w_name]
            if w.ndim != 3:
                raise SystemExit(f"unexpected weight rank {w.shape} for {n.name}")
            w4 = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
            name4 = w_name.replace("\n", "") + "_4d"
            new_inits.append(numpy_helper.from_array(w4, name4))
            dil = int(helper.get_attribute_value([a for a in n.attribute if a.name == "dilations"][0])[0])
            ins = ["raw_history" if i == norm_tensor else i for i in n.input]
            ins[1] = name4
            new_nodes.append(helper.make_node(
                "Conv", ins, [n.output[0]],
                kernel_shape=[1, w.shape[2]], strides=[1, 1], pads=[0, 0, 0, 0],
                dilations=[1, dil], group=1))
            continue
        ins = ["raw_history" if i == norm_tensor else i for i in n.input]
        new_nodes.append(helper.make_node(n.op_type, ins, list(n.output),
                                          **{a.name: helper.get_attribute_value(a)
                                             for a in n.attribute}))

    del g.node[:]
    g.node.extend(new_nodes)
    del g.input[:]
    g.input.extend([helper.make_tensor_value_info("raw_history", TensorProto.FLOAT,
                                                  [2, 12, 1, PAD_TO])])
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info(last_relu, TensorProto.FLOAT,
                                                   [2, 80, 1, 16])])
    del g.initializer[:]
    used = {i for n in new_nodes for i in n.input}
    kept = [t for t in m_init if t.name in used]
    g.initializer.extend(kept + new_inits)
    del g.value_info[:]

    onnx.checker.check_model(m)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(DST))

    PRE.write_text(json.dumps({
        "schema": "nature_bpu_preprocess_v1",
        "source_model_sha256": "9fbd709054e3da730cef43f7a5e3ccd9f670d205f1e3215b0c93e979acd248a0",
        "history": HISTORY,
        "graph_length": PAD_TO,
        "zero_prefix_frames": PAD_TO - HISTORY,
        "sample_hz": 200,
        "steps": [
            "raw [2,12,249] in author sensor frames, degrees/s and m/s^2, BEFORE mirroring",
            "mirrored = raw * mirror",
            "normalised = (mirrored - center) / scale   (per channel)",
            "input4d[2,12,1,264]: 15 zero frames, then the 249 normalised frames",
        ],
        "mirror": mirror.tolist(),
        "center": init["latest.center"].reshape(12).tolist(),
        "scale": init["latest.scale"].reshape(12).tolist(),
        "graph_output": {"name": last_relu, "shape": [2, 80, 1, 16],
                         "note": "take [:, :, 0, -1] -> [2,80], then @ head_weight.T + head_bias"},
        "head_weight": init["latest.linear.weight"].tolist(),
        "head_bias": init["latest.linear.bias"].tolist(),
        "output_order": ["LH", "LK", "RH", "RK"],
        "output_units": "net Nm/kg",
    }, indent=2) + "\n")

    print("wrote", DST, DST.stat().st_size, "bytes")

    # Some OE front-ends reject high opsets; emit a semantically identical
    # opset-11 copy so the checker can be tried against both.
    # onnx.version_converter has no 17->11 adapter chain, but every op used here
    # (Conv, Relu, Slice-10 signature, Add/Mul/Sub/Div, Gather, Gemm, Reshape,
    # Constant) has identical semantics in opset 11, so restamping is exact.
    lo = onnx.load(str(DST))
    for ops in lo.opset_import:
        if ops.domain in ("", "ai.onnx"):
            ops.version = 11
    lo.ir_version = 7
    LOW = DST.with_name("nature_bpu_opset11.onnx")
    onnx.save(lo, LOW)
    onnx.checker.check_model(str(LOW))
    print("wrote", LOW, LOW.stat().st_size, "bytes  opset=11 ir=7")

    print("wrote", PRE)


if __name__ == "__main__":
    main()
