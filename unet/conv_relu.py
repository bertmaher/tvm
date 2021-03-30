import tvm
import numpy as np

N, CI, H, W = 1, 64, 64, 64
CO = 64
K = 3
Pad = 1
Stride = 1
V = 16

OH = (H - K + 2 * Pad) // Stride + 1
OW = (W - K + 2 * Pad) // Stride + 1

CII = CI // V
CIII = V
COO = CO // V
COOO = V

input = tvm.placeholder((N, CII, H, W, CIII), name="input")
weight = tvm.placeholder((COO, CII, K, K, CIII, COOO), name="weight")

#pad = tvm.compute(
#    (N, C, H + 2 * Pad, W + 2 * Pad

kh = tvm.reduce_axis((0, K), name="kh")
kw = tvm.reduce_axis((0, K), name="kw")
cii = tvm.reduce_axis((0, CII), name="cii")
ciii = tvm.reduce_axis((0, CIII), name="ciii")

pad = tvm.compute(
    (N, COO, H + 2 * Pad, W + 2 * Pad, COOO),
    lambda n, coo, h, w, cooo: tvm.select(
        tvm.all(h >= Pad, h - Pad < H, w >= Pad, w - Pad < W), input[n, coo, h - Pad, w - Pad, cooo], 0.0),
    name="pad")

VH = 8
VW = 2

vec = tvm.compute(
    (N, CII, OH // VH, OW // VW, VH * Stride + K - 1, VW * Stride + K - 1, CIII),
    lambda n, cii, h, w, vh, vw, ciii:
    pad[n, cii, h * VH * Stride + vh, w * VW * Stride + vw, ciii],
    name="vec")

conv = tvm.compute(
    (N, COO, OH // VH, OW // VW, VH, VW, COOO),
    lambda n, coo, h, w, vh, vw, cooo: tvm.sum(
        vec[n, cii, h, w, vh * Stride + kh, vw * Stride + kw, ciii] * weight[coo, cii, kh, kw, ciii, cooo],
        axis=[cii, ciii, kh, kw]),
    name="conv")

unpack = tvm.compute(
    (N, COO, OH, OW, COOO),
    lambda n, coo, h, w, cooo:
    conv[n, coo, h // VH, w // VW, h % VH, w % VW, cooo],
    name="unpack")

relu = tvm.compute(
    (N, COO, OH, OW, COOO),
    lambda n, coo, h, w, cooo: tvm.max(unpack[n, coo, h, w, cooo], 0.0),
    name="relu")
    

sch = tvm.create_schedule(relu.op)
print("initial schedule")
print(tvm.lower(sch, [input, weight, relu], simple_mode=True))

# Schedule convolution.
n, coo, oh, ow, vh, vw, vc = sch[conv].op.axis
cii, ciii, kh, kw = sch[conv].op.reduce_axis
sch[pad].compute_inline()
sch[vec].compute_at(sch[conv], oh)
sch[conv].reorder(n, coo, cii, oh, ow, kh, kw, vc, ciii, vh, vw)
sch[conv].unroll(kh)
sch[conv].vectorize(vc)

# Schedule fused op.
n, coo, h, w, vc = sch[relu].op.axis
sch[relu].vectorize(vc)
oh, vh = sch[relu].split(h, VH)
ow, vw = sch[relu].split(w, VW)
sch[relu].reorder(n, coo, oh, ow, vc, vh, vw)
sch[unpack].compute_inline()

_, _, _, _, vh, vw, vc = sch[vec].op.axis
sch[vec].vectorize(vc)
sch[vec].unroll(vw)

print("final schedule")
print(tvm.lower(sch, [input, weight, relu], simple_mode=True))

target = "llvm -mcpu=skylake-avx512"
func = tvm.build(sch, [input, weight, relu], target=target)
ctx = tvm.context(target, 0)
a_tvm = tvm.nd.array(np.random.rand(N, CII, H, W, CIII).astype(np.float32), ctx=ctx)
b_tvm = tvm.nd.array(np.random.rand(COO, CII, K, K, CIII, COOO).astype(np.float32), ctx=ctx)
c_tvm = tvm.nd.empty((N, COO, OH, OW, COOO), ctx=ctx, dtype="float32")
func(a_tvm, b_tvm, c_tvm)
evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
flops = 2.0 * OH * OW * CO * CI * K * K
dur = evaluator(a_tvm, b_tvm, c_tvm).mean
print(f"gflops = {flops / dur / 1.0e9:.2f}")
"""
relu = tvm.compute(
    (N, CO, OH, OW),
    lambda n, c, h, w: tvm.max(0., conv[n, c, h, w]))

#sch = tvm.create_schedule(relu.op)

n, c, h, w = sch[conv].op.axis
sch[conv].reorder(n, h, w, c)

n, c, h, w = sch[relu].op.axis
sch[relu].reorder(n, h, w, c)

sch[conv].compute_at(sch[relu], c)

print(tvm.lower(sch, [input, weight, relu], simple_mode=True))
"""
