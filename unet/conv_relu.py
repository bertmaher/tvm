import tvm

N, C, H, W = 1, 64, 66, 66
K = 3
Pad = 0
Stride = 1

kh = tvm.reduce_axis((0, K), name="kh")
kw = tvm.reduce_axis((0, K), name="kw")
ci = tvm.reduce_axis((0, C), name="ci")

OH = (H - K + 2 * Pad) // Stride + 1
OW = (W - K + 2 * Pad) // Stride + 1

input = tvm.placeholder((N, C, H, W))
weight = tvm.placeholder((C, C, K, K))
conv = tvm.compute(
    (N, C, OH, OW),
    lambda n, co, h, w: tvm.sum(input[n, ci, h+kh, w+kw] * weight[co, ci, kh, kw],
                                axis=[ci, kh, kw]))
relu = tvm.compute(
    (N, C, OH, OW),
    lambda n, c, h, w: tvm.max(0., conv[n, c, h, w]))

sch = tvm.create_schedule(relu.op)

n, c, h, w = sch[conv].op.axis
sch[conv].reorder(n, h, w, c)

n, c, h, w = sch[relu].op.axis
sch[relu].reorder(n, h, w, c)

sch[conv].compute_at(sch[relu], c)

print(tvm.lower(sch, [input, weight, relu], simple_mode=True))
