import tvm

N = 4
H = 256
R = 3
Pad = 1

a = tvm.placeholder((N, H), name="a")

padded = tvm.compute(
    (N, H + 2 * Pad),
    lambda n, h: tvm.select(
        tvm.all(h >= Pad, h - Pad < H),
        a[n, h - Pad], 0.),
    name="padded")

r = tvm.reduce_axis((0, R), name="r")
conv = tvm.compute(
    (N, H),
    lambda n, h: tvm.sum(padded[n, h + r], axis=r),
    name="conv")

sch = tvm.create_schedule(conv.op)
print(tvm.lower(sch, [a, conv], simple_mode=True))

if False:
    a = tvm.placeholder((1024, 1024), name="a")
    b = tvm.placeholder((1024, 1024), name="b")
    apack = tvm.compute((1024, 1024), lambda i, j: a[i, j], name="apack")
    k = tvm.reduce_axis((0, 1024), "k")
    c = tvm.compute((1024, 1024), lambda i, j: tvm.sum(apack[i, k] * b[k, j], axis=k), name="c")
    s = tvm.create_schedule(c.op)
    i, j = s[c].op.axis
    k, = s[c].op.reduce_axis
    io, ii = s[c].split(i, 64)
    ko, ki = s[c].split(k, 128)
    s[c].reorder(io, ko, j, ki, ii)
    s[apack].compute_at(s[c], ko)
    print(tvm.lower(s, [a, b, c], simple_mode=True))

if False:
    a = tvm.placeholder((3, 4))
    b = tvm.compute((3, 4), lambda *args: a[args] * 11)
    c = tvm.compute((3, 4), lambda *args: b[args] + 17)
    s = tvm.create_schedule(c.op)
    s[b].compute_at(s[c], list(s[c].op.axis)[0])
    print(tvm.lower(s, [a, c], simple_mode=True))
