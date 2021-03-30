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

VH = 8
VW = 2

input = tvm.placeholder((N, CII, H, W, CIII), name="input")
weight = tvm.placeholder((COO, CII, K, K, CIII, COOO), name="weight")

kh = tvm.reduce_axis((0, K), name="kh")
kw = tvm.reduce_axis((0, K), name="kw")
cii = tvm.reduce_axis((0, CII), name="cii")
ciii = tvm.reduce_axis((0, CIII), name="ciii")

pad = tvm.compute(
    (N, COO, H + 2 * Pad, W + 2 * Pad, COOO),
    lambda n, coo, h, w, cooo: tvm.select(
        tvm.all(h >= Pad, h - Pad < H, w >= Pad, w - Pad < W), input[n, coo, h - Pad, w - Pad, cooo], 0.0),
    name="pad")

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
print("-" * 80)
print("initial schedule")
print("-" * 80)
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

print("-" * 80)
print("final schedule")
print("-" * 80)
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
print("-" * 80)
print(f"gflops = {flops / dur / 1.0e9:.2f}")
print("-" * 80)

"""
--------------------------------------------------------------------------------
initial schedule
--------------------------------------------------------------------------------
// attr [pad] storage_scope = "global"
allocate pad[float32 * 278784]
// attr [vec] storage_scope = "global"
allocate vec[float32 * 655360]
produce pad {
  for (coo, 0, 4) {
    for (h, 0, 66) {
      for (w, 0, 66) {
        for (cooo, 0, 16) {
          pad[((((((coo*66) + h)*66) + w)*16) + cooo)] = tvm_if_then_else(((((1 <= h) && (h < 65)) && (1 <= w)) && (w < 65)), input[(((((((coo*64) + h)*64) + w)*16) + cooo) + -1040)], 0.000000f)
        }
      }
    }
  }
}
produce vec {
  for (cii, 0, 4) {
    for (h, 0, 8) {
      for (w, 0, 32) {
        for (vh, 0, 10) {
          for (vw, 0, 4) {
            for (ciii, 0, 16) {
              vec[((((((((((cii*8) + h)*32) + w)*10) + vh)*4) + vw)*16) + ciii)] = pad[((((((cii*69696) + (h*8448)) + (w*32)) + (vh*1056)) + (vw*16)) + ciii)]
            }
          }
        }
      }
    }
  }
}
produce conv {
  for (coo, 0, 4) {
    for (h, 0, 8) {
      for (w, 0, 32) {
        for (vh, 0, 8) {
          for (vw, 0, 2) {
            for (cooo, 0, 16) {
              pad[((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16) + cooo)] = 0.000000f
              for (cii, 0, 4) {
                for (ciii, 0, 16) {
                  for (kh, 0, 3) {
                    for (kw, 0, 3) {
                      pad[((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16) + cooo)] = (pad[((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16) + cooo)] + (vec[(((((((((((h*32) + w)*10) + vh)*4) + vw) + (cii*10240))*16) + ciii) + (kh*64)) + (kw*16))]*weight[((((((coo*9216) + cooo) + (cii*2304)) + (ciii*16)) + (kh*768)) + (kw*256))]))
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
produce unpack {
  for (coo, 0, 4) {
    for (h, 0, 64) {
      for (w, 0, 64) {
        for (cooo, 0, 16) {
          vec[((((((coo*64) + h)*64) + w)*16) + cooo)] = pad[((((((((coo*8) + (h/8))*256) + (((w/2)*8) + (h % 8)))*2) + (w % 2))*16) + cooo)]
        }
      }
    }
  }
}
produce relu {
  for (coo, 0, 4) {
    for (h, 0, 64) {
      for (w, 0, 64) {
        for (cooo, 0, 16) {
          relu[((((((coo*64) + h)*64) + w)*16) + cooo)] = max(vec[((((((coo*64) + h)*64) + w)*16) + cooo)], 0.000000f)
        }
      }
    }
  }
}

--------------------------------------------------------------------------------
final schedule
--------------------------------------------------------------------------------
// attr [conv] storage_scope = "global"
allocate conv[float32x16 * 1 * 4 * 8 * 32 * 8 * 2 * 1]
// attr [vec] storage_scope = "global"
allocate vec[float32 * 1 * 1 * 1 * 32 * 10 * 4 * 16]
produce conv {
  for (coo, 0, 4) {
    for (h.init, 0, 8) {
      for (w.init, 0, 32) {
        for (vh.init, 0, 8) {
          for (vw.init, 0, 2) {
            conv[ramp((((((((((coo*8) + h.init)*32) + w.init)*8) + vh.init)*2) + vw.init)*16), 1, 16)] = x16(0.000000f)
          }
        }
      }
    }
    for (cii, 0, 4) {
      for (h, 0, 8) {
        produce vec {
          for (w, 0, 32) {
            for (vh, 0, 10) {
              vec[ramp((((w*10) + vh)*64), 1, 16)] = tvm_if_then_else(((((1 - vh) <= (h*8)) && ((h*8) < (65 - vh))) && (1 <= w)), input[ramp((((((((cii*8) + h)*256) + w) + (vh*32))*32) + -1040), 1, 16)], x16(0.000000f))
              vec[ramp(((((w*10) + vh)*64) + 16), 1, 16)] = tvm_if_then_else((((1 - vh) <= (h*8)) && ((h*8) < (65 - vh))), input[ramp((((((((cii*8) + h)*256) + w) + (vh*32))*32) + -1024), 1, 16)], x16(0.000000f))
              vec[ramp(((((w*10) + vh)*64) + 32), 1, 16)] = tvm_if_then_else((((1 - vh) <= (h*8)) && ((h*8) < (65 - vh))), input[ramp((((((((cii*8) + h)*256) + w) + (vh*32))*32) + -1008), 1, 16)], x16(0.000000f))
              vec[ramp(((((w*10) + vh)*64) + 48), 1, 16)] = tvm_if_then_else(((((1 - vh) <= (h*8)) && ((h*8) < (65 - vh))) && (w < 31)), input[ramp((((((((cii*8) + h)*256) + w) + (vh*32))*32) + -992), 1, 16)], x16(0.000000f))
            }
          }
        }
        for (w, 0, 32) {
          for (kw, 0, 3) {
            for (ciii, 0, 16) {
              for (vh, 0, 8) {
                for (vw, 0, 2) {
                  conv[ramp((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16), 1, 16)] = (conv[ramp((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16), 1, 16)] + (x16(vec[((((((w*40) + kw)*16) + ciii) + (vh*64)) + (vw*16))])*weight[ramp((((((((coo*4) + cii)*9) + kw)*16) + ciii)*16), 1, 16)]))
                }
              }
            }
          }
          for (kw, 0, 3) {
            for (ciii, 0, 16) {
              for (vh, 0, 8) {
                for (vw, 0, 2) {
                  conv[ramp((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16), 1, 16)] = (conv[ramp((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16), 1, 16)] + (x16(vec[(((((((w*40) + kw)*16) + ciii) + (vh*64)) + (vw*16)) + 64)])*weight[ramp(((((((((coo*4) + cii)*9) + kw)*16) + ciii)*16) + 768), 1, 16)]))
                }
              }
            }
          }
          for (kw, 0, 3) {
            for (ciii, 0, 16) {
              for (vh, 0, 8) {
                for (vw, 0, 2) {
                  conv[ramp((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16), 1, 16)] = (conv[ramp((((((((((coo*8) + h)*32) + w)*8) + vh)*2) + vw)*16), 1, 16)] + (x16(vec[(((((((w*40) + kw)*16) + ciii) + (vh*64)) + (vw*16)) + 128)])*weight[ramp(((((((((coo*4) + cii)*9) + kw)*16) + ciii)*16) + 1536), 1, 16)]))
                }
              }
            }
          }
        }
      }
    }
  }
}
produce relu {
  for (coo, 0, 4) {
    for (h.outer, 0, 8) {
      for (w.outer, 0, 32) {
        for (h.inner, 0, 8) {
          for (w.inner, 0, 2) {
            relu[ramp(((((((((coo*8) + h.outer)*256) + w.outer) + (h.inner*32))*2) + w.inner)*16), 1, 16)] = max(conv[ramp((((((((((coo*8) + h.outer)*32) + w.outer)*8) + h.inner)*2) + w.inner)*16), 1, 16)], x16(0.000000f))
          }
        }
      }
    }
  }
}

[23:36:59] /home/bertrand/src/tvm-tulloch/src/runtime/rpc/rpc_session.cc:1208: Average time taken per iteration (us): 3047.5
(0.0030475786,)
--------------------------------------------------------------------------------
gflops = 99.09
--------------------------------------------------------------------------------
"""
