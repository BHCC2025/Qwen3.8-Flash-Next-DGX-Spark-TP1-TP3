import json, os, sys, torch
os.environ["QWEN4EXP_TP_PAD"] = "1"
sys.path.insert(0, "/work")
import tp_pad
from safetensors import safe_open

M = "/models/qwen38fn"
wm = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{M}/{wm[n]}", "pt") as f: return f.get_tensor(n)
def tr(n):
    return tp_pad._transform(n, get(n))
ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)

# 1. full attention GQA: output with 2 kv heads (group 12) == 6 replicated kv heads (group 4)
L = "model.language_model.layers.3.self_attn."
k0, v0 = get(L + "k_proj.weight").float(), get(L + "v_proj.weight").float()
k1, v1 = tr(L + "k_proj.weight").float(), tr(L + "v_proj.weight").float()
check(k1.shape == (6 * 256, 2560) and v1.shape == (6 * 256, 2560), f"kv shapes {tuple(k1.shape)}")
x = torch.randn(7, 2560)
q = torch.randn(7, 24, 256)
def attn(k_w, v_w, nkv):
    K = (x @ k_w.T).view(7, nkv, 256); V = (x @ v_w.T).view(7, nkv, 256)
    g = 24 // nkv
    Ke = K.repeat_interleave(g, dim=1); Ve = V.repeat_interleave(g, dim=1)
    s = torch.einsum("thd,shd->hts", q, Ke) / 16
    return torch.einsum("hts,shd->thd", s.softmax(-1), Ve)
a0, a1 = attn(k0, v0, 2), attn(k1, v1, 6)
check(torch.allclose(a0, a1, atol=1e-4), f"GQA output identical (max diff {(a0 - a1).abs().max():.2e})")
# per-rank split: rank r gets q heads 8r..8r+7 and kv heads 2r,2r+1
for r in range(3):
    need = {h // 12 for h in range(8 * r, 8 * r + 8)}
    have = {j // 3 for j in (2 * r, 2 * r + 1)}
    check(need <= have, f"rank {r}: q heads need orig kv {sorted(need)}, local kv are copies of {sorted(have)}")

# 2. GatedDeltaNet: v head i must read (a copy of) original k head i//3
G = "model.language_model.layers.0.linear_attn."
w0, w1 = get(G + "in_proj_qkv.weight"), tr(G + "in_proj_qkv.weight")
c0, c1 = get(G + "conv1d.weight"), tr(G + "conv1d.weight")
check(w1.shape == (18432, 2560) and c1.shape == (18432, 1, 4), f"gdn shapes {tuple(w1.shape)} {tuple(c1.shape)}")
hd, kd0, kd1 = 128, 2048, 6144
good = True
for i in range(48):
    for off0, off1 in ((0, 0), (kd0, kd1)):      # q block then k block
        src = w0[off0 + (i // 3) * hd: off0 + (i // 3 + 1) * hd]
        dst = w1[off1 + i * hd: off1 + (i + 1) * hd]
        good &= torch.equal(src, dst)
        good &= torch.equal(c0[off0 + (i // 3) * hd: off0 + (i // 3 + 1) * hd], c1[off1 + i * hd: off1 + (i + 1) * hd])
check(good, "gdn: new q/k head i == original head i//3 (weights and conv)")
check(torch.equal(w0[2 * kd0:], w1[2 * kd1:]) and torch.equal(c0[2 * kd0:], c1[2 * kd1:]), "gdn: v block untouched")
check(all(n % 3 == 0 for n in (kd1, kd1, 6144)), "gdn: q/k/v dims 6144/6144/6144 divide by 3")

# 3. experts / shared expert padding
E = "model.language_model.layers.0.mlp.experts.0."
want = {"gate_proj.weight": (768, 1280), "gate_proj.weight_scale": (768, 160),
        "up_proj.weight": (768, 1280), "down_proj.weight": (2560, 384), "down_proj.weight_scale": (2560, 48)}
for leaf, shp in want.items():
    t0, t1 = get(E + leaf), tr(E + leaf)
    real = t1[:640] if leaf.startswith(("gate", "up")) else t1[:, : t0.shape[1]]
    pad = t1[640:] if leaf.startswith(("gate", "up")) else t1[:, t0.shape[1]:]
    check(tuple(t1.shape) == shp and torch.equal(real.view(torch.uint8), t0.view(torch.uint8))
          and not pad.view(torch.uint8).any(), f"nvfp4 {leaf} {tuple(t0.shape)}->{tuple(t1.shape)}, real kept, pad zero")
check(tr(E + "gate_proj.weight_scale_2") .shape == (), "nvfp4 scalar scales untouched")
X = "mtp.layers.0.mlp.experts.0."
for leaf, shp, fill in (("gate_proj.weight", (768, 2560), 0), ("gate_proj.weight_scale_inv", (6, 20), 1.0),
                        ("down_proj.weight", (2560, 768), 0), ("down_proj.weight_scale_inv", (20, 6), 1.0)):
    t1 = tr(X + leaf)
    pad = t1[-1] if leaf.startswith("gate") else t1[:, -1]
    check(tuple(t1.shape) == shp and bool((pad.float() == fill).all()), f"mtp fp8 {leaf} -> {tuple(t1.shape)} pad={fill}")
S = "model.language_model.layers.0.mlp.shared_expert."
g0, u0, d0 = get(S + "gate_proj.weight").float(), get(S + "up_proj.weight").float(), get(S + "down_proj.weight").float()
g1, u1, d1 = tr(S + "gate_proj.weight").float(), tr(S + "up_proj.weight").float(), tr(S + "down_proj.weight").float()
xx = torch.randn(5, 2560)
y0 = (torch.nn.functional.silu(xx @ g0.T) * (xx @ u0.T)) @ d0.T
y1 = (torch.nn.functional.silu(xx @ g1.T) * (xx @ u1.T)) @ d1.T
check(g1.shape == (768, 2560) and d1.shape == (2560, 768) and torch.allclose(y0, y1, atol=1e-3),
      f"shared expert MLP output identical (max diff {(y0 - y1).abs().max():.2e})")
check(768 // 3 == 256 and 256 % 128 == 0, "moe intermediate per rank 256 (2 FP8 blocks, 16 NVFP4 groups)")
check(tr("model.language_model.layers.0.mlp.gate.weight").shape == (512, 2560), "router untouched")
print("ALL PASS" if ok else "SOME FAILED")
