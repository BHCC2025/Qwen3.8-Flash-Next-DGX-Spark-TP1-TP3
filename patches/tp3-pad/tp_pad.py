"""Pad / replicate Qwen3.8-Flash-Next checkpoint tensors so TP=3 can load.

Mounted as vllm/models/qwen4_exp/nvidia/tp_pad.py and called from the load_weights of
Qwen4ExpForConditionalGeneration (model.py) and the MTP draft model (mtp.py). Enabled with
QWEN4EXP_TP_PAD=1; the model dir must carry a config.json edited to the padded sizes
(num_key_value_heads, linear_num_key_heads, moe_intermediate_size,
shared_expert_intermediate_size), which is what makes vLLM allocate the padded params.

Dims of the checkpoint that do not divide by 3, and what happens to each:
  full attention  num_key_value_heads 2    -> 6    each KV head copied 3x. q head h reads
                                                   kv h//4 = copy of original kv h//12: exact.
  GatedDeltaNet   linear_num_key_heads 16  -> 48   each q/k head (and its conv channels)
                                                   copied 3x, so v head i reads k head i =
                                                   copy of original k head i//3: exact.
  routed experts  moe_intermediate 640     -> 768  zero rows/cols (NVFP4 main, FP8-block MTP):
  shared expert   shared_expert_int 640    -> 768  silu(0)*0 = 0, zero down cols: exact.
                                                   768/3 = 256 per rank, 2 FP8 blocks of 128.
Replicated or vocab-parallel already (no change): indexer, hyperconnections, PLE, embeddings,
lm_head. MTP fc_embedding/fc_hidden (2560 out, not /3) are made replicated in mtp.py.
The padded block scales of the FP8 MTP experts are 1.0, not 0 (a 0/denormal scale on an
all-zero block tripped the DeepSeek-V4.1 TP3 re-encode; 0 x 1 = 0 either way).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator

import torch

logger = logging.getLogger(__name__)

KV_HEADS, KV_REP = 2, int(os.environ.get("QWEN4EXP_KV_REP", "3"))
GDN_K_HEADS, GDN_K_REP, GDN_HEAD_DIM = 16, int(os.environ.get("QWEN4EXP_GDN_K_REP", "3")), 128
FA_HEAD_DIM = 256
I_OLD, I_NEW = 640, int(os.environ.get("QWEN4EXP_MOE_I", "768"))


def enabled() -> bool:
    return os.environ.get("QWEN4EXP_TP_PAD", "0") not in ("0", "", "false", "off")


def _rep_heads(t: torch.Tensor, heads: int, head_dim: int, rep: int) -> torch.Tensor:
    rest = t.shape[1:]
    return t.reshape(heads, head_dim, *rest).repeat_interleave(rep, dim=0).reshape(heads * head_dim * rep, *rest)


def _pad_dim(t: torch.Tensor, dim: int, fill: float) -> torch.Tensor:
    old = t.shape[dim]
    if old * I_NEW % I_OLD:
        raise ValueError(f"tp_pad: dim {dim} of {tuple(t.shape)} does not scale {I_OLD}->{I_NEW}")
    new = old * I_NEW // I_OLD
    shape = list(t.shape)
    shape[dim] = new
    out = (torch.full(shape, fill, dtype=t.dtype, device=t.device) if fill
           else torch.zeros(shape, dtype=t.dtype, device=t.device))
    out.narrow(dim, 0, old).copy_(t)
    return out


def _transform(name: str, t: torch.Tensor) -> torch.Tensor:
    if name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        assert t.shape[0] == KV_HEADS * FA_HEAD_DIM, (name, t.shape)
        return _rep_heads(t, KV_HEADS, FA_HEAD_DIM, KV_REP)

    if name.endswith(("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")):
        kd = GDN_K_HEADS * GDN_HEAD_DIM
        assert t.shape[0] > 2 * kd, (name, t.shape)
        q, k, v = t[:kd], t[kd:2 * kd], t[2 * kd:]
        return torch.cat([_rep_heads(q, GDN_K_HEADS, GDN_HEAD_DIM, GDN_K_REP),
                          _rep_heads(k, GDN_K_HEADS, GDN_HEAD_DIM, GDN_K_REP), v], dim=0)

    if ".mlp.experts." in name or ".mlp.shared_expert." in name:
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("input_scale", "weight_scale_2") or t.dim() == 0:
            return t
        if ".gate_proj." in name or ".up_proj." in name:
            dim = 0                     # [I, H] weight / [I, H/16] or [I/128, H/128] scale
        elif ".down_proj." in name:
            dim = 1                     # [H, I] weight (U8 packs 2/byte) / [H, I/16] or [H/128, I/128]
        else:
            return t
        return _pad_dim(t, dim, 1.0 if leaf == "weight_scale_inv" else 0.0)
    return t


def transform(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    if not enabled():
        yield from weights
        return
    counts: dict[str, int] = {}
    for name, t in weights:
        new = _transform(name, t)
        if new is not t:
            key = ("kv" if "self_attn" in name else "gdn" if "linear_attn" in name
                   else "shared" if "shared_expert" in name else "experts")
            counts[key] = counts.get(key, 0) + 1
        yield name, new
    logger.warning("Qwen4Exp TP pad: transformed %s (kv x%d, gdn k x%d, moe I %d->%d)",
                   counts, KV_REP, GDN_K_REP, I_OLD, I_NEW)


def install() -> None:
    """Make every VocabParallelEmbedding / ParallelLMHead pad its vocab to a multiple of padding_size * TP.

    vLLM pads the vocab to a multiple of 64 and then divides it by TP; 248320 is a multiple of 64 but not of 3.
    Padding to 64*3 = 192 gives 248448 (82816 = 64*1294 rows per rank). The extra rows are zero and vLLM's
    LogitsProcessor already trims logits back to org_num_embeddings. Same fix as the DeepSeek-V4.1 TP3 adapter.
    Called at import of model.py (before any layer is built); a no-op unless QWEN4EXP_TP_PAD=1.
    """
    if not enabled():
        return
    import inspect
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe

    cls = vpe.VocabParallelEmbedding
    if getattr(cls.__init__, "_qwen4exp_tp_pad", False):
        return
    orig = cls.__init__
    sig = inspect.signature(orig)

    def __init__(self, *args, **kwargs):
        from vllm.distributed import get_tensor_model_parallel_world_size
        b = sig.bind(self, *args, **kwargs)
        b.apply_defaults()
        tp = get_tensor_model_parallel_world_size()
        if tp > 1 and not b.arguments.get("disable_tp", False):
            ps = int(b.arguments["padding_size"])
            if ps % tp:
                b.arguments["padding_size"] = ps * tp
        orig(*b.args, **b.kwargs)

    __init__._qwen4exp_tp_pad = True
    cls.__init__ = __init__
    logger.warning("Qwen4Exp TP pad: vocab-parallel layers pad to padding_size*TP")
