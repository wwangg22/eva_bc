# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the BC pipeline end-to-end on CPU, with no simulator, before spending GPU hours.

Why this exists
---------------
Three separate times on this task an unvalidated instrument produced a confident wrong number
(a determinism test that ran its own control condition last; a slip censor that measured past
the release; a headline success rate that was silently conditional). Training is the next
instrument, and it is the expensive one: two arms x three seeds x 100 k steps. Everything in
that path except the Isaac Sim rollout loop is ordinary PyTorch and can be checked in seconds.

What it checks, and why each one can actually fail
--------------------------------------------------
1. **Pool filters select what they claim.** ``--pool nominal`` silently drops every DART demo
   (``episode_kind == "dart"`` fails its ``== "nominal"`` test). Used for arm B that turns a
   1024-demo arm into a 512-demo one and re-creates the exact volume confound the experiment
   was designed to avoid -- while printing nothing unusual.
2. **The loss censor reaches the loss.** ``train_mask`` rides the ``action_is_pad`` channel.
   If the dataset dropped it, or the loss stopped multiplying by ``~action_is_pad``, censoring
   would be inert and nothing would say so. Checked by construction on a synthetic demo whose
   mask is known, and then by confirming a masked step contributes exactly zero gradient.
3. **Normalization round-trips.** The controller normalizes observations and unnormalizes the
   predicted chunk with the *checkpoint's* stats. A mismatch here is a silent scale error on
   joint targets -- the policy would look trained and behave like a different robot.
4. **The checkpoint round-trips bit-exactly.** ``load_checkpoint`` rebuilds the architecture
   from a config dict rather than storing it, so an architecture drift between train and eval
   loads a state dict into a *different* model. ``load_state_dict`` catches shape changes but
   not, say, a changed ``num_inference_steps`` or a changed chunk size that still fits.
5. **Chunk commitment holds.** This is the load-bearing property of the whole method: on
   pick-place, shortening the execution horizon collapsed success 59.4 -> 32.8 -> 3.1 -> 0 -> 0 %
   at ``n_action_steps`` 15/8/4/2/1. So the controller must call the policy exactly once per
   ``n_action_steps`` control steps, and a reset must force a fresh prediction immediately.
   Counted directly, via a proxy that tallies forward passes.

.. code-block:: bash

    python slot/scripts/test_pipeline_cpu.py --data slot/data/v2/nominal_s0.hdf5 \
                                             slot/data/v2/dart005_s20.hdf5
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_act.dataset import (  # noqa: E402
    ACTION_DIM,
    ENV_STATE_SLICE,
    OBS_DIM,
    STATE_SLICE,
    RebotDemoDataset,
    compute_stats,
    nominal_pool_filter,
    recovery_pool_filter,
    success_pool_filter,
)
from slot_act.eval_act import BatchedACTController, load_checkpoint  # noqa: E402
from slot_act.modeling_flow import FlowMatchingPolicy  # noqa: E402
from slot_act.normalize import MeanStdNormalizer  # noqa: E402
from slot_act.train_flow import make_config, save_checkpoint  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


class CountingPolicy(torch.nn.Module):
    """Wraps a policy and tallies predict_action_chunk calls and the batch sizes it saw."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.calls = 0
        self.batch_sizes: list[int] = []

    def predict_action_chunk(self, batch, **kw):
        self.calls += 1
        self.batch_sizes.append(int(next(iter(batch.values())).shape[0]))
        return self.policy.predict_action_chunk(batch, **kw)


# ---------------------------------------------------------------------------- 1. pool filters
def test_pool_filters(paths: list[Path]) -> None:
    print("\n1. POOL FILTERS -- do they select what the experiment assumes?")
    import h5py

    kinds: dict[str, int] = {}
    got = {"success": 0, "nominal": 0, "recovery": 0, "n_demos": 0, "n_failed": 0}
    for p in paths:
        with h5py.File(str(p), "r") as f:
            for k in f["data"].keys():
                a = {kk: (vv.decode() if isinstance(vv, bytes) else
                          vv.item() if isinstance(vv, np.generic) else vv)
                     for kk, vv in f["data"][k].attrs.items()}
                kind = str(a.get("episode_kind"))
                kinds[kind] = kinds.get(kind, 0) + 1
                got["n_demos"] += 1
                got["n_failed"] += (not bool(a.get("success")))
                got["success"] += bool(success_pool_filter(a))
                got["nominal"] += bool(nominal_pool_filter(a))
                got["recovery"] += bool(recovery_pool_filter(a))

    print(f"    {got['n_demos']} demos, kinds = {kinds}, {got['n_failed']} failed")
    check("success filter == every successful demo",
          got["success"] == got["n_demos"] - got["n_failed"],
          f"{got['success']} kept vs {got['n_demos'] - got['n_failed']} successful")
    check("nominal + recovery partition the successful demos",
          got["nominal"] + got["recovery"] == got["success"],
          f"{got['nominal']} nominal + {got['recovery']} recovery = {got['success']}")
    if "dart" in kinds:
        check("nominal filter EXCLUDES dart demos (this is why arm B needs --pool success)",
              got["nominal"] <= got["n_demos"] - kinds["dart"],
              f"nominal kept {got['nominal']}, non-dart demos number "
              f"{got['n_demos'] - kinds['dart']}")


# --------------------------------------------------------------- 2. the censor reaches the loss
def test_mask_reaches_loss(chunk_size: int) -> None:
    """Build a 2-frame synthetic demo whose train_mask is known, and prove a censored step
    contributes exactly zero gradient. Uses the real dataset __getitem__ + the real loss."""
    print("\n2. LOSS CENSOR -- does train_mask actually zero a gradient?")
    import h5py

    T = 40
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "synthetic.hdf5"
        with h5py.File(str(p), "w") as f:
            d = f.create_group("data")
            d.attrs["total"] = 1
            g = d.create_group("demo_0")
            g.create_dataset("obs/policy", data=np.zeros((T, OBS_DIM), np.float32))
            g.create_dataset("actions", data=np.arange(T * ACTION_DIM, dtype=np.float32
                                                       ).reshape(T, ACTION_DIM))
            mask = np.ones(T, np.uint8)
            mask[T // 2:] = 0  # censor the back half
            g.create_dataset("train_mask", data=mask)
            g.attrs.update({"num_samples": T, "success": True, "episode_kind": "nominal",
                            "outcomes": "{}", "segments": "[]"})

        ds = RebotDemoDataset([p], chunk_size=chunk_size, demo_filter=None)
        s = ds[0]  # t = 0: chunk covers [0, chunk_size)
        pad = s["action_is_pad"].numpy()
        expect = np.zeros(chunk_size, bool)
        expect[T // 2:] = True  # censored steps, and everything past T is edge-pad -> also True
        check("action_is_pad carries train_mask == 0", bool((pad == expect).all()),
              f"{int(pad.sum())} of {chunk_size} steps padded/censored, expected {int(expect.sum())}")

        # A censored step must contribute no gradient. Compare the loss gradient w.r.t. a
        # prediction that is wrong ONLY on censored steps against one that is right everywhere.
        cfg = make_config(argparse.Namespace(chunk_size=chunk_size, n_action_steps=15,
                                             num_inference_steps=10, device="cpu"))
        policy = FlowMatchingPolicy(cfg)
        batch = {k: v.unsqueeze(0) for k, v in s.items()}
        torch.manual_seed(0)
        loss_a, _ = policy.forward({**batch, "action": batch["action"].clone()})
        corrupted = batch["action"].clone()
        corrupted[:, T // 2:] += 1e3  # garbage, but only where the mask is 0
        torch.manual_seed(0)
        loss_b, _ = policy.forward({**batch, "action": corrupted})
        check("loss is INVARIANT to garbage on censored steps",
              torch.allclose(loss_a, loss_b, atol=1e-5),
              f"{loss_a.item():.6f} vs {loss_b.item():.6f}")

        torch.manual_seed(0)
        corrupted2 = batch["action"].clone()
        corrupted2[:, : T // 4] += 1e3  # garbage on TRAINABLE steps must change the loss
        loss_c, _ = policy.forward({**batch, "action": corrupted2})
        check("loss DOES respond to garbage on trainable steps",
              not torch.allclose(loss_a, loss_c, atol=1e-3),
              f"{loss_a.item():.6f} vs {loss_c.item():.6f}")


# ------------------------------------------------------------------------- 3/4. train + reload
def test_train_and_reload(paths: list[Path], chunk_size: int, n_action_steps: int, steps: int):
    print("\n3. NORMALIZER + 4. CHECKPOINT ROUND-TRIP")
    ds = RebotDemoDataset(paths, chunk_size=chunk_size, demo_filter=success_pool_filter)
    stats = compute_stats(ds)
    print(f"    dataset: {len(ds.demos)} demos, {len(ds)} samples")

    norm = MeanStdNormalizer(stats)
    x = torch.randn(8, STATE_SLICE.stop - STATE_SLICE.start)
    rt = norm.unnormalize("observation.state", norm.normalize({"observation.state": x})["observation.state"])
    check("normalize -> unnormalize round-trips", torch.allclose(x, rt, atol=1e-4),
          f"max abs error {(x - rt).abs().max().item():.2e}")
    check("no zero/degenerate std in the stats",
          all(float(v["std"].min()) > 0 for v in stats.values()),
          "  ".join(f"{k}:min_std={float(v['std'].min()):.2e}" for k, v in stats.items()))

    cfg = make_config(argparse.Namespace(chunk_size=chunk_size, n_action_steps=n_action_steps,
                                         num_inference_steps=10, device="cpu"))
    torch.manual_seed(0)
    policy = FlowMatchingPolicy(cfg)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    losses = []
    it = iter(loader)
    for _ in range(steps):
        b = norm.normalize({k: v for k, v in next(it).items()})
        loss, _ = policy.forward(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    check("training loss is finite", bool(np.isfinite(losses).all()),
          f"first {losses[0]:.4f}  last {losses[-1]:.4f}")
    check("gradients reached the trunk (params moved)",
          any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in policy.parameters()))

    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / "ckpt.pt"
        save_checkpoint(ck, policy, norm, cfg, steps)
        loaded, lstats, lcfg = load_checkpoint(ck, "cpu")
        check("checkpoint config survives the round-trip",
              lcfg["chunk_size"] == chunk_size and lcfg["n_action_steps"] == n_action_steps
              and lcfg["policy_type"] == "flow", str({k: lcfg[k] for k in
                                                      ("chunk_size", "n_action_steps", "policy_type")}))
        same = all(torch.equal(a, b) for a, b in
                   zip(policy.state_dict().values(), loaded.state_dict().values()))
        check("every weight is bit-identical after reload", same)
        check("normalizer stats survive the round-trip",
              all(torch.allclose(stats[k]["mean"], lstats[k]["mean"]) and
                  torch.allclose(stats[k]["std"], lstats[k]["std"]) for k in stats))

        # Same x0 + same weights must give the same chunk, or nothing downstream is comparable.
        obs = torch.from_numpy(ds.demos[0]["obs"][:4]).float()
        x0 = torch.randn(chunk_size, ACTION_DIM)
        policy.eval()
        c1 = BatchedACTController(policy, stats, n_action_steps, chunk_size, "cpu", fixed_x0=x0).act(obs)
        c2 = BatchedACTController(loaded, lstats, n_action_steps, chunk_size, "cpu", fixed_x0=x0).act(obs)
        check("saved and reloaded policies produce the SAME action from the same x0",
              torch.allclose(c1, c2, atol=1e-6), f"max abs diff {(c1 - c2).abs().max().item():.2e}")
        return loaded, lstats, ds


# ------------------------------------------------------------------- 5. chunk-commitment cadence
def test_chunk_commitment(policy, stats, ds, chunk_size: int, n_action_steps: int) -> None:
    print("\n5. CHUNK COMMITMENT -- one forward per n_action_steps, and a reset forces a refill")
    n_env, n_steps = 4, 3 * n_action_steps
    cp = CountingPolicy(policy)
    ctrl = BatchedACTController(cp, stats, n_action_steps, chunk_size, "cpu")
    obs = torch.from_numpy(ds.demos[0]["obs"][:n_env]).float()
    acts = torch.stack([ctrl.act(obs) for _ in range(n_steps)])  # (n_steps, n_env, 7)

    check(f"exactly {n_steps // n_action_steps} forwards over {n_steps} control steps",
          cp.calls == n_steps // n_action_steps, f"{cp.calls} calls, batch sizes {cp.batch_sizes}")
    check("every refill covered all envs at once (queues stay in phase)",
          all(b == n_env for b in cp.batch_sizes), str(cp.batch_sizes))
    check("action shape is (n_env, ACTION_DIM)", acts.shape == (n_steps, n_env, ACTION_DIM),
          str(tuple(acts.shape)))
    check("actions are finite", bool(torch.isfinite(acts).all()))

    # Within one committed window the actions must come from the same prediction: with the
    # queue populated, feeding a WILDLY different observation must not change what is emitted.
    cp2 = CountingPolicy(policy)
    ctrl2 = BatchedACTController(cp2, stats, n_action_steps, chunk_size, "cpu")
    a0 = ctrl2.act(obs)
    a1 = ctrl2.act(obs + 100.0)  # absurd obs, mid-window
    check("mid-window actions ignore new observations (the chunk is committed)",
          cp2.calls == 1, f"{cp2.calls} forwards after 2 steps -- must be 1")
    check("mid-window actions differ from step 0 (queue advances, not repeats)",
          not torch.allclose(a0, a1), f"max abs diff {(a0 - a1).abs().max().item():.3e}")

    # A reset must break commitment immediately -- that is the flush mechanism.
    ctrl2.reset([0, 2])
    ctrl2.act(obs)
    check("reset forces an immediate refill of exactly the reset envs",
          cp2.calls == 2 and cp2.batch_sizes[-1] == 2,
          f"calls {cp2.calls}, last batch {cp2.batch_sizes[-1]}")

    # The gripper channel: demos use exactly +/-1, so a trained policy's output should live
    # near that scale. On an undertrained model this is a sanity range, not a correctness test.
    grip = acts[..., 6]
    print(f"    gripper channel over {n_steps * n_env} emitted actions: "
          f"min {grip.min():.2f}  max {grip.max():.2f}  (demos are exactly +/-1)")


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU validation of the BC pipeline.")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=15)
    ap.add_argument("--steps", type=int, default=20, help="training steps for the smoke run")
    a = ap.parse_args()
    paths = [Path(p) for p in a.data]

    print("\n" + "=" * 78 + "\n  CPU PIPELINE VALIDATION (no simulator)\n" + "=" * 78)
    test_pool_filters(paths)
    test_mask_reaches_loss(a.chunk_size)
    policy, stats, ds = test_train_and_reload(paths, a.chunk_size, a.n_action_steps, a.steps)
    test_chunk_commitment(policy, stats, ds, a.chunk_size, a.n_action_steps)

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 78 + "\n")
        return 1
    print("  ALL CHECKS PASSED -- the non-simulator half of the training path is sound.")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
