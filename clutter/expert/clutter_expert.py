# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""The scripted expert for `Rebot-ClutterExtract-v0`, and the reasons for every choice.

This module is the Stage-1 deliverable. Everything in it was decided by a measurement, and
each measurement is named so a later reader can go and check it rather than trust it.

The recipe
----------
================  ======================  =================================================
opening axis      `o_hat = x_hat`         Fingers straddle the target FORE AND AFT, sitting
                                          at x ~ 205 / 295 mm with |y| < 5 mm. They never
                                          enter the 12 mm row gaps, so the row's y-pitch --
                                          the stated difficulty of the task -- stops being
                                          the binding constraint.                    (P11)
grip height       **z = 0.055 m**         NOT 0.065. The block is 70 mm tall and settles
                                          with its top at ~67 mm. A grip 2 mm below the top
                                          face lets the closing impulse tip it; a grip
                                          23 mm down does not. Paired, one spawn:
                                          65 mm -> 43.0 % topple, 55 mm -> 13.3 %,
                                          50 mm -> 53.1 % (too low: the wrist reaches
                                          z = 15 mm and the grip destabilises).      (P20)
approach axis     UNCONSTRAINED, but      A downward approach axis is genuinely unattainable
                  the WHOLE ARM is        here (`a_des = (0,0,-1)` achieves `a_align =
                  kept out of the row     -0.11`), so the wrist stub -- which trails the TCP
                                          by 41.9 mm along `a_hat` -- must thread a 12 mm
                                          gap. Selecting on TCP error and `o_align` alone
                                          does not see this and will happily put it inside
                                          a neighbour.                          (P13/P14/P15)
pose selection    penetration FIRST,      Lexicographic, never a weighted sum. A pose that
                  then `o_align`,         puts a link inside a block is a different kind of
                  under a 1.5 mm gate     object, not a slightly worse one.            (P15)
path              DENSE CARTESIAN,        The single largest fix in Stage 1. Independently
                  <= 30 mm spacing,       solved waypoints can land in different IK branches;
                  local solves only       the joint-space line between them swept 108 mm off
                                          its Cartesian path and gave the `place` segment a
                                          **100 % contact hazard**. Densifying with
                                          `restarts = 1` and `std0 = 0.08` cut the worst
                                          deviation to 1.1 mm and took success 18 % -> 62.5 %.
                                                                                      (P17)
yaw              MATCHED PER ENV          The reset draws `yaw ~ U(-0.20, 0.20)` rad. A square
                 through the grasp        jaw turns the block 3.7 deg while closing; a matched
                 AND the lift             one turns it 0.28 deg. The lift must carry the same
                                          yaw or it twists the block while still in the row.
                                                                                      (P16)
lift              +150 mm                 The block hangs 68 mm below the TCP when gripped at
                                          65 mm, so the old 75 mm lift left its bottom face
                                          5 mm above the distractor tops, on a path that runs
                                          diagonally over d0 and d1 to the goal.       (P12)
withdraw          VERTICAL AT THE GOAL    The old script drove back across the row with an
                                          empty gripper: 98.4 % contact hazard, for a motion
                                          the task never asked for.                   (P17)
execution         physics-only for        `env.step` runs the MDP, so a topple resets the
                  measurement;            scene and re-spawns the blocks upright -- the
                  `env.step` for demos    evidence erases itself.                       (7.5)
================  ======================  =================================================

Two things this expert deliberately does NOT do
-----------------------------------------------
* **It does not ramp the gripper.** Driving the finger joint target smoothly shut scored
  97.7 % in one paired arm, and it is not a legal action:
  `ActionsCfg.gripper` is a `BinaryJointPositionActionCfg` with no intermediate aperture.
  Demonstrations containing an action a policy cannot emit are worse than useless. The
  legal duty-cycled approximations were measured and are *worse* than a plain binary close
  (29.7 % and 24.2 % against 39.8 %), so the plain close is what ships.            (P19/P20)
* **It does not manipulate the clutter.** P06-P09 established there is no reachable inner
  face to push against, wedging yields 4.2 mm against ~31 mm needed, and every reachable
  push topples. The orthogonal grasp makes all of that unnecessary.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys

import torch
from isaaclab.utils.math import quat_apply

from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes"))
from _kin import ArmKin, Q_CLOSE, Q_OPEN, _t  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
DIST = mdp_cl.DISTRACTOR_NAMES
ROW_Y = (-0.084, -0.042, 0.042, 0.084)

#: Defaults, each traceable to the table above.
GRIP_Z = 0.055
LIFT_DZ = 0.150
STANDOFF = 0.070
PHI = 90.0
SEG = 0.030
MARGIN = 0.005
#: `o_align` below this has never produced a good run; 0.848 and 0.885 both scored under
#: 30 %. Necessary but nowhere near sufficient -- see `SCREEN`.
O_ALIGN_MIN = 0.99
#: Candidate grasp poses to screen IN THE SIMULATOR before committing. **4, not more.**
#: P26, three independent selections x two held-out spawn batches per arm, 768 episodes each:
#:
#:      screen = 0    63.7 %   spread 58.6-68.8 %   sd  4.1 %
#:      screen = 4    73.7 %   spread 67.2-80.5 %   sd  4.8 %    screen scores 79/84/88 %
#:      screen = 8    67.2 %   spread 52.3-78.9 %   sd 10.2 %    screen scores 84/85/91 %
#:
#: More candidates score *better on the screening batch* and *worse on held-out ones*: the
#: optimism gap goes from +9.9 to +19.3 points. With one 128-env batch as the sample, eight
#: candidates is enough for the winner to be one that got lucky. The fix is not fewer
#: candidates but more screening spawns -- see the Stage-1 screening sweep (doc retired 2026-08-03).
SCREEN = 4

#: Hold durations, in **physics** steps. One env step = `decimation` 8 physics steps, so the
#: env-step cost of each is `value / 8`. These were chosen for a physics-only measurement,
#: where duration is nearly free -- and the whole schedule totals 382 env steps of which
#: **180 (47.1 %) are a held pose**.
#:
#: That is a problem for behaviour cloning and not for the expert. During a hold the 42-D
#: observation is constant to numerical noise while the correct action chunk differs at every
#: step, because the next motion begins at a fixed offset from the hold's *start*. A memoryless
#: chunk policy cannot recover phase from a constant observation; it can only learn the
#: distribution over "how many steps until the move", and at inference it draws one sample from
#: it. See `REFERENCE.md` §6 for the N2 chunk-ambiguity risk (the P27 measurement was retired
#: with the 2026-08-03 reset: it was scored on the topple-only predicate).
HOLDS = {"settle": 80, "predwell": 160, "close": 560, "dwell": 160,
         "release": 240, "final": 240}


@contextlib.contextmanager
def _rng_seeded(seed: int | None, dev):
    """Seed torch for a block, then put the RNG back exactly as it was.

    `seed=None` is a no-op, so Stage-1 behaviour is untouched. The restore matters: probes
    draw spawn batches from the same generator, and a chain solve that silently advanced it
    would make the batches depend on how the expert was configured.
    """
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class ClutterExpert:
    """Solves one nominal chain, then adapts it per env at reset time.

    The split matters: the CEM is a global search and is far too slow to run per env, but a
    damped-least-squares `refine` around a CEM-verified pose is a handful of batched FK
    calls and handles the whole spawn distribution (target x +/- 12 mm and yaw +/- 0.20 rad,
    distractors +/- 10 / +/- 5 mm). Nothing is ever trusted as commanded -- every adapted
    pose is re-read from the sim before it is used.
    """

    def __init__(self, env, grip_z: float = GRIP_Z, lift_dz: float = LIFT_DZ,
                 phi: float = PHI, seg: float = SEG, margin: float = MARGIN,
                 iters: int = 60, restarts: int = 12, tries: int = 10,
                 o_min: float = O_ALIGN_MIN, verbose: bool = True,
                 plan_full: bool = True, wrist_min_z: float = 0.0,
                 screen: int = SCREEN, holds: dict | None = None,
                 wrist_side: int = 0, roll_mode: str = "rule",
                 pose_q=None, chain_seed: int | None = None, chain=None, approach=None):
        self.env, self.e = env, env.unwrapped
        self.K = ArmKin(env)
        self.dev, self.n = self.K.dev, self.K.n
        self.grip_z, self.lift_dz, self.seg = grip_z, lift_dz, seg
        self.margin, self.iters, self.restarts = margin, iters, restarts
        self.tries, self.o_min, self.verbose = tries, o_min, verbose
        #: `plan_full = False` solves the grasp pose only and skips the dense chain -- about
        #: a tenth of the cost. Used by probes that study the close in isolation and never
        #: travel, so a height or `phi` sweep can afford many more cells and pose draws.
        self.plan_full = plan_full
        #: Require the wrist stub to sit at least this high, i.e. ABOVE the row rather than
        #: threading a 12 mm gap. P23's grip-height sweep found the effect by accident: at
        #: grip_z = 80 mm two of three pose draws placed `gripper_end` at z = 72-73 mm,
        #: clear of the 67 mm block tops, and scored **3.9 % and 2.3 %** isolated-close
        #: topple. The third draw kept it in the gap at z = 44 mm and scored **27.3 %**.
        #: Grip height itself is nearly flat across 40-80 mm (19.5-31.5 %); the wrist's
        #: height is what actually separates the cells, and unlike grip height it can be
        #: SELECTED. Set to ~0.070 to demand clearance; 0 disables the gate.
        self.wrist_min_z = wrist_min_z
        #: Number of candidate grasp poses to SCREEN IN THE SIMULATOR before committing.
        #:
        #: The pose draw is the largest remaining source of variance -- 32.8 % to 74.2 %
        #: success between draws of the same configuration (P25 arm A, sd 17.5 %) -- and no
        #: forward-kinematic statistic tried so far predicts it. `o_align`, keep-out
        #: penetration and wrist height are all near-identical across a good draw and a bad
        #: one. So stop guessing: run each candidate's close in the sim and keep the one that
        #: measurably works.
        #:
        #: The score is `enclosed AND NOT toppled`, and the conjunction is the point. P25
        #: scored candidates on disturbance alone and selected poses that disturb nothing
        #: because they *grasp* nothing -- wrist above the row, fingers laid flat, enclosure
        #: 19-27 %, isolated topple 2-4 %, end-to-end success **2.0 %**. A proxy that omits
        #: the success condition will find the poses that fail it.
        self.screen = screen
        #: hold durations in physics steps; see `HOLDS`. Overriding these changes the
        #: manoeuvre, so any override must be measured paired against the defaults.
        self.holds = dict(HOLDS)
        self.holds.update(holds or {})
        #: which 12 mm row gap the wrist stub must thread: 0 = either, +1 = the +y gap,
        #: -1 = the -y gap. P28 found the search picks one at random (5 of 8 draws went +y),
        #: producing mirror-image joint trajectories for near-identical observations. Harmless
        #: for the expert, a defect in the data-generating process for BC. See `_gated_solve`.
        self.wrist_side = int(wrist_side)
        #: how the wrist roll -- a free binary worth ~7 points and invisible to forward
        #: kinematics -- is chosen. `"rule"` applies the measured 12/12 rule at no cost;
        #: `"screen"` tries both in the sim, which doubles the candidate pool and was
        #: measured WORSE; `"off"` is Stage-1 behaviour (random). See `_canon_roll`.  (P30)
        assert roll_mode in ("rule", "screen", "off"), roll_mode
        self.roll_mode = roll_mode
        #: a FIXED grasp pose (6 joint angles), bypassing the CEM entirely.
        #:
        #: P32 measured the intraclass correlation of pose quality at **0.82** -- success is
        #: a stable property of the pose, reproducing on an independent spawn batch to within
        #: ~6 points, of which 4.4 is the binomial floor -- while a true pose-to-pose sd of
        #: **12.0 points** separates draws that every forward statistic calls identical.
        #: P33 therefore stopped predicting and ran a tournament: 8 candidates on shared
        #: batches, winner re-measured on 512 FRESH episodes at **75.4 %** against a 56.6 %
        #: candidate mean.
        #:
        #: Passing that winner here is what makes the data-generating process for Stage 2
        #: both good and DETERMINISTIC. Re-solving per run would re-draw from a 12-point-sd
        #: distribution, which is how every P26 number acquired its +/-12-point interval.
        #: `expert/pose_p33.json` holds the vector; `screen`, `wrist_side` and `roll_mode`
        #: are all ignored when this is set, because the pose is already chosen.
        self.pose_q = pose_q
        #: seed for the dense-chain CEM, or None for Stage-1 behaviour (unseeded).
        #:
        #: **This narrows the spread; it does NOT make the chain reproducible.** P34 measured
        #: two experts built with the SAME `chain_seed` and got `max |dq| = 0.109 rad` over
        #: the 23 waypoints, with 39 episode outcomes flipping. The cause is not a missed RNG
        #: source: `_dense` runs 22 waypoints x 60 CEM iterations, and each iteration reduces
        #: with `elite.mean(0)` / `elite.std(0)` on the GPU, whose summation order is not
        #: bit-stable. 1e-7 differences compound over 1 320 iterations into a different local
        #: optimum. Use `chain=` instead when reproducibility is required.
        self.chain_seed = chain_seed
        #: a FULLY FROZEN manoeuvre: `{"pts": [[x,y,z], ...], "qs": [[6], ...],
        #: "i_grip": int}`. Bypasses `_dense` entirely, so there is no CEM and nothing to be
        #: non-deterministic. This is the only configuration that reproduces a measured
        #: number exactly, and therefore the only one Stage 2 should generate demos from.
        self.chain = chain
        #: the FROZEN `home -> chain[0]` approach, `{"pts": [...], "qs": [...]}`.
        #:
        #: `run_physics` teleports to `chain[0]`, so Stage 0 and Stage 1 never executed this
        #: segment at all. `env.step` cannot teleport, and P29 measured what happens when it
        #: is improvised: a joint-space lerp drives `gripper_left` 4.85 mm into
        #: `distractor_1` and scores **5.5 %**, while a forward dense-Cartesian solve is
        #: geometrically spotless and scores **0.0 %** -- it ends 1.90 rad from `qs[0]` on
        #: `joint6`, a different IK branch at the same TCP, and the schedule's first hold
        #: commands that as a step change above the row (100 % topple, 96 % still delivered).
        #:
        #: The surviving path is solved BACKWARD from `qs[0]` and reversed, so the seam is the
        #: identity: **73.0 % against a 74.2 % teleport baseline**, approach hazard 0 %. It is
        #: frozen for the same reason the chain is -- `_dense` runs a CEM (P34).
        self.approach = approach
        self.goal = torch.tensor(mdp_cl.GOAL_XY, device=self.dev)

        p = math.radians(phi)
        self.o_nom = torch.tensor([math.sin(p), math.cos(p), 0.0], device=self.dev)
        #: an orthogonal jaw closes on the block's 36 mm x-faces, not its 30 mm y-faces
        self.width = 2 * HX if phi >= 45.0 else 2 * HY
        #: keep-out volumes: the four neighbours AND the target. The target belongs here --
        #: at the grasp the fingers straddle it in x (205 / 295 mm) and are outside its
        #: 232-268 mm span, so nothing legitimate is inside it.
        self.boxes = torch.tensor([[ROW_X, y, 0.032, HX, HY, HZ] for y in ROW_Y]
                                  + [[ROW_X, 0.0, 0.032, HX, HY, HZ]], device=self.dev)
        self.plan()

    # ------------------------------------------------------------------ nominal planning
    def _solve(self, pos, seed, tries, std0, restarts, pos_max=0.0015):
        """Penetration first, then alignment, under a hard position gate."""
        best = None
        for _ in range(tries):
            c = self.K.cem(pos, seed, o_des=self.o_nom, w_o=0.60, iters=self.iters,
                           std0=std0, restarts=restarts, avoid=self.boxes,
                           avoid_margin=self.margin)
            if c["pos_err"] > pos_max:
                continue
            key = (round(c["pen"], 4), -c["o_align"])
            if best is None or key < (round(best["pen"], 4), -best["o_align"]):
                best = c
        return best or c

    def _dense(self, pts, seed):
        """Cartesian polyline at <= `seg` spacing, refined locally so the branch is fixed.

        `restarts = 1` and a small `std0` are the entire point. Restarting from uniform
        draws is what lets adjacent waypoints land in different IK branches, and the
        joint-space line between two branches is the 108 mm arc P17 caught.
        """
        qs, out, qq = [], [], seed
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            k = max(1, int(math.ceil(float((b - a).norm()) / self.seg)))
            for j in range(k):
                t = a + (b - a) * (j + 1) / k
                qq = self._solve(t, qq, 2, 0.08, 1)["q"]
                qs.append(qq)
                out.append(t)
        return [pts[0]] + out, [seed] + qs

    def _score_q(self, q, pos):
        """Rebuild a candidate dict for a GIVEN joint vector -- mirrors `ArmKin.cem`'s return.

        Everything is read back from forward kinematics against the current scene, so a
        loaded pose is reported on exactly the same footing as a solved one: the same
        `o_align`, the same keep-out penetration against the same boxes, the same gate.
        No solver runs.
        """
        q = torch.as_tensor(q, dtype=torch.float32, device=self.dev).reshape(6)
        g = self.K.fk(q.unsqueeze(0).repeat(self.n, 1))
        c = {"q": q, "cost": float("nan"),
             "pos_err": float((g["tcp"][0] - pos).norm()),
             "tcp": g["tcp"][0].clone(),
             "a_hat": g["a_hat"][0].clone(),
             "o_hat": g["o_hat"][0].clone(),
             "low_z": float(g["low_z"][0]),
             "pen": float(self.K.box_penetration(g["bodies"][:1], self.boxes, self.margin)[0]),
             "roll": 0}
        c["o_align"] = float((c["o_hat"] @ self.o_nom).abs())
        c["a_align"] = float("nan")
        c["wrist"] = g["bodies"][0, self.K.i_end].clone()
        c["wrist_z"] = float(c["wrist"][2])
        return c

    def _wrist(self, q):
        """`gripper_end` origin in the env frame for a nominal joint vector."""
        return self.K.fk(q.unsqueeze(0).repeat(self.n, 1))["bodies"][0, self.K.i_end].clone()

    def _rolls(self, c):
        """`c`, plus the same pose with `joint6` rolled by +/- pi -- a SECOND candidate.

        A parallel jaw is symmetric under a pi rotation about its approach axis, and every
        statistic this class uses to choose a pose is **provably** blind to the difference:
        P30 measured TCP agreement to **0.00 mm**, both axes to **1.00000**, identical
        keep-out penetration, and each finger origin landing exactly where the other one was
        (0.00 mm). The CEM cannot see it either -- its orientation term is `|o_hat . o_des|`,
        deliberately sign-free, and `box_penetration` takes a max over body origins, so
        swapping which finger is where changes nothing.

        And yet the two are **7.0 points apart end to end** (69.3 % vs 76.3 % pooled over 384
        paired episodes, +10.2 / +6.2 / +4.7 on three independent draws), because the collision
        MESHES rotate with the bodies even though the origins merely swap. The whole difference
        lands in topple -- enclosure is 100 % in both arms -- which is the residual failure mode
        P22 isolated and the first thing since screening to move it.

        So this is a free binary variable worth as much as half of what simulator screening
        itself bought, and `plan()` assigned it at random from Stage 0 until this was found.
        It is SCREENED rather than fixed to a sign: the preference is expected to be systematic
        within a wrist-side family and to invert for the mirrored family, so a hardcoded sign
        would be right half the time and silently wrong the other half.               (P30)
        """
        out = [dict(c, roll=0)]
        for s in (+1.0, -1.0):
            v = float(c["q"][5]) + s * math.pi
            if float(self.K.lo[5]) <= v <= float(self.K.hi[5]):
                q = c["q"].clone()
                q[5] = v
                out.append(dict(c, q=q, roll=int(s)))
                break
        return out

    def _canon_roll(self, c):
        """Pick the wrist roll by rule instead of by screening it.  **12 of 12.**

        P30 established that the roll is worth ~7 points and is invisible to forward
        kinematics. The obvious response -- screen both -- was implemented, measured, and is
        WRONG: it doubles the candidate pool from 4 to 8, and `p26_screen_v2` came back at
        **66.5 %, sd 11.9 %, optimism +18.7** against `screen = 4`'s **73.7 %, sd 4.8 %,
        optimism +9.9**. That is `screen = 8`'s signature to the point (67.2 %, sd 10.2 %,
        +19.3), and `HANDOFF.md` §10.2 had already warned that the candidate count cannot rise
        until the selection statistic stops being noise-dominated. It rose anyway.

        But the 12 roll pairs that run produced are not noise. Writing `w_y` for the wrist's
        y-coordinate and comparing each pair's screen scores:

            w_y      j6      score  |   j6      score  |  winner    margin
           +20.8   +1.202   70.3 %  |  -1.939   60.9 %  |  +1.202    +9.4
           -20.1   +1.912   80.5 %  |  -1.230   89.1 %  |  -1.230    +8.6
           +21.5   +1.217   79.7 %  |  -1.925   71.1 %  |  +1.217    +8.6
           +20.1   +1.155   74.2 %  |  -1.987   64.1 %  |  +1.155   +10.1
           +20.3   -1.844   81.2 %  |  +1.297   88.3 %  |  +1.297    +7.1
           -21.1   -1.185   74.2 %  |  +1.957   64.8 %  |  -1.185    +9.4
           -20.5   +1.947   77.3 %  |  -1.194   87.5 %  |  -1.194   +10.2
           -21.2   -1.215   83.6 %  |  +1.926   80.5 %  |  -1.215    +3.1
           -20.9   -1.244   78.1 %  |  +1.898   70.3 %  |  -1.244    +7.8
           +20.5   -1.942   64.1 %  |  +1.199   70.3 %  |  +1.199    +6.2
           -20.7   -1.217   65.6 %  |  +1.925   64.8 %  |  -1.217    +0.8
           -21.6   -1.061   65.6 %  |  +2.081   64.1 %  |  -1.061    +1.5

        **Every winner has `sign(j6) == sign(w_y)`. Twelve out of twelve** (p = 2^-12), mean
        margin **+6.9 points** -- which is P30's independently-measured +7.0 to within 0.1.

        So the roll does not need screening. It needs a rule, and the rule costs nothing:
        the candidate pool stays at the validated 4.

        **The confound, stated.** Every winner ALSO has `|j6| < pi/2` (1.06-1.30) while every
        loser has `|j6| > pi/2` (1.84-2.08). The two descriptions are perfectly correlated in
        this data and cannot be separated by it. The sign rule ships because it has a mechanism
        -- the blades are asymmetric about the roll axis, so which way their tails point
        relative to the neighbours depends on which 12 mm gap the wrist is in -- and `|j6|`
        does not. `roll="screen"` remains available and is what would settle it.
        """
        w_y = float(c["wrist"][1])
        for v in self._rolls(c):
            if float(v["q"][5]) * w_y > 0.0:
                return v
        return dict(c, roll=0)

    def _screen(self, grip, incumbent):
        """Try `self.screen` candidate grasp poses in the sim; keep the one that works.

        Each candidate is screened in BOTH wrist rolls (`_rolls`), so a screen of K costs up
        to 2K trials. Candidates are gated on `o_align` before they are tried at all -- see
        `plan()` for why that gate used to leak.

        Requires the scene to be in a settled spawn state on entry -- the caller resets and
        settles before constructing the expert, which every probe already does. The snapshot
        is restored after each candidate and again at the end, so screening leaves the scene
        exactly as it found it and the chain solve that follows is unaffected.

        Screening happens on ONE spawn batch; evaluation must use different ones, or the
        number is just the training score.
        """
        K, e, n = self.K, self.e, self.n
        snap = {k: _t(e.scene[k].data.root_state_w).clone()
                for k in ("target",) + DIST}

        def restore():
            for k, v in snap.items():
                e.scene[k].write_root_state_to_sim(v.clone())
            e.sim.forward()
            e.scene.update(e.physics_dt)

        def trial(c):
            restore()
            q = K.refine(c["q"].unsqueeze(0).repeat(n, 1), self._grip_targets(),
                         iters=4, o_des=self.target_axis())
            K.teleport_arm(q, Q_OPEN)
            for qf, steps in ((Q_OPEN, 160), (Q_CLOSE, 560)):
                qd = K._drive(q, qf)
                for _ in range(steps):
                    K.robot.set_joint_position_target(qd)
                    K.robot.write_data_to_sim()
                    e.sim.step()
                    e.scene.update(e.physics_dt)
            up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
            held = (K.gap() - self.width).abs() < 0.012
            ok = held & ~(up < mdp_cl.TOPPLE_DOT).any(dim=1)
            return float(ok.float().mean()), float(held.float().mean())

        best, best_s = None, -1.0
        for k in range(self.screen):
            # attempts = 4, not plan()'s 8: `_gated_solve` returns on the first pose clearing
            # every gate, and 6 of P28's 8 draws cleared `o_align` immediately, so a smaller
            # budget costs almost nothing and bounds the worst case at 4 x `tries` CEM runs.
            c = incumbent if k == 0 else self._gated_solve(grip, attempts=4)
            if "wrist" not in c:
                c["wrist"] = self._wrist(c["q"])
            # THIRD leak, found by p26-v3 and NOT YET RE-MEASURED. `_gated_solve` returns its
            # best-alignment attempt when none clears the gate, so a sub-gate candidate can
            # still enter the screen -- and then WIN on screen score. In v3 exactly one of
            # three selections did (o_align 0.9788 against a 0.99 floor, screen 87.5 %) and it
            # was the worst cell of the run by 7 points: 66.8 % against 83.2 % and 73.8 %.
            # Skip it unless nothing else is available.
            if c["o_align"] < self.o_min and best is not None:
                continue
            # `roll="rule"` leaves the candidate pool at the validated 4; `roll="screen"`
            # doubles it and was measured WORSE (66.5 % vs 73.7 %) -- see `_canon_roll`.
            for v in ([c] if self.roll_mode == "off" else
                      self._rolls(c) if self.roll_mode == "screen" else
                      [self._canon_roll(c)]):
                s, h = trial(v)
                if self.verbose:
                    print(f"   [screen] cand {k} roll {v['roll']:+d}: o_align "
                          f"{v['o_align']:.4f} | wrist ({v['wrist'][1] * 1000:+5.1f},"
                          f"{v['wrist'][2] * 1000:5.1f}) mm | j6 {float(v['q'][5]):+.3f} | "
                          f"enclosed {h:6.1%} | enclosed&clean {s:6.1%}")
                if s > best_s:
                    best, best_s = v, s
        best = best if best is not None else incumbent
        best["screen_score"] = best_s
        restore()
        return best

    def _grip_targets(self):
        """Per-env TCP target at the grasp: where the target block actually is."""
        org = self.e.scene.env_origins
        tp = _t(self.e.scene["target"].data.root_pos_w) - org
        return torch.stack([tp[:, 0], tp[:, 1],
                            torch.full((self.n,), self.grip_z, device=self.dev)], dim=1)

    def _gated_solve(self, grip, attempts: int = 8):
        """Solve the grasp pose under the gates the recipe actually claims.

        Until P28 this gate LEAKED, in two independent places, and two of eight solved poses
        shipped with `o_align` 0.863 / 0.866 against an advertised floor of 0.99:

        * `plan()`'s fallback kept the highest-WRIST attempt when no attempt cleared the gate,
          which says nothing about alignment;
        * `_screen` generated its other `screen - 1` candidates with a bare `_solve` call that
          gates only on position error.

        Both leaks fed the same family: a more extended arm (`j2` -1.28 against -1.72), a higher
        wrist (25 mm against 18), poor alignment -- and the two HIGHEST screen scores of the run
        (98.4 %, 85.9 %). That is the P25 family, measured end to end at 2.0 % success. The
        screen does not catch it because its enclosure test is `|gap - 36 mm| < 12 mm`, so any
        stall from 24 to 48 mm counts as held, and a jaw 30 deg off the faces closing on a
        corner sits comfortably inside that window.

        `o_align` is NECESSARY and nowhere near sufficient -- that has been the documented
        position since P18 and it survives P28 intact. The simulator screen stays the decider;
        it simply stops being offered candidates the recipe already excludes.       (P28)

        `wrist_side` additionally locks which 12 mm gap the wrist threads. Over eight
        independent solves it went +y five times and -y three, with nothing in the search
        preferring either -- so `HANDOFF.md`'s claim that it is "put there deliberately" was
        wrong: it is DRAWN, and the keep-out term merely verifies whichever gap it lands in.
        Harmless for the expert; not harmless for BC, where the two families are mirror images
        producing opposite joint signs for observations that differ only by the row's own
        +/-5 mm of jitter.
        """
        best = None
        for _ in range(attempts):
            c = self._solve(grip, self.K.q_arm0, self.tries, 0.6, self.restarts)
            c["wrist"] = self._wrist(c["q"])
            c["wrist_z"] = float(c["wrist"][2])
            if self.roll_mode == "rule":
                c = self._canon_roll(c)      # free; see `_canon_roll`
            side_ok = (self.wrist_side == 0
                       or float(c["wrist"][1]) * self.wrist_side > 0)
            if (best is None or c["o_align"] > best["o_align"]) and side_ok:
                best = c
            if (c["o_align"] >= self.o_min and c["pen"] <= 1e-4 and side_ok
                    and c["wrist_z"] >= self.wrist_min_z):
                return c
        # No attempt cleared every gate. Return the best ALIGNMENT seen on the right side --
        # never the highest wrist, which is what used to select the P25 family.
        return best if best is not None else c

    def plan(self):
        """Solve the nominal chain. Retries the grasp pose until it clears the gates."""
        dev = self.dev
        self.approach_qs = ([torch.tensor(q, dtype=torch.float32, device=dev)
                             for q in self.approach["qs"]] if self.approach else None)
        grip = torch.tensor([ROW_X, 0.0, self.grip_z], device=dev)
        if self.pose_q is not None:
            c0 = self._score_q(self.pose_q, grip)
        else:
            c0 = self._gated_solve(grip)
            if self.screen:
                c0 = self._screen(grip, c0)
        self.pose = c0
        self.gates_met = bool(c0["o_align"] >= self.o_min and c0["pen"] <= 1e-4)
        self.wrist_cleared = c0["wrist_z"] >= self.wrist_min_z if self.wrist_min_z else None
        if not self.plan_full:
            self.pts, self.qs, self.i_grip = [grip], [c0["q"]], 0
            return self
        if self.chain is not None:
            self.pts = [torch.tensor(p, dtype=torch.float32, device=dev)
                        for p in self.chain["pts"]]
            self.qs = [torch.tensor(q, dtype=torch.float32, device=dev)
                       for q in self.chain["qs"]]
            self.i_grip = int(self.chain["i_grip"])
            if self.verbose:
                print(f"   [expert] chain LOADED: {len(self.qs)} waypoints, grasp at index "
                      f"{self.i_grip} (no CEM ran)")
            return self
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        lift = grip + self.lift_dz * up
        carry = torch.tensor([self.goal[0], self.goal[1], self.grip_z + self.lift_dz],
                             device=dev)
        place = torch.tensor([self.goal[0], self.goal[1], self.grip_z], device=dev)
        appr = [grip + STANDOFF * up * ((3 - k) / 3.0) for k in range(3)]

        # `_dense` runs a CEM per waypoint (`_solve(..., restarts=1, std0=0.08)`), so the 22
        # non-grasp waypoints are DRAWN, not derived -- freezing the grasp pose does not
        # freeze the chain. P34 caught this: replaying P33's own verification seeds with the
        # identical grasp pose returned 74.2 % against P33's 75.4 %, a 1.2-point gap that had
        # nowhere else to come from. Seeding here makes the whole chain reproducible; the
        # global RNG state is saved and restored so callers see no side effect.
        with _rng_seeded(self.chain_seed, self.dev):
            up_pts, up_qs = self._dense([grip, lift, carry, place], c0["q"])
            dn_pts, dn_qs = self._dense([grip, appr[2], appr[1], appr[0]], c0["q"])
        self.pts = list(reversed(dn_pts)) + up_pts[1:]
        self.qs = list(reversed(dn_qs)) + up_qs[1:]
        self.i_grip = len(dn_pts) - 1
        if self.verbose:
            w = self.K.fk(c0["q"].unsqueeze(0).repeat(self.n, 1))["bodies"][0, self.K.i_end]
            print(f"   [expert] grasp: err {c0['pos_err'] * 1000:.2f} mm | o_align "
                  f"{c0['o_align']:.4f} | pen {c0['pen'] * 1000:.1f} mm | wrist "
                  f"({float(w[0]) * 1000:.0f},{float(w[1]) * 1000:+.0f},"
                  f"{float(w[2]) * 1000:.0f}) mm | j6 {float(c0['q'][5]):+.3f} "
                  f"roll {c0.get('roll', 0):+d}"
                  + ("" if self.gates_met else "  <-- GATES NOT MET")
                  + (f" | wrist gate {'MET' if self.wrist_cleared else 'MISSED'}"
                     if self.wrist_min_z else ""))
            print(f"   [expert] chain: {len(self.qs)} waypoints, grasp at index "
                  f"{self.i_grip}, lift +{self.lift_dz * 1000:.0f} mm")
        return self

    # ------------------------------------------------------------------ per-env adaptation
    def target_axis(self, gain: float = 1.0) -> torch.Tensor:
        """Direction the jaw should open along, `gain` of the way to the block's own axis.

        `gain = 1` meets the 36 mm faces squarely and is right for the *grip*. It may be
        wrong for the *close*, and the blade geometry says why. Perpendicular to the opening
        axis the blade is only +/-19.2 mm, against neighbour faces at +/-27 mm -- 7.8 mm of
        margin. But the blade reaches ~47 mm ALONG the opening axis, and rotating that axis
        by the full 11.4 deg of spawn yaw swings its corner by

            47 * sin(11.4 deg) = 9.3 mm   >   7.8 mm of margin

        so a fully yaw-matched jaw puts its own corners into the neighbours. P16 measured
        exactly that signature and it was not recognised at the time: matching yaw raised the
        close-phase contact count from 65/128 to 93/128 while improving every grip statistic.
        P19 then found the square jaw outscoring the matched one end to end (69.5 % vs
        64.1 %) and it was written off as noise.

        `gain` makes the trade explicit and sweepable rather than assumed.
        """
        q = _t(self.e.scene["target"].data.root_quat_w)
        ax = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device)
                        .expand(q.shape[0], 3)).clone()
        ax[:, 2] = 0.0
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9)
        if gain >= 1.0:
            return ax
        # rotate the nominal opening axis `gain` of the way toward the block's, sign-free
        a_nom = torch.atan2(self.o_nom[1], self.o_nom[0])
        a_tgt = torch.atan2(ax[:, 1], ax[:, 0])
        d = torch.atan2(torch.sin(a_tgt - a_nom), torch.cos(a_tgt - a_nom))
        d = torch.where(d > math.pi / 2, d - math.pi, d)
        d = torch.where(d < -math.pi / 2, d + math.pi, d)
        a = a_nom + gain * d
        return torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], dim=1)

    def adapt(self, match_yaw: bool = True, yaw_gain: float = 1.0) -> list[torch.Tensor]:
        """Bind the nominal chain to where the blocks actually are, for the current reset."""
        org = self.e.scene.env_origins
        tp = _t(self.e.scene["target"].data.root_pos_w) - org
        o_env = self.target_axis(yaw_gain) if match_yaw else None
        shift = torch.stack([tp[:, 0] - ROW_X, tp[:, 1],
                             torch.zeros(self.n, device=self.dev)], dim=1)
        out = []
        for i, (p, q) in enumerate(zip(self.pts, self.qs)):
            # the pre-grasp half tracks the target; the goal half is absolute. The lift
            # (i_grip + 1) keeps the matched yaw -- squaring the wrist while the block is
            # still between its neighbours twists it into them.
            s = shift if i <= self.i_grip else torch.zeros_like(shift)
            od = o_env if (o_env is not None and i <= self.i_grip + 1) else None
            out.append(self.K.refine(q.unsqueeze(0).repeat(self.n, 1),
                                     p.unsqueeze(0) + s, iters=3, o_des=od))
        return out

    # --------------------------------------------------------------------- schedule
    def schedule(self, chain: list[torch.Tensor]) -> list[tuple]:
        """The trajectory as `(phase, kind, args)`, so callers can execute it either way.

        `kind` is `hold` or `move`; `close` is the binary gripper state. This is the single
        source of truth for the motion -- the physics-only evaluator and the `env.step`
        demo recorder both consume it, which is what makes their equivalence checkable.
        """
        g, m = self.i_grip, len(chain) - 1
        h = self.holds
        st_in = max(6, int(round(75 / max(1, g))))
        st_out = max(6, int(round(90 / max(1, m - g))))
        sched = [("settle", "hold", chain[0], h["settle"], False)]
        sched += [("descend", "move", chain[i], chain[i + 1], st_in, False) for i in range(g)]
        sched += [("predwell", "hold", chain[g], h["predwell"], False),
                  ("close", "hold", chain[g], h["close"], True)]
        sched += [("carry", "move", chain[i], chain[i + 1], st_out, True)
                  for i in range(g, m)]
        sched += [("dwell", "hold", chain[m], h["dwell"], True),
                  ("release", "hold", chain[m], h["release"], False),
                  ("withdraw", "move", chain[m], chain[m - 1], 25, False),
                  ("final", "hold", chain[m - 1], h["final"], False)]
        return sched

    def env_steps(self, chain: list[torch.Tensor]) -> dict:
        """Length of the schedule in **env steps**, per phase and total.

        `hold` args are physics steps (`/ 8`); `move` args are already env steps, since
        `_move` runs `substeps = 8` physics steps per control step. Needed to size the demo
        horizon and to see how much of a demo is a held pose.
        """
        out, tot = {}, 0
        for item in self.schedule(chain):
            k = item[3] // 8 if item[1] == "hold" else item[4]
            out[item[0]] = out.get(item[0], 0) + k
            tot += k
        out["TOTAL"] = tot
        out["STATIC"] = sum(v for p, v in out.items()
                            if p in self.holds)
        return out

    # --------------------------------------------------------------------- scoring
    def score(self, gap_stall: torch.Tensor) -> dict:
        """Score with the env's own predicates, read at the end rather than enforced."""
        org = self.e.scene.env_origins
        b = _t(self.e.scene["target"].data.root_pos_w) - org
        up = torch.stack([mdp_cl._up_z(self.e, d) for d in DIST], dim=1)
        topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)
        held = (gap_stall - self.width).abs() < 0.012
        at_goal = (((b[:, :2] - self.goal).norm(dim=1) < mdp_cl.GOAL_RADIUS)
                   & (b[:, 2] < 0.055))
        return {"held": held, "at_goal": at_goal, "topple": topp,
                "success": at_goal & ~topp}

    # --------------------------------------------------------------------- execution
    def run_physics(self, chain: list[torch.Tensor], tape=None) -> dict:
        """Execute physics-only. Use for MEASUREMENT: `env.step` resets on topple."""
        K = self.K
        gap_stall = None
        K.teleport_arm(chain[0], Q_OPEN)
        for item in self.schedule(chain):
            phase, kind = item[0], item[1]
            if tape is not None:
                tape.label = phase
            qf = Q_CLOSE if item[-1] else Q_OPEN
            if kind == "hold":
                _hold(K, item[2], item[3], qf, tape)
            else:
                _move(K, item[2], item[3], item[4], qf, tape)
            if phase == "close":
                gap_stall = K.gap().clone()
        return self.score(gap_stall)


def _hold(K, q, steps, qf, tape, every=8):
    qd = K._drive(q, qf)
    for s in range(steps):
        K.robot.set_joint_position_target(qd)
        K.robot.write_data_to_sim()
        K.e.sim.step()
        K.e.scene.update(K.e.physics_dt)
        if tape is not None and (s + 1) % every == 0:
            tape.sample()


def _move(K, a, b, steps, qf, tape, substeps=8):
    for s in range(steps):
        f = (s + 1) / steps
        qd = K._drive((1 - f) * a + f * b, qf)
        for _ in range(substeps):
            K.robot.set_joint_position_target(qd)
            K.robot.write_data_to_sim()
            K.e.sim.step()
            K.e.scene.update(K.e.physics_dt)
        if tape is not None:
            tape.sample()
