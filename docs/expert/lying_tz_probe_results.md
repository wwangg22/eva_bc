# lying-can tz feasibility probe (v6 failing states, offline cuRobo)

Grasp-plan result per (failing state, close-height tz), tz passed directly to
`table_candidates`/`plan_grasp` (runner formula bypassed). World = both cans on
the table, none held; start q6 = recorded joints at the failing attempt.

OK = plan success (goalset idx in parens), TF = trajopt-fail, NC = no-candidates.

| case | tz=0.012 | tz=0.014 | tz=0.016 | tz=0.018 | tz=0.020 | tz=0.022 | tz=0.024 | tz=0.026 | tz=0.028 |
|---|---|---|---|---|---|---|---|---|---|
| demo_6/object_a | OK (9) | OK (0) | OK (0) | OK (0) | OK (9) | OK (5) | OK (9) | OK (5) | OK (7) |
| demo_7/object_a | OK (12) | OK (15) | OK (13) | OK (13) | OK (10) | OK (5) | OK (9) | OK (6) | OK (14) |
| demo_19/object_a | OK (10) | OK (12) | OK (8) | OK (0) | OK (10) | OK (0) | OK (9) | OK (9) | OK (3) |
| demo_19/object_b | OK (14) | OK (15) | OK (12) | OK (3) | OK (13) | OK (15) | OK (6) | OK (14) | OK (12) |
| demo_28/object_b | OK (6) | OK (5) | OK (6) | OK (12) | OK (6) | OK (12) | OK (14) | OK (4) | OK (3) |
| demo_24/object_b | NC | NC | OK (0) | OK (1) | OK (14) | OK (14) | OK (11) | OK (11) | OK (10) |

Candidate counts (unique poses in the K=16 goalset):

- demo_6/object_a: 16, 16, 16, 16, 16, 16, 16, 16, 16
- demo_7/object_a: 16, 16, 16, 16, 16, 16, 16, 16, 16
- demo_19/object_a: 16, 16, 16, 16, 16, 16, 16, 16, 16
- demo_19/object_b: 16, 16, 16, 16, 16, 16, 16, 16, 16
- demo_28/object_b: 16, 16, 16, 16, 16, 16, 16, 16, 16
- demo_24/object_b: 0, 0, 1, 2, 16, 16, 16, 16, 16

## Decoded failing states (from demos_smoke_32ep.h5)

| case | t | xy | r | z | axis (world) | log xy err | q6 |
|---|---|---|---|---|---|---|---|
| demo_6/object_a | 219 | (+0.242,-0.068) | 0.251 | 0.008 | (+0.60,+0.80,-0.03) | 0.4 mm | [-0.822, -0.903, -0.284, 0.218, -0.776, -0.389] |
| demo_7/object_a | 353 | (+0.239,+0.153) | 0.284 | 0.008 | (+0.23,+0.97,-0.01) | 0.3 mm | [0.193, -1.104, -0.193, -0.228, -0.74, -0.502] |
| demo_19/object_a | 10 | (+0.258,-0.105) | 0.278 | 0.008 | (+0.50,+0.87,-0.01) | 0.6 mm | [0.0, -1.353, -0.299, -0.85, 0.0, -0.0] |
| demo_19/object_b | 353 | (+0.237,+0.187) | 0.302 | 0.008 | (+0.16,-0.99,-0.03) | - | [-0.657, -0.926, -0.144, -0.103, -0.669, -0.426] |
| demo_28/object_b | 10 | (+0.230,+0.104) | 0.252 | 0.015 | (+0.02,+0.02,+1.00) | - | [0.0, -1.353, -0.299, -0.85, 0.0, -0.0] |
| demo_24/object_b | 10 | (+0.167,-0.142) | 0.219 | 0.008 | (-0.43,+0.90,-0.02) | 0.5 mm | [0.0, -1.353, -0.299, -0.85, 0.0, -0.0] |

basket centers: demo_6/object_a (0.178,0.119); demo_7/object_a (0.225,0.009); demo_19/object_a (0.262,0.036); demo_19/object_b (0.262,0.036); demo_28/object_b (0.203,-0.092); demo_24/object_b (0.179,0.158)

## Findings

1. **v6's actual lying tz was 0.012, not 0.014**: lying cans report root z = 0.008
   in the obs/sim (can origin is ~4 mm below the geometric center; upright rest
   z = 0.015, not 0.018), so `clip(0.008 + 0.002, 0.012, 0.045)` hit the clip FLOOR.
2. **The live trajopt PLAN-FAILs do not reproduce from a fresh planner**: every
   lying case (incl. demo_19/object_a from the untouched settle state, and demo_6's
   exact live attempt-1 call with tz=0.012 + exclude=(14,)) plans successfully at
   EVERY tz 0.012-0.028. The live failures at r 0.24-0.29 are long-running-process
   solver-state/stochastic effects (consistent with the runner's note that
   reset_seed() shifted plan_fail 4 -> 17), not tz geometry.
3. **NO-CANDIDATES (demo_24, r=0.219) is tz-sensitive**: empty at tz <= 0.014,
   1-2 candidates at 0.016-0.018, full 16 at tz >= 0.020 (higher target_z lets the
   mid-band (r >= 0.223) table entries pass the score gate). Raising lying tz to
   ~0.020 buys goalset coverage below the lying band's r >= 0.270 floor.
4. **demo_28/object_b was UPRIGHT at its first attempt** (axis z = +1.00,
   z = 0.015), not lying: its 3 air-closes at tz = 0.031 are an upright-family
   failure. A can in that episode is knocked over lying only later (t~300).
5. Obs decode confirmed: [0:8] joint_pos_rel vs default (0,-1.35,-0.3,-0.85,0,0)
   + gripper (0.04,0.04); [16:32] objects_canonical target-first, pos3 + quat4
   XYZW in robot-root frame == env-local (base at origin, identity); [32:34]
   basket_center_xy. Log xy cross-checks all within 0.6 mm.
