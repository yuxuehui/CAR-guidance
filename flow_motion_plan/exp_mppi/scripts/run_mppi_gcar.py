#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.experiments.exp1_static import Exp1Static
from experiments.core.config_loader import load_config
from experiments.core.evaluator import Evaluator
from experiments.utils.data_loader import load_success_trajectories
from exp_mppi.core.mppi_flow_controller import MPPIFlowController

def main():
    parser = argparse.ArgumentParser(description="MPPI + g^car (static obstacles)")
    parser.add_argument("--gcov-config", default="experiments/configs/exp1_static_gcov.yaml")
    parser.add_argument("--mppi-config", default="exp_mppi/configs/mppi_exp1_static.yaml")
    parser.add_argument("--cases", default="experiments/data/base_model_images/success_trajectories.json")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--output", default="exp_mppi/outputs/mppi_gcar_exp1/results.json")
    args = parser.parse_args()

    gcov_cfg = load_config(str(project_root / args.gcov_config))
    with open(project_root / args.mppi_config) as f:
        mppi_cfg = yaml.safe_load(f)["mppi"]

    test_cases = load_success_trajectories(str(project_root / args.cases))[: args.num_cases]

    exp = Exp1Static(model_checkpoint=gcov_cfg["model"]["checkpoint_path"], config=gcov_cfg)
    controller = MPPIFlowController(exp.model, mppi_cfg)
    evaluator = Evaluator({"goal_tolerance": 0.3, "collision_margin": 0.0, "wall_size": 1.0})

    results, n_success, n_safe = [], 0, 0
    for i, case in enumerate(test_cases):
        start, goal, walls = case["start"], case["goal"], case["walls"]
        print(f"\n===== case {i + 1}/{len(test_cases)} =====")

        guided = np.asarray(exp.generate_trajectory(start, goal, walls, num_samples=1)[0])

        ec = getattr(exp, "_last_energy_centers", None)
        es = gcov_cfg["guidance"].get("static_energy_scales", [-2.0] * (len(ec) if ec is not None else 1))

        final = np.asarray(controller.generate_trajectory(
            start, goal, walls, base_traj=guided, energy_centers=ec, energy_scales=es))

        m = evaluator.evaluate(trajectory=final, goal_pos=np.asarray(goal), walls=np.asarray(walls))
        n_success += int(m["success"]); n_safe += int(not m.get("collision", False))
        results.append({"case": i, "success": bool(m["success"]),
                        "collision": bool(m.get("collision", False)),
                        "goal_distance": float(m["goal_distance"])})
        print(f"  success={m['success']} collision={m.get('collision')} goal_dist={m['goal_distance']:.3f}")

    out = project_root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"num_cases": len(test_cases), "success_rate": n_success / max(len(test_cases), 1),
               "safe_rate": n_safe / max(len(test_cases), 1)}
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print(f"\n==== MPPI + g^car summary ====\n{summary}\n saved -> {out}")

if __name__ == "__main__":
    main()
