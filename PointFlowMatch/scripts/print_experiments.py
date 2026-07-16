import wandb
import pandas as pd

CHAINED_DIFF_RESULTS = [
    {"task_name": "unplug_charger", "exp_name": "chain_dif", "success_rate": 65},
    {"task_name": "close_door", "exp_name": "chain_dif", "success_rate": 21},
    {"task_name": "open_box", "exp_name": "chain_dif", "success_rate": 46},
    {"task_name": "open_fridge", "exp_name": "chain_dif", "success_rate": 37},
    {"task_name": "take_frame_off_hanger", "exp_name": "chain_dif", "success_rate": 43},
    {"task_name": "open_oven", "exp_name": "chain_dif", "success_rate": 16},
    {"task_name": "put_books_on_bookshelf", "exp_name": "chain_dif", "success_rate": 40},
    {"task_name": "take_shoes_out_of_box", "exp_name": "chain_dif", "success_rate": 9},
]

def exp_name_from_run(run_config: dict) -> str:
    model = run_config["model"]["_target_"]
    backbone = run_config["model"]["obs_encoder"]["_target_"]
    if model == "pfp.policy.fm_so3_policy.FMSO3Policy":
        if (
            "noise_type" not in run_config["model"]
            or run_config["model"]["noise_type"] == "uniform"
        ):
            return "pfp_so3"
        else:
            return "pfp_so3_b"
    elif model == "pfp.policy.fm_policy.FMPolicy":
        return "pfp_euclid"
    elif (
        model == "pfp.policy.ddim_policy.DDIMPolicy"
        and backbone == "pfp.backbones.pointnet.PointNetBackbone"
    ):
        return "pfp_ddim"
    elif model == "pfp.policy.fm_so3_policy.FMSO3PolicyImage":
        return "pfp_images"
    elif backbone == "pfp.backbones.mlp_3dp.MLP3DP":
        return "dp3"
    elif model == "pfp.policy.fm_policy.FMPolicyImage":
        return "adaflow"
    elif model == "pfp.policy.ddim_policy.DDIMPolicyImage":
        return "diffusion_policy"
    else:
        raise ValueError(f"Unknown experiment name from model: {model} and backbone: {backbone}")
    return

pd.set_option("display.precision", 2)
api = wandb.Api()
runs = api.runs("rl-lab-chisari/pfp-eval-fixed")

data_list = CHAINED_DIFF_RESULTS
for run in runs:
    if run.state in ["running", "failed", "crashed"]:
        continue
    if run.config["policy"]["num_k_infer"] != 50:
        continue
    if (
        "snr_sampler" in run.config["model"]
        and run.config["model"]["snr_sampler"] == "logit_normal"
    ):
        continue
    assert run.summary["episode"] == 99, "Not all runs have 100 episodes"
    data = {
        "task_name": run.config["env_runner"]["env_config"]["task_name"],
        "exp_name": exp_name_from_run(run.config),
        "success_rate": run.summary["success"]["mean"] * 100,
    }
    data_list.append(data)

rows = list(
    [
        "diffusion_policy",
        "adaflow",
        "dp3",
        "chain_dif",
        "pfp_images",
        "pfp_ddim",
        "pfp_euclid",
        "pfp_so3",
        "pfp_so3_b",
    ]
)
columns = [
    "unplug_charger",
    "close_door",
    "open_box",
    "open_fridge",
    "take_frame_off_hanger",
    "open_oven",
    "put_books_on_bookshelf",
    "take_shoes_out_of_box",
]
data_frame = pd.DataFrame.from_records(data_list)
comparison_frame = data_frame.groupby(["task_name", "exp_name"])
exp_count = comparison_frame.size().unstack(level=0)
exp_count = exp_count.reindex(index=rows, columns=columns)

exp_count = exp_count.style.applymap(lambda x: "background-color: yellow" if x != 3 else "")

paddings = [
    ("padding-right", "20px"),
    ("padding-left", "20px"),
    ("padding-bottom", "10px"),
    ("padding-top", "10px"),
]
exp_count.set_table_styles(
    [
        {
            "selector": "th, td",
            "props": paddings,
        }
    ]
)

exp_count.format("{:.0f}")
exp_count.to_html("experiments/exp_count.html")

exp_mean = comparison_frame.mean()["success_rate"].unstack(level=0)
exp_mean = exp_mean.reindex(index=rows, columns=columns)

exp_mean["Mean"] = exp_mean.mean(axis=1)

def highlight_max(s):
    return ["background-color: lightgreen" if v == s.max() else "" for v in s]

exp_mean_styled = exp_mean.style.apply(highlight_max, axis=0)

exp_mean_styled = exp_mean_styled.set_table_styles([{"selector": "th, td", "props": paddings}])

exp_mean_styled = exp_mean_styled.format("{:.1f}")

exp_mean_styled.to_html("experiments/exp_mean.html")

print("Done")
