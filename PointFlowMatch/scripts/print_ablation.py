import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exp_name_from_run(run_config: dict) -> str:
    model = run_config["model"]["_target_"]
    backbone = run_config["model"]["obs_encoder"]["_target_"]
    noise_type = run_config["model"]["noise_type"] if "noise_type" in run_config["model"] else None
    if model == "pfp.policy.fm_so3_policy.FMSO3Policy":
        exp_name = "pfp_so3"
    elif model == "pfp.policy.fm_policy.FMPolicy" and noise_type == "gaussian":
        exp_name = "pfp_euclid"
    elif model == "pfp.policy.fm_policy.FMPolicy" and noise_type == "igso3":
        exp_name = "pfp_euclid_igso3"
    elif (
        model == "pfp.policy.ddim_policy.DDIMPolicy"
        and backbone == "pfp.backbones.pointnet.PointNetBackbone"
    ):
        exp_name = "pfp_ddim"
    elif model == "pfp.policy.fm_so3_policy.FMSO3PolicyImage":
        exp_name = "pfp_images"
    elif backbone == "pfp.backbones.mlp_3dp.MLP3DP":
        exp_name = "dp3"
    elif model == "pfp.policy.fm_policy.FMPolicyImage":
        exp_name = "adaflow"
    elif model == "pfp.policy.ddim_policy.DDIMPolicyImage":
        exp_name = "diffusion_policy"
    else:
        exp_name = "other"

    if run_config["model"].get("noise_type") == "biased":
        exp_name += "biased"
    if run_config["model"].get("snr_sampler") == "logit_normal":
        exp_name += "_logitnorm"
    return exp_name

pd.set_option("display.precision", 2)
api = wandb.Api()
runs = api.runs("rl-lab-chisari/pfp-eval-rebuttal")

data_list = []
for run in runs:
    if run.state in ["running", "failed", "crashed"]:
        continue
    exp_name = exp_name_from_run(run.config)
    if exp_name in ["other", "pfp_images", "dp3", "adaflow", "diffusion_policy"]:
        continue
    assert run.summary["episode"] == 99, "Not all runs have 100 episodes"
    data = {
        "task_name": run.config["env_runner"]["env_config"]["task_name"],
        "exp_name": exp_name,
        "k_steps": run.config["policy"]["num_k_infer"],
        "success_rate": run.summary["success"]["mean"] * 100,
    }
    data_list.append(data)

rows = list(
    [
        "pfp_ddim",
        "pfp_so3",
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
comparison_frame = data_frame.groupby(["task_name", "exp_name", "k_steps"])
exp_count = comparison_frame.size().unstack(level=0)
exp_count = exp_count.reindex(columns=columns)

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

slice_ = pd.IndexSlice[pd.IndexSlice[:, 16], :]
exp_count.set_properties(**{"border-bottom": "1px solid black"}, subset=slice_)

exp_count.format("{:.0f}")
exp_count.to_html("experiments/ablation_count.html")

exp_mean = comparison_frame.mean()["success_rate"].unstack(level=0)
exp_mean = exp_mean.reindex(columns=columns)

exp_mean["Mean"] = exp_mean.mean(axis=1)

def highlight_max(s):
    return ["background-color: lightgreen" if v == s.max() else "" for v in s]

exp_mean_styled = exp_mean.style.apply(highlight_max, axis=0)

exp_mean_styled = exp_mean_styled.set_table_styles([{"selector": "th, td", "props": paddings}])

slice_ = pd.IndexSlice[pd.IndexSlice[:, 16], :]
exp_mean_styled.set_properties(**{"border-bottom": "1px solid black"}, subset=slice_)

exp_mean_styled = exp_mean_styled.format("{:.1f}")

exp_mean_styled.to_html("experiments/ablation_mean.html")

ax = sns.relplot(
    data=data_frame[data_frame["exp_name"].isin(["pfp_euclid", "pfp_ddim"])],
    kind="line",
    x="k_steps",
    y="success_rate",
    hue="exp_name",
    hue_order=["pfp_euclid", "pfp_ddim"],
    errorbar=None,
    marker="o",
    markersize=8,
    legend=False,
    aspect=1.5,
)
ax.set_xlabels("K Inference Steps")
ax.set_ylabels("Success Rate")
plt.xscale("log")
plt.minorticks_off()
plt.xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
plt.legend(
    title="",
    labels=["CFM", "DDIM"],

)

plt.savefig("experiments/ablation_plot.png")

print("Done")
