import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 32,
        "axes.labelsize": 38,
    }
)

df_benchmark = pd.DataFrame(columns=["Baseline", "Success Rate", "std"])
df_benchmark.loc[len(df_benchmark)] = ["Dif. Policy", 18.7, 2.3]
df_benchmark.loc[len(df_benchmark)] = ["AdaFlow", 19.0, 2.3]
df_benchmark.loc[len(df_benchmark)] = ["3D-DP", 28.5, 2.2]
df_benchmark.loc[len(df_benchmark)] = ["OL-ChDif", 34.6, 0]
df_benchmark.loc[len(df_benchmark)] = ["PFM(ours)", 67.8, 4.1]

plt.figure(figsize=(16, 8))
ax = sns.barplot(df_benchmark, x="Baseline", y="Success Rate", color="#344A9A", width=0.6)
ax.errorbar(
    df_benchmark.index,
    df_benchmark["Success Rate"],
    yerr=df_benchmark["std"],
    fmt="none",
    c="black",
    capsize=10,
    capthick=5,
    elinewidth=5,
)
ax.set(xlabel="", ylabel="Success Rate (↑)")
plt.tight_layout()
plt.savefig("benchmark_plot.png")
plt.savefig("benchmark_plot.svg")
plt.clf()

df_ablation = pd.DataFrame(columns=["Baseline", "Success Rate", "std"])
df_ablation.loc[len(df_ablation)] = ["Img-CFM-R6", 40.1, 3.3]
df_ablation.loc[len(df_ablation)] = ["Pcd-DDIM-R6", 68.0, 4.3]
df_ablation.loc[len(df_ablation)] = ["Pcd-CFM-SO3", 67.4, 4.4]
df_ablation.loc[len(df_ablation)] = ["Pcd-CFM-R6", 67.8, 4.1]

plt.figure(figsize=(16, 8))
ax = sns.barplot(df_ablation, x="Baseline", y="Success Rate", color="#344A9A", width=0.6)
ax.errorbar(
    df_ablation.index,
    df_ablation["Success Rate"],
    yerr=df_ablation["std"],
    fmt="none",
    c="black",
    capsize=10,
    capthick=5,
    elinewidth=5,
)
ax.set(xlabel="", ylabel="Success Rate (↑)")
plt.tight_layout()
plt.savefig("ablation_plot.png")
plt.savefig("ablation_plot.svg")
plt.clf()
