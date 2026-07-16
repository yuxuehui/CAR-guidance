import sys, yaml, random
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[2]); sys.path.insert(0, ROOT)
from experiments.experiments.exp1_static import Exp1Static
from experiments.core.config_loader import load_config
from experiments.core.evaluator import Evaluator
from experiments.utils.data_loader import load_success_trajectories
from exp_mppi.core.mppi_flow_controller import MPPIFlowController

N=10
gcov = load_config(f"{ROOT}/experiments/configs/exp1_static_gcov.yaml")
with open(f"{ROOT}/exp_mppi/configs/mppi_exp1_static.yaml") as f: mppi=yaml.safe_load(f)["mppi"]
cases = load_success_trajectories(f"{ROOT}/experiments/data/base_model_images/success_trajectories.json")[:N]
es=[-1.0,-1.0]
exp = Exp1Static(model_checkpoint=gcov["model"]["checkpoint_path"], config=gcov)
ctrl = MPPIFlowController(exp.model, mppi)
ev = Evaluator({"goal_tolerance":0.3,"collision_margin":0.0,"wall_size":1.0})

def seed(s=42): torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)
def jerk(t):
    a = t[2:] - 2*t[1:-1] + t[:-2]
    return float(np.linalg.norm(a,axis=1).mean())
def draw(ax, traj, walls, ec, start, goal, title):
    for w in walls:
        if abs(w[0])<1e-6 and abs(w[1])<1e-6: continue
        ax.add_patch(patches.Rectangle((w[0]-0.5,w[1]-0.5),1,1,color="royalblue",alpha=0.7))
    ax.plot(traj[:,0],traj[:,1],"-",color="crimson",lw=2)
    ax.scatter(*start,c="green",s=70,zorder=5); ax.scatter(*goal,c="red",marker="*",s=150,zorder=5)
    ax.scatter(ec[:,0],ec[:,1],c="orange",marker="x",s=110,lw=3,zorder=5)
    ax.set_xlim(0,5); ax.set_ylim(0,5); ax.set_aspect("equal"); ax.set_title(title,fontsize=9)

rows=[]
for i,c in enumerate(cases):
    start,goal,walls=c["start"],c["goal"],c["walls"]

    exp.guidance_config["use_gcov_optimization"]=True; exp.auto_select_centers=True
    seed(42); t_gcar=np.asarray(exp.generate_trajectory(start,goal,walls,num_samples=1)[0]); ec=np.asarray(exp._last_energy_centers,dtype=np.float32)

    exp.guidance_config["use_gcov_optimization"]=False
    seed(42); t_gapprox=np.asarray(exp.generate_trajectory(start,goal,walls,num_samples=1)[0])

    seed(42); t_base=ctrl._get_base_traj(start,goal,walls).cpu().numpy()

    t_mppi=np.asarray(ctrl.generate_trajectory(start,goal,walls,base_traj=t_base,energy_centers=ec.tolist(),energy_scales=es))
    t_gc=np.asarray(ctrl.generate_trajectory(start,goal,walls,base_traj=t_base,gcar_traj=t_gcar,energy_centers=ec.tolist(),energy_scales=es))

    conds=[("(1) base (no field)",t_base),("(2) g^approx",t_gapprox),("(3) MPPI",t_mppi),("(4) MPPI+g^car",t_gc)]
    fig,axes=plt.subplots(1,4,figsize=(20,5))
    for ax,(name,t) in zip(axes,conds):
        m=ev.evaluate(trajectory=t,goal_pos=np.array(goal),walls=np.array(walls))
        jk=jerk(t)
        draw(ax,t,walls,ec,start,goal,f"{name}\nreach={m['success']} wallcoll={m.get('collision')} jerk={jk:.3f}")
        rows.append({"case":i,"method":name,"reach":bool(m["success"]),"wall_coll":bool(m.get("collision",False)),"jerk":round(jk,3)})
    import os; out=f"{ROOT}/exp_mppi/outputs/compare_4way/case_{i}.png"; os.makedirs(os.path.dirname(out),exist_ok=True)
    fig.tight_layout(); fig.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig); print(f"case {i} -> {out}")

print("\n=== metrics (jerk=mean accel, lower=smoother/on-manifold) ===")
for r in rows: print(f"  case{r['case']} {r['method']:22s} reach={r['reach']} wall_coll={r['wall_coll']} jerk={r['jerk']}")

from collections import defaultdict
agg=defaultdict(lambda:[0,0.0,0])
for r in rows:
    ok = r["reach"] and not r["wall_coll"]
    agg[r["method"]][0]+= int(ok); agg[r["method"]][1]+= r["jerk"]; agg[r["method"]][2]+=1
print("\n=== AGGREGATE over", N, "cases ===")
for m,(s,jk,n) in agg.items():
    print(f"  {m:22s} success(reach&no-coll)={s}/{n}={s/n:.0%}  mean_jerk={jk/n:.3f}")
