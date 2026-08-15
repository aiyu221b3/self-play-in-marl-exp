from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .config import DEV,N
from .environment import maps
from .evaluation import trace,evaluate,density

T1="#AFCBE8"
T2="#C7B8EA"
MM="#E6B6BE"
TEXT="#4F4F5A"

terrain=LinearSegmentedColormap.from_list("terrain",["#F8D7DA","#D9EAF7","#D8F3DC","#FFF1C7","#E4D7F5"])
t1cm=LinearSegmentedColormap.from_list("t1",["#F2F6FB","#DDEAF5","#C5DAEC","#AFCBE8"])
t2cm=LinearSegmentedColormap.from_list("t2",["#F5F0FB","#E8DEF4","#D8C9EE","#C7B8EA"])
mmcm=LinearSegmentedColormap.from_list("mm",["#F9EFF1","#F1D6DA","#EBC5CB","#E6B6BE"])

def save(fig,o,n):
    o.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(o/n,dpi=240,bbox_inches="tight")
    plt.close(fig)

def run(a3p,h3p,a3m,h3m,l3m,o="results"):
    out=Path(o)
    out.mkdir(exist_ok=True)
    zs=maps(4,.15).detach().cpu().numpy()

    fig,ax=plt.subplots(2,2,figsize=(10,9))
    for i,a0 in enumerate(ax.flat):
        a0.imshow(zs[i],cmap=terrain,origin="lower",interpolation="bicubic")
        a0.set_title(f"Landscape {i+1}",color=TEXT)
        a0.set_xticks([]);a0.set_yticks([])
    save(fig,out,"reward_landscapes.png")

    z=torch.as_tensor(zs[0],dtype=torch.float32,device=DEV).unsqueeze(0)
    q1,q2,qm=trace(z,a3m)

    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111,projection="3d")
    xx,yy=np.meshgrid(np.arange(N),np.arange(N))
    ax.plot_surface(xx,yy,zs[0],cmap=terrain,alpha=.68,linewidth=0)
    for q,c,n in [(q1,T1,"Trader 1"),(q2,T2,"Trader 2"),(qm,MM,"Market Maker")]:
        ax.plot(q[:,1],q[:,0],zs[0][q[:,0],q[:,1]]+.03,color=c,linewidth=2,label=n)
        ax.scatter(q[0,1],q[0,0],zs[0][q[0,0],q[0,1]]+.05,color=c,s=70)
    ax.set_title("Three-Agent Play on a Hidden Reward Landscape",color=TEXT)
    ax.set_xlabel("X",color=TEXT);ax.set_ylabel("Y",color=TEXT);ax.set_zlabel("Reward",color=TEXT)
    ax.legend(frameon=False)
    save(fig,out,"terrain_trajectory.png")

    fig,ax=plt.subplots(figsize=(8,7))
    ax.imshow(zs[0],cmap=terrain,origin="lower",interpolation="bicubic")
    for q,c,n in [(q1,T1,"Trader 1"),(q2,T2,"Trader 2"),(qm,MM,"Market Maker")]:
        ax.plot(q[:,1],q[:,0],color=c,linewidth=2,label=n)
        ax.scatter(q[0,1],q[0,0],color=c,s=80)
        ax.scatter(q[-1,1],q[-1,0],color=c,s=90,marker="X")
    ax.set_title("Learned Three-Agent Trajectories",color=TEXT)
    ax.set_xticks([]);ax.set_yticks([]);ax.legend(frameon=False)
    save(fig,out,"agent_trajectories.png")

    den=density(a3m,zs)
    for d,nm,cm in zip(den,["Trader 1","Trader 2","Market Maker"],[t1cm,t2cm,mmcm]):
        fig,ax=plt.subplots(figsize=(7,6))
        im=ax.imshow(np.sqrt(d),origin="lower",cmap=cm,interpolation="bicubic")
        ax.set_title(f"{nm} Trajectory Density (200 episodes)",color=TEXT)
        ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=ax,label="√Visits")
        save(fig,out,f"{nm.lower().replace(' ','_')}_density.png")

    pmean=h3p[-20:].mean(0);pstd=h3p[-20:].std(0)
    mmean=h3m[-20:].mean(0);mstd=h3m[-20:].std(0)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for j,(name,c) in enumerate([("Trader 1",T1),("Trader 2",T2),("Market Maker",MM)]):
        ax[j].bar(["PPO","MAPPO"],[pmean[j],mmean[j]],yerr=[pstd[j],mstd[j]],capsize=6,color=c,edgecolor="white")
        ax[j].set_title(name,color=TEXT);ax[j].set_ylabel("Mean reward",color=TEXT)
    fig.suptitle("PPO vs MAPPO",fontsize=17,color=TEXT)
    save(fig,out,"ppo_vs_mappo.png")

    fig,ax=plt.subplots(figsize=(9,5))
    for j,(c,n) in enumerate([(T1,"Trader 1"),(T2,"Trader 2"),(MM,"Market Maker")]):
        ax.plot(h3p[:,j],color=c,alpha=.55,label=n)
    ax.set_xlabel("Training update");ax.set_ylabel("Mean reward");ax.set_title("Independent PPO Training",color=TEXT);ax.legend(frameon=False)
    save(fig,out,"ppo_training.png")

    fig,ax=plt.subplots(figsize=(9,5))
    for j,(c,n) in enumerate([(T1,"Trader 1"),(T2,"Trader 2"),(MM,"Market Maker")]):
        ax.plot(h3m[:,j],color=c,alpha=.3)
        ax.plot(pd.Series(h3m[:,j]).rolling(10,min_periods=1).mean(),color=c,linewidth=2,label=n)
    ax.set_xlabel("Training update");ax.set_ylabel("Mean reward");ax.set_title("MAPPO Training Dynamics",color=TEXT);ax.legend(frameon=False)
    save(fig,out,"mappo_training.png")

    ds=[0,.1,.2,.4,.6,.8]
    pe=[];me=[]
    for d in ds:
        r,h=evaluate(a3p,d=d);pe.append([d,*r,h])
        r,h=evaluate(a3m,d=d);me.append([d,*r,h])
    pd.DataFrame(pe,columns=["dist","trader1","trader2","mm","capture"]).to_csv(out/"ppo_distortion.csv",index=False)
    pd.DataFrame(me,columns=["dist","trader1","trader2","mm","capture"]).to_csv(out/"mappo_distortion.csv",index=False)
    for n,o0 in [("ppo_history.npy",h3p),("mappo_history.npy",h3m),("mappo_losses.npy",l3m),("trader1_density.npy",den[0]),("trader2_density.npy",den[1]),("market_maker_density.npy",den[2])]:
        np.save(out/n,o0)
    pd.DataFrame(h3p,columns=["trader1","trader2","market_maker"]).to_csv(out/"ppo_history.csv",index=False)
    pd.DataFrame(h3m,columns=["trader1","trader2","market_maker"]).to_csv(out/"mappo_history.csv",index=False)
    pd.DataFrame({"agent":["trader1","trader2","market_maker"],"ppo":h3p[-20:].mean(0),"mappo":h3m[-20:].mean(0)}).to_csv(out/"ppo_mappo_final.csv",index=False)
    pd.DataFrame({"agent":["trader1","trader2","market_maker"],"observation_dim":[13,13,13],"observation":["position + terrain patch + last-seen maker","position + terrain patch + last-seen maker","position + terrain patch + last-seen trader"]}).to_csv(out/"observation_spec.csv",index=False)
    return out
