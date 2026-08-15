import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm.auto import tqdm
from .config import DEV,B,T,G,GAE,CLIP,LR,D
from .environment import maps,reset,obs,step
from .agents import Agent,C3

def gae(r,v,dn,nv,g=G,lam=GAE):
    ad=torch.zeros_like(r)
    z=torch.zeros(r.shape[1],device=DEV)
    for t in range(r.shape[0]-1,-1,-1):
        q=nv if t==r.shape[0]-1 else v[t+1]
        m=1-dn[t].float()
        z=r[t]+g*q*m-v[t]+g*lam*m*z
        ad[t]=z
    return ad,ad+v

def upd(ag,ob,ac,old,ad,ret,ep=4,bs=8192):
    x=ob.flatten(0,1)
    a=ac.flatten()
    old=old.flatten()
    ad=ad.flatten()
    ret=ret.flatten()
    n=x.shape[0]
    for _ in range(ep):
        ix=torch.randperm(n,device=DEV)
        for j in range(0,n,bs):
            k=ix[j:j+bs]
            lg,v=ag.net(x[k])
            d=Categorical(logits=lg)
            lp=d.log_prob(a[k])
            rat=(lp-old[k]).exp()
            pl=-torch.min(rat*ad[k],rat.clamp(1-CLIP,1+CLIP)*ad[k]).mean()
            vl=(ret[k]-v).pow(2).mean()
            loss=pl+.5*vl-.01*d.entropy().mean()
            ag.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ag.net.parameters(),.5)
            ag.opt.step()

def q3(z,a,t=T):
    b=z.shape[0]
    p1,p2,pm,s1,s2,sm=reset(b)
    O=[torch.empty(t,b,D,device=DEV) for _ in range(3)]
    A=[torch.empty(t,b,dtype=torch.long,device=DEV) for _ in range(3)]
    R=[torch.empty(t,b,device=DEV) for _ in range(3)]
    V=[torch.empty(t,b,device=DEV) for _ in range(3)]
    L=[torch.empty(t,b,device=DEV) for _ in range(3)]
    DN=torch.empty(t,b,device=DEV)
    for k in range(t):
        x1=obs(z,p1,s1)
        x2=obs(z,p2,s2)
        xm=obs(z,pm,sm)
        a1,l1,v1=a[0].act(x1)
        a2,l2,v2=a[1].act(x2)
        am,lm,vm=a[2].act(xm)
        p1,p2,pm,s1,s2,sm,r1,r2,rm,dn=step(z,p1,p2,pm,s1,s2,sm,a1,a2,am)
        O[0][k],O[1][k],O[2][k]=x1,x2,xm
        A[0][k],A[1][k],A[2][k]=a1,a2,am
        R[0][k],R[1][k],R[2][k]=r1,r2,rm
        V[0][k],V[1][k],V[2][k]=v1,v2,vm
        L[0][k],L[1][k],L[2][k]=l1,l2,lm
        DN[k]=dn
        if dn.any():
            n=int(dn.sum().item())
            q1,q2,qm,u1,u2,um=reset(n)
            p1[dn],p2[dn],pm[dn]=q1,q2,qm
            s1[dn],s2[dn],sm[dn]=u1,u2,um
    return O,A,R,V,L,DN

def train_ppo(n=300,b=B,t=T,d=.15):
    a=[Agent(),Agent(),Agent()]
    h=[]
    for _ in tqdm(range(n)):
        O,A,R,V,L,DN=q3(maps(b,d),a,t)
        for j in range(3):
            with torch.no_grad():
                _,nv=a[j].net(O[j][-1])
                ad,ret=gae(R[j],V[j],DN,nv)
                ad=(ad-ad.mean())/(ad.std()+1e-8)
            upd(a[j],O[j],A[j],L[j],ad,ret)
        h.append([r.mean().item() for r in R])
    return a,np.asarray(h)

def mr3(z,a,c,t=T):
    b=z.shape[0]
    p1,p2,pm,s1,s2,sm=reset(b)
    O=[torch.empty(t,b,D,device=DEV) for _ in range(3)]
    A=[torch.empty(t,b,dtype=torch.long,device=DEV) for _ in range(3)]
    R=[torch.empty(t,b,device=DEV) for _ in range(3)]
    V=torch.empty(t,b,3,device=DEV)
    L=[torch.empty(t,b,device=DEV) for _ in range(3)]
    DN=torch.empty(t,b,device=DEV)
    with torch.no_grad():
        for k in range(t):
            x1=obs(z,p1,s1)
            x2=obs(z,p2,s2)
            xm=obs(z,pm,sm)
            a1,l1,_=a[0].act(x1)
            a2,l2,_=a[1].act(x2)
            am,lm,_=a[2].act(xm)
            p1,p2,pm,s1,s2,sm,r1,r2,rm,dn=step(z,p1,p2,pm,s1,s2,sm,a1,a2,am)
            O[0][k],O[1][k],O[2][k]=x1,x2,xm
            A[0][k],A[1][k],A[2][k]=a1,a2,am
            R[0][k],R[1][k],R[2][k]=r1,r2,rm
            V[k]=c(torch.cat((x1,x2,xm),-1))
            L[0][k],L[1][k],L[2][k]=l1,l2,lm
            DN[k]=dn
            if dn.any():
                n=int(dn.sum().item())
                q1,q2,qm,u1,u2,um=reset(n)
                p1[dn],p2[dn],pm[dn]=q1,q2,qm
                s1[dn],s2[dn],sm[dn]=u1,u2,um
        nv=c(torch.cat((obs(z,p1,s1),obs(z,p2,s2),obs(z,pm,sm)),-1))
    return O,A,R,V,L,DN,nv

def mu3(a,c,opt,q,ep=4,bs=8192):
    O,A,R,V,L,DN,nv=q
    with torch.no_grad():
        AD=[];RT=[]
        for j in range(3):
            ad,ret=gae(R[j],V[:,:,j],DN,nv[:,j])
            AD.append(((ad-ad.mean())/(ad.std()+1e-8)).detach())
            RT.append(ret.detach())
        X=[O[j].detach().flatten(0,1) for j in range(3)]
        AC=[A[j].detach().flatten() for j in range(3)]
        OL=[L[j].detach().flatten() for j in range(3)]
        AF=[AD[j].flatten() for j in range(3)]
        RR=[RT[j].flatten() for j in range(3)]
    n=X[0].shape[0]
    for _ in range(ep):
        ix=torch.randperm(n,device=DEV)
        for j in range(0,n,bs):
            k=ix[j:j+bs]
            lg=[a[i].net(X[i][k])[0] for i in range(3)]
            ds=[Categorical(logits=x) for x in lg]
            lp=[ds[i].log_prob(AC[i][k]) for i in range(3)]
            rat=[(lp[i]-OL[i][k]).exp() for i in range(3)]
            pl=sum(-torch.min(rat[i]*AF[i][k],rat[i].clamp(1-CLIP,1+CLIP)*AF[i][k]).mean() for i in range(3))/3
            v=c(torch.cat([X[i][k] for i in range(3)],-1))
            vl=sum((RR[i][k]-v[:,i]).pow(2).mean() for i in range(3))/3
            ent=sum(d.entropy().mean() for d in ds)/3
            loss=pl+.5*vl-.01*ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(a[0].net.parameters())+list(a[1].net.parameters())+list(a[2].net.parameters())+list(c.parameters()),.5)
            opt.step()
    return loss.item(),pl.item(),vl.item()

def train_mappo(n=300,b=B,t=T,d=.15):
    a=[Agent(),Agent(),Agent()]
    c=C3().to(DEV)
    opt=optim.Adam(list(a[0].net.parameters())+list(a[1].net.parameters())+list(a[2].net.parameters())+list(c.parameters()),lr=LR)
    h=[];l=[]
    for _ in tqdm(range(n)):
        q=mr3(maps(b,d),a,c,t)
        l.append(mu3(a,c,opt,q))
        h.append([r.mean().item() for r in q[2]])
    return a,c,np.asarray(h),np.asarray(l)
