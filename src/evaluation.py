import numpy as np
import torch
from .config import DEV
from .environment import maps,reset,obs,step

@torch.no_grad()
def trace(z,a,t=80):
    p1,p2,pm,s1,s2,sm=reset(1)
    q1=[p1[0].cpu().numpy()]
    q2=[p2[0].cpu().numpy()]
    qm=[pm[0].cpu().numpy()]
    for _ in range(t):
        x1=obs(z,p1,s1)
        x2=obs(z,p2,s2)
        xm=obs(z,pm,sm)
        a1=a[0].net(x1)[0].argmax(1)
        a2=a[1].net(x2)[0].argmax(1)
        am=a[2].net(xm)[0].argmax(1)
        p1,p2,pm,s1,s2,sm,r1,r2,rm,dn=step(z,p1,p2,pm,s1,s2,sm,a1,a2,am)
        q1.append(p1[0].cpu().numpy())
        q2.append(p2[0].cpu().numpy())
        qm.append(pm[0].cpu().numpy())
        if dn[0]:
            break
    return np.asarray(q1),np.asarray(q2),np.asarray(qm)

@torch.no_grad()
def evaluate(a,n=512,t=100,d=.15):
    z=maps(n,d)
    p1,p2,pm,s1,s2,sm=reset(n)
    rr=[[],[],[]]
    hh=[]
    for _ in range(t):
        x1=obs(z,p1,s1)
        x2=obs(z,p2,s2)
        xm=obs(z,pm,sm)
        a1=a[0].net(x1)[0].argmax(1)
        a2=a[1].net(x2)[0].argmax(1)
        am=a[2].net(xm)[0].argmax(1)
        p1,p2,pm,s1,s2,sm,r1,r2,rm,dn=step(z,p1,p2,pm,s1,s2,sm,a1,a2,am)
        rr[0].append(r1);rr[1].append(r2);rr[2].append(rm);hh.append(dn)
    return np.array([torch.stack(rr[j]).mean().item() for j in range(3)]),torch.stack(hh).any(0).float().mean().item()

def density(a,zs,e=200):
    n=zs[0].shape[-1]
    d=[np.zeros((n,n)),np.zeros((n,n)),np.zeros((n,n))]
    for _ in range(e):
        z=torch.as_tensor(zs[np.random.randint(len(zs))],dtype=torch.float32,device=DEV).unsqueeze(0)
        q=trace(z,a)
        for j,v in enumerate(q):
            for y,x in v.astype(int):
                d[j][y,x]+=1
    return d
