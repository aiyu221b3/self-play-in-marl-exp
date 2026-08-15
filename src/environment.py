import torch
import numpy as np
from .config import DEV,N,B

u=torch.linspace(0,1,N,device=DEV)
v=torch.linspace(0,1,N,device=DEV)
X,Y=torch.meshgrid(u,v,indexing="ij")
MV=torch.tensor([[-1,0],[1,0],[0,-1],[0,1],[0,0]],device=DEV)

def norm(z):
    z=z-z.amin((1,2),keepdim=True)
    return z/(z.amax((1,2),keepdim=True)+1e-8)

def maps(b=B,d=.15):
    f=torch.randint(0,5,(b,),device=DEV)
    z=torch.zeros(b,N,N,device=DEV)
    c=torch.rand(b,2,device=DEV)
    s=torch.rand(b,device=DEV)*.18+.07
    g=torch.exp(-((X[None]-c[:,0,None,None])**2+(Y[None]-c[:,1,None,None])**2)/(2*s[:,None,None]**2))
    z[f==0]=g[f==0]
    ax=torch.rand(b,device=DEV)*1.5+.5
    ay=torch.rand(b,device=DEV)*1.5+.5
    z[f==1]=(ax[:,None,None]*X[None]+ay[:,None,None]*Y[None])[f==1]
    sx=torch.rand(b,device=DEV)*8+6
    sy=torch.rand(b,device=DEV)*8+6
    z[f==2]=((torch.sin(X[None]*sx[:,None,None])+torch.cos(Y[None]*sy[:,None,None])+2)/4)[f==2]
    c1=torch.rand(b,2,device=DEV)
    c2=torch.rand(b,2,device=DEV)
    s1=torch.rand(b,device=DEV)*.12+.05
    s2=torch.rand(b,device=DEV)*.12+.05
    g1=torch.exp(-((X[None]-c1[:,0,None,None])**2+(Y[None]-c1[:,1,None,None])**2)/(2*s1[:,None,None]**2))
    g2=torch.exp(-((X[None]-c2[:,0,None,None])**2+(Y[None]-c2[:,1,None,None])**2)/(2*s2[:,None,None]**2))
    z[f==3]=(g1+.8*g2)[f==3]
    g=torch.abs(torch.sin(X*10)*torch.cos(Y*10))
    z[f==4]=g.expand(b,-1,-1)[f==4]
    return norm((1-d)*z+d*torch.rand_like(z))

def reset(b=B):
    p1=torch.randint(0,N,(b,2),device=DEV)
    p2=torch.randint(0,N,(b,2),device=DEV)
    pm=torch.randint(0,N,(b,2),device=DEV)
    while True:
        q=(p1==p2).all(1)|(p1==pm).all(1)|(p2==pm).all(1)
        if not q.any():
            break
        p2[q]=torch.randint(0,N,(q.sum(),2),device=DEV)
        pm[q]=torch.randint(0,N,(q.sum(),2),device=DEV)
    s1=torch.full_like(p1,-1)
    s2=torch.full_like(p2,-1)
    sm=torch.full_like(pm,-1)
    return p1,p2,pm,s1,s2,sm

def patch(z,p,r=1):
    b=z.shape[0]
    d=torch.arange(-r,r+1,device=DEV)
    dx,dy=torch.meshgrid(d,d,indexing="ij")
    ix=(p[:,0,None,None]+dx).clamp(0,N-1)
    iy=(p[:,1,None,None]+dy).clamp(0,N-1)
    q=z[torch.arange(b,device=DEV)[:,None,None],ix,iy]
    return q.reshape(b,-1)

def obs(z,p,last):
    return torch.cat((p.float()/(N-1),patch(z,p),last.float()/(N-1)),1)

def near(a,b):
    return (a-b).abs().amax(1)<=3

def step(z,p1,p2,pm,s1,s2,sm,a1,a2,am):
    b=z.shape[0]
    i=torch.arange(b,device=DEV)
    p1=(p1+MV[a1]).clamp(0,N-1)
    p2=(p2+MV[a2]).clamp(0,N-1)
    pm=(pm+MV[am]).clamp(0,N-1)
    r1=z[i,p1[:,0],p1[:,1]]-.05
    r2=z[i,p2[:,0],p2[:,1]]-.05
    rm=torch.full((b,),-.05,device=DEV)
    h1=(p1==pm).all(1)
    h2=(p2==pm).all(1)
    h=h1|h2
    r1=torch.where(h1,torch.full_like(r1,-10),r1)
    r2=torch.where(h2,torch.full_like(r2,-10),r2)
    rm=torch.where(h,torch.full_like(rm,10),rm)
    n1=near(p1,pm)
    n2=near(p2,pm)
    s1=torch.where(n1[:,None],pm,s1)
    s2=torch.where(n2[:,None],pm,s2)
    d1=(p1-pm).abs().amax(1)
    d2=(p2-pm).abs().amax(1)
    q=torch.where((d1<=d2)[:,None],p1,p2)
    sm=torch.where((torch.minimum(d1,d2)<=3)[:,None],q,sm)
    return p1,p2,pm,s1,s2,sm,r1,r2,rm,h
