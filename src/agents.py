import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .config import DEV,A,D,LR

class Net(nn.Module):
    def __init__(self,d=D,h=128):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(d,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.pi=nn.Linear(h,A)
        self.v=nn.Linear(h,1)
    def forward(self,x):
        h=self.f(x)
        return self.pi(h),self.v(h).squeeze(-1)

class Agent:
    def __init__(self,d=D):
        self.net=Net(d).to(DEV)
        self.opt=optim.Adam(self.net.parameters(),lr=LR)
    @torch.no_grad()
    def act(self,x):
        lg,v=self.net(x)
        d=Categorical(logits=lg)
        a=d.sample()
        return a,d.log_prob(a),v

class C3(nn.Module):
    def __init__(self,d=39,h=128):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(d,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,3))
    def forward(self,x):
        return self.f(x)
