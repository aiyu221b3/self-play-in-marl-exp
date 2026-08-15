import torch

DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
N,B,T,A=16,512,64,5
G,GAE,CLIP,LR=.99,.95,.2,3e-4
D=13
