import torch, torch.nn as nn
class DeepFM(nn.Module):
    def __init__(s, nf=10, nfeat=1000, ed=16):
        super().__init__()
        s.bias = nn.Parameter(torch.zeros(1))
        s.lin = nn.Embedding(nfeat, 1)
        s.emb = nn.Embedding(nfeat, ed)
        s.dnn = nn.Sequential(nn.Linear(nf*ed,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,1))
    def forward(s, x):
        lp = s.lin(x).sum(1) + s.bias
        e = s.emb(x)
        fm = 0.5*(e.sum(1).pow(2)-e.pow(2).sum(1)).sum(1,keepdim=True)
        return torch.sigmoid(lp + fm + s.dnn(e.view(e.size(0),-1)))
if __name__ == "__main__":
    d = "cuda" if torch.cuda.is_available() else "cpu"
    m = DeepFM().to(d)
    print(f"DeepFM on {torch.cuda.get_device_name(0)}, {sum(p.numel() for p in m.parameters())/1e6:.1f}M")