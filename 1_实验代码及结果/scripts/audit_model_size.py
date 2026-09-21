"""Report the actual serialized parameter size used by the Torch model."""
from mprbps.fl_torch import _imports

torch, nn, F, *_ = _imports()
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.conv1=nn.Conv2d(3,32,3,padding=1); self.conv2=nn.Conv2d(32,64,3,padding=1); self.fc1=nn.Linear(64*8*8,128); self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2); x=F.max_pool2d(F.relu(self.conv2(x)),2); return self.fc2(F.relu(self.fc1(torch.flatten(x,1))))
m=SmallCNN(); n=sum(p.numel() for p in m.parameters()); print(f"parameters={n}; float32_megabits={n*32/1e6:.6f}")
