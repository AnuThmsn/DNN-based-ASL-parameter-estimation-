import torch
import torch.nn as nn
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

X = torch.randn(2000000, 4, device=device)
Y = torch.randn(2000000, 2, device=device)

net = nn.Sequential(
    nn.Linear(4, 256), nn.ELU(),
    nn.Linear(256, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 64), nn.ELU(),
    nn.Linear(64, 2)
).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.L1Loss()

batch_size = 4096
# Because it's already on GPU, we can iterate manually
start = time.time()
for epoch in range(2):
    permutation = torch.randperm(X.size()[0], device=device)
    for i in range(0, X.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        xb, yb = X[indices], Y[indices]
        optimizer.zero_grad()
        pred = net(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} done in {time.time() - start:.2f}s')
    start = time.time()

