import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
from data_gen import *

results = []

def train_model(net, x_tr, y_tr, x_val, y_val, criterion, lr=1e-3, epochs=60, batch_size=4096, patience=15, is_shared=False, wd=0, opt='adam'):
    if opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        
    best_val = float("inf")
    best_state = None
    no_imp = 0
    
    for epoch in range(epochs):
        net.train()
        permutation = torch.randperm(x_tr.size()[0], device=device)
        for i in range(0, x_tr.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            xb, yb = x_tr[indices], y_tr[indices]
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, x_val.size()[0], batch_size):
                xb, yb = x_val[i:i+batch_size], y_val[i:i+batch_size]
                pred = net(xb)
                vl += criterion(pred, yb).item() * xb.size(0)
            vl /= x_val.size(0)
            
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    net.load_state_dict(best_state)
    return net

def evaluate(net, x_val, is_shared=False, output_constraint=False):
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, x_val.size()[0], 4096):
            p = net(x_val[i:i+4096])
            preds.append(p.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
    
    if output_constraint:
        # If constraint is applied, the output is directly in physical units
        # No rescaling needed!
        if is_shared:
            cbf_pred = preds[:, 0]
            att_pred = preds[:, 1]
        else:
            cbf_pred = preds[:, 0]
            att_pred = preds[:, 0] # actually this will just be returned
    else:
        if is_shared:
            cbf_pred = preds[:, 0] * CBF_std + CBF_mean
            att_pred = preds[:, 1] * ATT_std + ATT_mean
        else:
            # Single output
            cbf_pred = preds[:, 0] * CBF_std + CBF_mean
            att_pred = preds[:, 0] * ATT_std + ATT_mean
            
    # clamp
    cbf_pred = np.clip(cbf_pred, CBF_MIN, CBF_MAX)
    att_pred = np.clip(att_pred, ATT_MIN, ATT_MAX)
    return cbf_pred, att_pred

class StandardizedNet(nn.Module):
    def __init__(self, input_dim, n_hidden_layers, n_neurons, out_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, n_neurons), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ELU()]
        layers.append(nn.Linear(n_neurons, out_dim))
        self.backbone = nn.Sequential(*layers)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.backbone(x)

class SharedNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU()
        )
        self.cbf_head = nn.Sequential(
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 1)
        )
        self.att_head = nn.Sequential(
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x):
        s = self.shared(x)
        c = self.cbf_head(s)
        a = self.att_head(s)
        return torch.cat([c, a], dim=1)

# Baseline is Exp 0
cbf_net0 = StandardizedNet(4, 9, 50).to(device).eval().requires_grad_(False)
cbf_net0.load_state_dict(torch.load("asl_dnn_papermatched/CBF_papermatched_9x50_MAE_best.pth", map_location=device))
cbf_pred0, _ = evaluate(cbf_net0, X_val_t)

att_net0 = StandardizedNet(4, 9, 100).to(device).eval().requires_grad_(False)
att_net0.load_state_dict(torch.load("asl_dnn_papermatched/ATT_papermatched_9x100_MAE_best.pth", map_location=device))
_, att_pred0 = evaluate(att_net0, X_val_t)
cbf_pred0 = cbf_pred0.squeeze()
att_pred0 = att_pred0.squeeze()

results.append({
    "ID": 0, "Architecture": "Model A (9x50 / 9x100)", "Input": "S1-S4", "Loss": "MAE", "Physics lambda": 0, "Optimizer": "Adam", "LR": "1e-3",
    "CBF Val RMSE": calc_rmse(Y_val[:,0], cbf_pred0),
    "ATT Val RMSE": calc_rmse(Y_val[:,1], att_pred0),
    "CBF Val MAE": calc_mae(Y_val[:,0], cbf_pred0),
    "ATT Val MAE": calc_mae(Y_val[:,1], att_pred0)
})

def run_exp(id_num, arch_name, net_cbf, net_att, loss_name, criterion, is_shared, opt='adam', lr=1e-3, wd=0, inp_name="S1-S4", phys_lambda=0):
    print(f"Running Exp {id_num}...", flush=True)
    if is_shared:
        net_cbf = train_model(net_cbf.to(device), X_tr_t, Y_tr_both_t, X_val_t, Y_val_both_t, criterion, opt=opt, lr=lr, wd=wd, epochs=30, patience=5)
        c, a = evaluate(net_cbf, X_val_t, is_shared=True)
    else:
        net_cbf = train_model(net_cbf.to(device), X_tr_t, Y_tr_cbf_t, X_val_t, Y_val_cbf_t, criterion, opt=opt, lr=lr, wd=wd, epochs=30, patience=5)
        net_att = train_model(net_att.to(device), X_tr_t, Y_tr_att_t, X_val_t, Y_val_att_t, criterion, opt=opt, lr=lr, wd=wd, epochs=30, patience=5)
        c, _ = evaluate(net_cbf, X_val_t)
        _, a = evaluate(net_att, X_val_t)
        
    results.append({
        "ID": id_num, "Architecture": arch_name, "Input": inp_name, "Loss": loss_name, "Physics lambda": phys_lambda, "Optimizer": opt.capitalize(), "LR": str(lr),
        "CBF Val RMSE": calc_rmse(Y_val[:,0], c),
        "ATT Val RMSE": calc_rmse(Y_val[:,1], a),
        "CBF Val MAE": calc_mae(Y_val[:,0], c),
        "ATT Val MAE": calc_mae(Y_val[:,1], a)
    })
    pd.DataFrame(results).to_csv("val_results.csv", index=False)
    print(f"Exp {id_num} done. CBF RMSE={calc_rmse(Y_val[:,0], c):.4f}, ATT RMSE={calc_rmse(Y_val[:,1], a):.4f}", flush=True)

    
try:
    print("Starting phase 3...", flush=True)
    # Phase 3
    run_exp(2, "Model A", StandardizedNet(4,9,50), StandardizedNet(4,9,100), "MSE", nn.MSELoss(), False)
    run_exp(3, "Model A", StandardizedNet(4,9,50), StandardizedNet(4,9,100), "Huber", nn.SmoothL1Loss(), False)

    class WeightedLoss(nn.Module):
        def __init__(self, lmbda=1.0):
            super().__init__()
            self.lmbda = lmbda
            self.crit = nn.L1Loss(reduction='none')
        def forward(self, pred, true):
            loss = self.crit(pred, true)
            return torch.mean(loss[:, 0] + self.lmbda * loss[:, 1])

    run_exp(4, "Model D", SharedNet(4), None, "MAE + 1.0 ATT", WeightedLoss(1.0), True)

    # Phase 4
    run_exp(5, "Model B (256-256-128-64)", StandardizedNet(4, 4, 256, out_dim=2), None, "MAE", nn.L1Loss(), True)

    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, dim), nn.ELU())
        def forward(self, x):
            return x + self.net(x)

    class ResNetModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.in_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU())
            self.res1 = ResBlock(256)
            self.res2 = ResBlock(256)
            self.res3 = ResBlock(256)
            self.out_layer = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 2))
        def forward(self, x):
            x = self.in_layer(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            return self.out_layer(x)

    run_exp(6, "Model C (ResNet)", ResNetModel(4), None, "MAE", nn.L1Loss(), True)
    run_exp(7, "Model D (Shared+Heads)", SharedNet(4), None, "MAE", nn.L1Loss(), True)

    pd.DataFrame(results).to_csv("val_results.csv", index=False)
    print("Done Phase 3 and 4.", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
