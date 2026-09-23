import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import *

# Define architectures dynamically based on input_dim
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

def get_best_arch(dim):
    df = pd.read_csv("val_results.csv")
    # Only consider Phase 4 models (ID 5, 6, 7) or baseline if it's somehow better
    # Actually just pick best CBF Val RMSE from architecture experiments
    df_arch = df[df['Architecture'].str.contains('Model')].copy()
    best_row = df_arch.loc[df_arch['CBF Val RMSE'].idxmin()]
    arch = best_row['Architecture']
    print(f"Best architecture selected: {arch}", flush=True)
    if "Model B" in arch:
        return StandardizedNet(dim, 4, 256, out_dim=2), arch
    elif "Model C" in arch:
        return ResNetModel(dim), arch
    elif "Model D" in arch:
        return SharedNet(dim), arch
    else:
        # Fallback to shared Model D if baseline was chosen, because we need a shared net
        # for physics loss later.
        return SharedNet(dim), "Model D (Shared+Heads)"

results = pd.read_csv("val_results.csv")
results = results[results['ID'] <= 7].to_dict('records')

def evaluate(net, x_val, output_constraint=False):
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, x_val.size()[0], 4096):
            p = net(x_val[i:i+4096])
            preds.append(p.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
    
    if output_constraint:
        cbf_pred = preds[:, 0]
        att_pred = preds[:, 1]
    else:
        cbf_pred = preds[:, 0] * CBF_std + CBF_mean
        att_pred = preds[:, 1] * ATT_std + ATT_mean
            
    cbf_pred = np.clip(cbf_pred, CBF_MIN, CBF_MAX)
    att_pred = np.clip(att_pred, ATT_MIN, ATT_MAX)
    return cbf_pred, att_pred

def train_model(net, x_tr, y_tr, x_val, y_val, criterion, lr=1e-3, epochs=60, batch_size=4096, patience=15, wd=0, opt='adam'):
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
            loss = criterion(pred, yb, xb) if hasattr(criterion, 'is_physics') else criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, x_val.size()[0], batch_size):
                xb, yb = x_val[i:i+batch_size], y_val[i:i+batch_size]
                pred = net(xb)
                loss = criterion(pred, yb, xb) if hasattr(criterion, 'is_physics') else criterion(pred, yb)
                vl += loss.item() * xb.size(0)
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

def run_exp(id_num, arch_name, net, loss_name, criterion, opt='adam', lr=1e-3, wd=0, inp_name="S1-S4", phys_lambda=0, output_constraint=False, x_tr=X_tr_t, x_val=X_val_t):
    print(f"Running Exp {id_num}...", flush=True)
    y_tr = Y_tr_both_t
    y_val = Y_val_both_t
    
    if output_constraint:
        y_tr = torch.tensor(Y_tr, device=device)
        y_val = torch.tensor(Y_val, device=device)
        
    net = train_model(net.to(device), x_tr, y_tr, x_val, y_val, criterion, opt=opt, lr=lr, wd=wd, epochs=30, patience=5)
    c, a = evaluate(net, x_val, output_constraint=output_constraint)
        
    results.append({
        "ID": id_num, "Architecture": arch_name, "Input": inp_name, "Loss": loss_name, "Physics lambda": phys_lambda, "Optimizer": opt.capitalize(), "LR": str(lr),
        "CBF Val RMSE": calc_rmse(Y_val[:,0], c),
        "ATT Val RMSE": calc_rmse(Y_val[:,1], a),
        "CBF Val MAE": calc_mae(Y_val[:,0], c),
        "ATT Val MAE": calc_mae(Y_val[:,1], a)
    })
    pd.DataFrame(results).to_csv("val_results.csv", index=False)
    print(f"Exp {id_num} done. CBF RMSE={calc_rmse(Y_val[:,0], c):.4f}", flush=True)

print("Phase 5...", flush=True)
def make_features(X_raw, type_name):
    eps = 1e-8
    if type_name == "B":
        f = [X_raw, X_raw[:,1:2]/(X_raw[:,0:1]+eps), X_raw[:,2:3]/(X_raw[:,0:1]+eps), X_raw[:,3:4]/(X_raw[:,0:1]+eps)]
    elif type_name == "C":
        f = [X_raw, np.log(X_raw + eps)]
    elif type_name == "D":
        ratios = [X_raw[:,1:2]/(X_raw[:,0:1]+eps), X_raw[:,2:3]/(X_raw[:,0:1]+eps), X_raw[:,3:4]/(X_raw[:,0:1]+eps)]
        logs = [np.log(X_raw + eps)]
        diffs = [X_raw[:,1:2]-X_raw[:,0:1], X_raw[:,2:3]-X_raw[:,1:2], X_raw[:,3:4]-X_raw[:,2:3]]
        f = [X_raw] + ratios + logs + diffs
    return np.concatenate(f, axis=1).astype(np.float32)

def prepare_input(type_name):
    X_tr_f = make_features(X_tr, type_name)
    X_val_f = make_features(X_val, type_name)
    m = X_tr_f.mean(axis=0, keepdims=True)
    s = X_tr_f.std(axis=0, keepdims=True) + 1e-8
    return torch.tensor((X_tr_f - m) / s, device=device), torch.tensor((X_val_f - m) / s, device=device), X_tr_f.shape[1], m, s

xtrB, xvalB, dimB, _, _ = prepare_input("B")
netB, arch = get_best_arch(dimB)
run_exp(8, arch, netB, "MAE", nn.L1Loss(), inp_name="Input B (ratios)", x_tr=xtrB, x_val=xvalB)

xtrC, xvalC, dimC, _, _ = prepare_input("C")
netC, arch = get_best_arch(dimC)
run_exp(9, arch, netC, "MAE", nn.L1Loss(), inp_name="Input C (log)", x_tr=xtrC, x_val=xvalC)

xtrD, xvalD, dimD, _, _ = prepare_input("D")
netD, arch = get_best_arch(dimD)
run_exp(10, arch, netD, "MAE", nn.L1Loss(), inp_name="Input D (all)", x_tr=xtrD, x_val=xvalD)

df = pd.DataFrame(results)
best_idx = df.loc[df['ID'].isin([7, 8, 9, 10]), 'CBF Val RMSE'].idxmin()
best_inp_row = df.loc[best_idx]
best_inp_name = best_inp_row['Input']
print(f"Best input: {best_inp_name}", flush=True)

if "Input B" in best_inp_name:
    X_tr_best, X_val_best, best_dim = xtrB, xvalB, dimB
elif "Input C" in best_inp_name:
    X_tr_best, X_val_best, best_dim = xtrC, xvalC, dimC
elif "Input D" in best_inp_name:
    X_tr_best, X_val_best, best_dim = xtrD, xvalD, dimD
else:
    X_tr_best, X_val_best, best_dim = X_tr_t, X_val_t, 4

print("Phase 6...", flush=True)
class PhysicsLoss(nn.Module):
    def __init__(self, lmbda, is_physical_pred=False):
        super().__init__()
        self.lmbda = lmbda
        self.is_physical_pred = is_physical_pred
        self.mae = nn.L1Loss()
        self.is_physics = True
    def forward(self, pred, true, x_in):
        loss_param = self.mae(pred, true)
        
        if self.is_physical_pred:
            cbf_phys = pred[:, 0]
            att_phys = pred[:, 1]
        else:
            cbf_phys = pred[:, 0] * CBF_std + CBF_mean
            att_phys = pred[:, 1] * ATT_std + ATT_mean
        
        f_per_s = cbf_phys / (6000.0 * lmbda)
        delta = att_phys
        
        pld = torch.tensor(PLDs, device=device, dtype=torch.float32).unsqueeze(0)
        prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s.unsqueeze(1)
        e_att = torch.exp(-delta / T1a).unsqueeze(1)
        
        t1 = torch.exp(-torch.clamp(pld - delta.unsqueeze(1), min=0.0) / T1t)
        t2 = torch.exp(-torch.clamp(tau + pld - delta.unsqueeze(1), min=0.0) / T1t)
        
        recon_sig = prefix * e_att * (t1 - t2) * SCALE
        
        X_raw = x_in[:, :4] * torch.tensor(X_std[:, :4], device=device) + torch.tensor(X_mean[:, :4], device=device)
        
        # Scale down physics loss because it can be huge, actually just MSE
        loss_phys = torch.mean((X_raw - recon_sig)**2)
        # Note: if pred is physical, loss_param is physical (MAE is large). If pred is standardized, MAE is small.
        return loss_param + self.lmbda * loss_phys

net, arch = get_best_arch(best_dim)
run_exp(11, arch, net, "MAE + 0.001 Physics", PhysicsLoss(0.001), inp_name=best_inp_name, phys_lambda=0.001, x_tr=X_tr_best, x_val=X_val_best)
net, arch = get_best_arch(best_dim)
run_exp(12, arch, net, "MAE + 0.01 Physics", PhysicsLoss(0.01), inp_name=best_inp_name, phys_lambda=0.01, x_tr=X_tr_best, x_val=X_val_best)
net, arch = get_best_arch(best_dim)
run_exp(13, arch, net, "MAE + 0.1 Physics", PhysicsLoss(0.1), inp_name=best_inp_name, phys_lambda=0.1, x_tr=X_tr_best, x_val=X_val_best)

df = pd.DataFrame(results)
best_p_idx = df.loc[df['ID'].isin([7, 8, 9, 10, 11, 12, 13]), 'CBF Val RMSE'].idxmin()
best_phys_row = df.loc[best_p_idx]
best_lambda = best_phys_row['Physics lambda']
best_loss_name = best_phys_row['Loss']

print("Phase 7...", flush=True)
class ConstrainedNet(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.base = base_net
    def forward(self, x):
        z = self.base(x)
        cbf = 100.0 * torch.sigmoid(z[:, 0:1])
        att = 0.5 + 2.5 * torch.sigmoid(z[:, 1:2])
        return torch.cat([cbf, att], dim=1)

net, arch = get_best_arch(best_dim)
c_net = ConstrainedNet(net)
if best_lambda > 0:
    c_crit = PhysicsLoss(best_lambda, is_physical_pred=True)
else:
    c_crit = nn.L1Loss()
run_exp(14, arch + " (Constrained)", c_net, best_loss_name, c_crit, inp_name=best_inp_name, phys_lambda=best_lambda, output_constraint=True, x_tr=X_tr_best, x_val=X_val_best)

print("Phase 8...", flush=True)
df = pd.DataFrame(results)
best_c_idx = df.loc[df['ID'].isin([int(best_phys_row['ID']), 14]), 'CBF Val RMSE'].idxmin()
best_constr_row = df.loc[best_c_idx]
use_constraint = (best_constr_row['ID'] == 14)

def get_base_net():
    n, a = get_best_arch(best_dim)
    if use_constraint:
        return ConstrainedNet(n), a + " (Constrained)"
    return n, a

if use_constraint:
    if best_lambda > 0:
        opt_crit = PhysicsLoss(best_lambda, is_physical_pred=True)
    else:
        opt_crit = nn.L1Loss()
else:
    if best_lambda > 0:
        opt_crit = PhysicsLoss(best_lambda)
    else:
        opt_crit = nn.L1Loss()

net, arch = get_base_net()
run_exp(15, arch, net, best_loss_name, opt_crit, opt='adamw', lr=1e-3, wd=1e-5, inp_name=best_inp_name, phys_lambda=best_lambda, output_constraint=use_constraint, x_tr=X_tr_best, x_val=X_val_best)
net, arch = get_base_net()
run_exp(16, arch, net, best_loss_name, opt_crit, opt='adamw', lr=5e-4, wd=1e-5, inp_name=best_inp_name, phys_lambda=best_lambda, output_constraint=use_constraint, x_tr=X_tr_best, x_val=X_val_best)
net, arch = get_base_net()
run_exp(17, arch, net, best_loss_name, opt_crit, opt='adamw', lr=1e-4, wd=0, inp_name=best_inp_name, phys_lambda=best_lambda, output_constraint=use_constraint, x_tr=X_tr_best, x_val=X_val_best)

pd.DataFrame(results).to_csv("val_results.csv", index=False)
print("All experiments done!", flush=True)
