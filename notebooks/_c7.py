# Cell 7: UV Training (baseline + fine-tune + bootstrap)
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import copy

class UVDataset(Dataset):
    def __init__(self, feats_dict, label_map):
        self.samples, self.labels = [], []
        for uid, feats in feats_dict.items():
            lbl = label_map[uid]
            for t in feats:
                self.samples.append(t.astype(np.float32))
                self.labels.append(lbl)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return torch.tensor(self.samples[i]), torch.tensor(self.labels[i], dtype=torch.long)

def load_all_users():
    data = {}
    for f in sorted(os.listdir(PROCESSED_UV)):
        if not f.endswith(".npz"): continue
        try: uid = int(os.path.splitext(f)[0])
        except: continue
        data[uid] = np.load(os.path.join(PROCESSED_UV, f))["features"]
    return data

def split_attempts(feats, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(feats.shape[0])
    n = len(idx); n_tr = int(n * 0.70); n_v = int(n * 0.15)
    return idx[:n_tr], idx[n_tr:n_tr+n_v], idx[n_tr+n_v:]

def score_verification(model, X_genuine, X_impostor):
    if len(X_genuine) == 0 or len(X_impostor) == 0: return np.array([]), np.array([])
    model.eval()
    with torch.no_grad():
        ge = model.get_siamese_embed(torch.tensor(X_genuine, dtype=torch.float32).to(device)).cpu().numpy()
        ie = model.get_siamese_embed(torch.tensor(X_impostor, dtype=torch.float32).to(device)).cpu().numpy()
    tmpl = ge.mean(0, keepdims=True); tmpl /= (np.linalg.norm(tmpl) + 1e-8)
    gn = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-8)
    in_ = ie / (np.linalg.norm(ie, axis=1, keepdims=True) + 1e-8)
    return (gn @ tmpl.T).squeeze(), (in_ @ tmpl.T).squeeze()

def compute_far_at_tar(genuine, impostor, target_tar=0.90, n_steps=100_000):
    gen, imp = np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)
    if len(gen) == 0 or len(imp) == 0: return 1.0, 0.0
    all_s = np.concatenate([gen, imp])
    for t in np.linspace(all_s.max(), all_s.min(), n_steps):
        if np.mean(gen >= t) >= target_tar: return float(np.mean(imp >= t)), float(t)
    return 1.0, float(all_s.min())

def bootstrap_far(genuine, impostor, n_repeats=5000, target_tar=0.90, seed=0):
    rng = np.random.default_rng(seed)
    gen, imp = np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)
    far_list = []
    for _ in range(n_repeats):
        f, _ = compute_far_at_tar(rng.choice(gen, len(gen), replace=True),
                                   rng.choice(imp, len(imp), replace=True), target_tar, 10_000)
        far_list.append(f)
    arr = np.array(far_list)
    return float(arr.mean()), float(arr.std())

def format_far(v):
    if v == 0.0: return "0", "1/inf"
    return f"{v:.2e}", f"1/{int(round(1.0/v))}"

# Load data
all_users = load_all_users()
all_uids = sorted(all_users.keys())
n_total = len(all_uids)
print(f"Loaded {n_total} UV users")

# Adaptive split based on actual user count
if n_total >= 101:
    subset_base  = all_uids[:cfg.uv_baseline_n]
    val_add      = all_uids[cfg.uv_baseline_n:90]
    testfinal    = all_uids[90:101]
elif n_total >= 20:
    n_base  = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    subset_base = all_uids[:n_base]
    val_add     = all_uids[n_base:n_base+n_val]
    testfinal   = all_uids[n_base+n_val:]
    print(f"Adaptive split: baseline={len(subset_base)}, val={len(val_add)}, test={len(testfinal)}")
else:
    raise ValueError(f"Too few UV users: {n_total}. Check UV preprocessing.")

assert len(testfinal) > 0, "testfinal is empty"

# Step 1: Baseline
print(f"\n=== Baseline (n={len(subset_base)}) ===")
seed_models, avs = [], []
for seed in range(cfg.n_training_runs):
    ckpt = os.path.join(CKPT_DIR, f"baseline_n{len(subset_base)}_seed{seed}.pt")
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=device)
        m = UVModel(n_classes=len(subset_base)).to(device)
        m.load_state_dict(c["model_state"]); seed_models.append(m); avs.append(c["acc"])
        print(f"  seed={seed}: loaded (acc={c['acc']:.1f}%)"); continue
    torch.manual_seed(seed); np.random.seed(seed)
    tr_f, vf, tf = {}, {}, {}
    for uid in subset_base:
        f = all_users[uid]; it, iv, ite = split_attempts(f, seed)
        tr_f[uid] = f[it]; vf[uid] = f[iv]; tf[uid] = f[ite]
    uid2cls = {u: i for i, u in enumerate(subset_base)}
    tr_lo = DataLoader(UVDataset(tr_f, uid2cls), cfg.batch_size, shuffle=True, drop_last=True)
    va_lo = DataLoader(UVDataset(vf, uid2cls), cfg.batch_size)
    model = UVModel(n_classes=len(subset_base)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.baseline_lr)
    loss_fn = TotalLoss(cfg.alpha_tm, cfg.supcon_temperature)
    best_acc, best_ckpt = 0, None
    for epoch in range(cfg.baseline_epochs):
        model.train()
        for Xb, yb in tr_lo:
            Xb, yb = Xb.to(device), yb.to(device); opt.zero_grad()
            logits, p1 = model(Xb, True); _, p2 = model(Xb, True)
            loss, _ = loss_fn(logits, torch.cat([p1, p2], 0), yb)
            loss.backward(); opt.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for Xb, yb in va_lo:
                pred = model(Xb.to(device))[0].argmax(1).cpu()
                correct += (pred == yb).sum().item(); total += yb.size(0)
        acc = correct / total * 100
        if acc > best_acc:
            best_acc = acc; best_ckpt = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_ckpt)
    torch.save({"model_state": model.state_dict(), "acc": best_acc}, ckpt)
    seed_models.append(model); avs.append(best_acc)
    print(f"  seed={seed}: acc={best_acc:.1f}%")

chosen = int(np.argsort(avs)[len(avs)//2])
best_model = seed_models[chosen]
print(f"Chosen seed: {chosen}")

# Step 2: Fine-tune + test
print(f"\n=== Fine-tune + Bootstrap Test ({len(testfinal)} users) ===")
b_rows = [{"n_baseline": len(subset_base), "acc_val_mean": np.mean(avs), "acc_val_std": np.std(avs)}]
f_rows = []
for target_uid in testfinal:
    ft_ckpt = os.path.join(CKPT_DIR, f"ft_user{target_uid}_n{len(subset_base)}.pt")
    if os.path.exists(ft_ckpt):
        c = torch.load(ft_ckpt, map_location=device)
        ft = UVModel(n_classes=2).to(device); ft.load_state_dict(c["model_state"])
    else:
        torch.manual_seed(chosen); np.random.seed(chosen)
        ft = copy.deepcopy(best_model)
        for p in ft.branches.parameters(): p.requires_grad = False
        ft.head_a = nn.Linear(ft.embed_dim, 2).to(device); ft = ft.to(device)
        trainable = list(ft.head_a.parameters()) + list(ft.head_b.parameters()) + list(ft.siamese_proj.parameters())
        opt = torch.optim.Adam(trainable, lr=cfg.finetune_lr)
        crit = nn.CrossEntropyLoss()
        rng = np.random.default_rng(chosen); n_per = 150
        imp = np.concatenate([all_users[u][rng.choice(len(all_users[u]), max(1, n_per//len(subset_base)), replace=False)]
                              for u in subset_base], axis=0)[:n_per]
        own = all_users[target_uid][:n_per]
        X_mix = np.concatenate([imp, own]); y_mix = np.array([0]*len(imp) + [1]*len(own))
        shuf = rng.permutation(len(X_mix)); X_mix, y_mix = X_mix[shuf], y_mix[shuf]
        ft_lo = DataLoader(TensorDataset(
            torch.tensor(X_mix, dtype=torch.float32),
            torch.tensor(y_mix, dtype=torch.long)), 32, shuffle=True)
        val_imp = np.concatenate([all_users[u][:9] for u in val_add[:10]], 0) if val_add else imp[:90]
        val_gen = all_users[target_uid][:90]
        best_far, best_ft_ckpt = 1.0, None
        for epoch in range(cfg.finetune_epochs):
            ft.train()
            for Xb, yb in ft_lo:
                opt.zero_grad(); crit(ft(Xb.to(device))[0], yb.to(device)).backward(); opt.step()
            g, i = score_verification(ft, val_gen, val_imp)
            if len(g) > 0:
                fv = compute_far_at_tar(g, i, cfg.tar_threshold)[0]
                if fv < best_far:
                    best_far = fv; best_ft_ckpt = {k: v.clone() for k, v in ft.state_dict().items()}
        if best_ft_ckpt: ft.load_state_dict(best_ft_ckpt)
        torch.save({"model_state": ft.state_dict()}, ft_ckpt)
    gen = all_users[target_uid][:90]
    imp = np.concatenate([all_users[u][:9] for u in testfinal if u != target_uid], axis=0)[:90]
    g, i = score_verification(ft, gen, imp)
    if len(g) > 0:
        m, s = bootstrap_far(g, i, cfg.bootstrap_repeats, cfg.tar_threshold, seed=target_uid)
    else:
        m, s = 1.0, 0.0
    d, f = format_far(m)
    print(f"  user={target_uid}: FAR={m*100:.1f}+/-{s*100:.1f}% ({d}, {f})")
    f_rows.append({"user_id": target_uid, "n_baseline": len(subset_base), "far_mean": m, "far_std": s})
pd.DataFrame(b_rows).to_csv(os.path.join(RESULTS_DIR, "results_baseline.csv"), index=False)
pd.DataFrame(f_rows).to_csv(os.path.join(RESULTS_DIR, "results_uv_final.csv"), index=False)
print("\nUV training done.")