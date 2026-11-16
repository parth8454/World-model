import glob, copy, numpy as np, torch, torch.nn.functional as F, cv2
from diffnet import DiffNet, sample

dev = "cuda"
CTX_NOISE = 0.03   # tiny noise on the old frames so it learns to fix its own mistakes

# ---------- data (same as before) ----------
frames, acts, offsets, o = [], [], [], 0
for f in sorted(glob.glob("data/processed/*.npz")):
    d = np.load(f); n = len(d["frames"])
    if n < 100: continue
    frames.append(torch.from_numpy(d["frames"]).permute(0, 3, 1, 2))
    acts.append(torch.from_numpy(d["actions"]))
    offsets.append((o, n)); o += n
F_all = torch.cat(frames).contiguous().to(dev)
A_all = torch.cat(acts).to(dev)
valid = lambda vs: torch.cat([torch.arange(o + 3, o + n - 1) for o, n in vs]).to(dev)
train_idx = valid(offsets[:-1])
val_start = offsets[-1][0]
print("train samples:", len(train_idx))

def batch(idx):
    ii = idx[:, None] + torch.arange(-3, 1, device=dev)
    ctx = (F_all[ii].float() / 255) * 2 - 1
    y = (F_all[idx + 1].float() / 255) * 2 - 1
    return ctx.reshape(len(idx), 12, 64, 64), A_all[idx], y

model = DiffNet().to(dev)
ema = copy.deepcopy(model).eval()
for p in ema.parameters(): p.requires_grad_(False)
print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

@torch.no_grad()
def rollout(path, steps=24):
    g = val_start + 100
    ctx = (F_all[g - 3:g + 1].float() / 255) * 2 - 1
    preds = []
    for t in range(steps):
        a = A_all[g + t][None]
        c_in = ctx.reshape(1, 12, 64, 64)
        c_in = c_in + CTX_NOISE * torch.randn_like(c_in)
        nxt = sample(ema, c_in, a)
        preds.append(nxt[0]); ctx = torch.cat([ctx[1:], nxt], 0)
    real = (F_all[g + 1:g + 1 + steps].float() / 255) * 2 - 1
    pick = list(range(0, steps, 4))
    row = lambda fr: np.hstack([((fr[i].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8) for i in pick])
    img = np.vstack([row(real), row(torch.stack(preds))])
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ---------- train ----------
STEPS, BS = 20000, 64
for step in range(1, STEPS + 1):
    idx = train_idx[torch.randint(len(train_idx), (BS,), device=dev)]
    ctx, a, y = batch(idx)
    ctx = ctx + CTX_NOISE * torch.randn_like(ctx)
    t = torch.rand(BS, device=dev)
    z = torch.randn_like(y)
    xt = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * y   # between noise and real
    target = y - z                                                       # direction to walk
    with torch.autocast(dev, dtype=torch.bfloat16):
        loss = F.mse_loss(model(xt, ctx, a, t).float(), target)
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                       # slow-moving average copy = smoother, better pictures
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.lerp_(p, 1 - 0.999)
    if step % 200 == 0: print(f"step {step}  loss {loss.item():.4f}")
    if step % 2000 == 0:
        rollout(f"diff_rollout_{step}.png")
        torch.save(ema.state_dict(), "model_diff.pt")
print("done")