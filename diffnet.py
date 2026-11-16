import math, torch, torch.nn as nn, torch.nn.functional as F

def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
        nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())

class DiffNet(nn.Module):
    def __init__(s, E=128):
        super().__init__()
        s.emb = nn.Sequential(nn.Linear(10 + 64, E), nn.SiLU(), nn.Linear(E, E), nn.SiLU())
        s.e1, s.e2, s.e3 = block(15, 64), block(64, 128), block(128, 256)
        s.mid = block(256, 256)
        s.d3, s.d2, s.d1 = block(512, 128), block(256, 64), block(128, 64)
        s.out = nn.Conv2d(64, 3, 1)
        s.c_e1, s.c_e2, s.c_e3 = nn.Linear(E, 64), nn.Linear(E, 128), nn.Linear(E, 256)
        s.c_mid = nn.Linear(E, 256)
        s.c_d3, s.c_d2, s.c_d1 = nn.Linear(E, 128), nn.Linear(E, 64), nn.Linear(E, 64)

    def forward(s, xt, ctx, a, t):
        # xt: noisy next frame (B,3,64,64) | ctx: last 4 frames (B,12,64,64)
        # a: action (B,10) | t: how noisy, 0..1 (B,)
        f = torch.arange(32, device=xt.device)
        ang = (t[:, None] * 1000) * torch.exp(-math.log(10000) * f / 32)[None]
        c = s.emb(torch.cat([a, ang.sin(), ang.cos()], 1))
        g = lambda lin: lin(c)[:, :, None, None]
        up = lambda z: F.interpolate(z, scale_factor=2)
        e1 = s.e1(torch.cat([xt, ctx], 1)) + g(s.c_e1)
        e2 = s.e2(F.avg_pool2d(e1, 2)) + g(s.c_e2)
        e3 = s.e3(F.avg_pool2d(e2, 2)) + g(s.c_e3)
        m = s.mid(F.avg_pool2d(e3, 2)) + g(s.c_mid)
        h = s.d3(torch.cat([up(m), e3], 1)) + g(s.c_d3)
        h = s.d2(torch.cat([up(h), e2], 1)) + g(s.c_d2)
        h = s.d1(torch.cat([up(h), e1], 1)) + g(s.c_d1)
        return s.out(h)

@torch.no_grad()
def sample(model, ctx, a, steps=8):
    """Start from pure noise, clean it up in a few steps."""
    x = torch.randn(ctx.shape[0], 3, 64, 64, device=ctx.device)
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        x = x + model(x, ctx, a, t).float() / steps
    return x.clamp(-1, 1)