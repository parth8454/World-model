import glob, numpy as np, torch, pygame
from diffnet import DiffNet, sample

dev = "cuda"
SAMPLE_STEPS = 6      # fewer = faster but rougher. try 4 to 10
CTX_NOISE = 0.03

model = DiffNet().to(dev)
model.load_state_dict(torch.load("model_diff.pt", map_location=dev))
model.eval()

d = np.load(sorted(glob.glob("data/processed/*.npz"))[0])
start = (torch.from_numpy(d["frames"][100:104]).permute(0, 3, 1, 2).float().to(dev) / 255) * 2 - 1
ctx = start.clone()

pygame.init()
screen = pygame.display.set_mode((512, 512))
clock = pygame.time.Clock()
print("w a s d = move | j l = look left/right | i k = look up/down | space = jump | c = click | r = reset | q = quit")

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q: running = False
            if e.key == pygame.K_r: ctx = start.clone()
    k = pygame.key.get_pressed()
    act = np.zeros(10, dtype=np.float32)
    act[0], act[1], act[2], act[3] = k[pygame.K_w], k[pygame.K_a], k[pygame.K_s], k[pygame.K_d]
    act[4] = k[pygame.K_SPACE]
    act[7] = 0.6 * (k[pygame.K_l] - k[pygame.K_j])
    act[8] = 0.6 * (k[pygame.K_k] - k[pygame.K_i])
    act[9] = k[pygame.K_c]
    a = torch.from_numpy(act)[None].to(dev)

    c_in = ctx.reshape(1, 12, 64, 64)
    c_in = c_in + CTX_NOISE * torch.randn_like(c_in)
    with torch.autocast(dev, dtype=torch.bfloat16):
        nxt = sample(model, c_in, a, SAMPLE_STEPS)
    ctx = torch.cat([ctx[1:], nxt], 0)

    img = ((nxt[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    screen.blit(pygame.transform.scale(surf, (512, 512)), (0, 0))
    pygame.display.flip()
    clock.tick(20)
pygame.quit()