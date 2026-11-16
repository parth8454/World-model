# preprocess.py
import json, glob, os, cv2, numpy as np

KEYS = ["w", "a", "s", "d", "space", "shift", "ctrl"]
os.makedirs("data/processed", exist_ok=True)

def action_vec(step):
    pressed = set(step["keyboard"]["keys"])
    keys = [float(f"key.keyboard.{k}" in pressed) for k in
            ["w", "a", "s", "d", "space", "left.shift", "left.control"]]
    dx = np.clip(step["mouse"]["dx"] / 50.0, -1, 1)
    dy = np.clip(step["mouse"]["dy"] / 50.0, -1, 1)
    click = float(0 in step["mouse"]["buttons"])
    return keys + [dx, dy, click]          # 10 numbers

for mp4 in sorted(glob.glob("data/raw/*.mp4")):
    name = os.path.basename(mp4)[:-4]
    out = f"data/processed/{name}.npz"
    if os.path.exists(out):
        continue

    actions = [action_vec(json.loads(l)) for l in open(mp4[:-4] + ".jsonl")]

    cap, frames = cv2.VideoCapture(mp4), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    n = min(len(frames), len(actions))
    np.savez_compressed(out,
        frames=np.array(frames[:n], dtype=np.uint8),
        actions=np.array(actions[:n], dtype=np.float32))
    print(name, n, "frames")