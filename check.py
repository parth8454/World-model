import numpy as np, cv2, glob
d = np.load(sorted(glob.glob("data/processed/*.npz"))[0])
print(d["frames"].shape, d["actions"].shape)
cv2.imwrite("check.png", cv2.cvtColor(np.hstack(d["frames"][::200][:6]), cv2.COLOR_RGB2BGR))