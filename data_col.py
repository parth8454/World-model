# download.py
import json, os, urllib.request

INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json"
N_VIDEOS = 10
OUT = "data/raw"
os.makedirs(OUT, exist_ok=True)

import time

def download(url, dest, tries=5):
    for attempt in range(1, tries + 1):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r   {done/1e6:.0f} / {total/1e6:.0f} MB ({done/total:.0%})", end="")
            os.rename(tmp, dest)
            print()
            return
        except Exception as e:
            print(f"\n   failed ({type(e).__name__}), retry {attempt}/{tries}")
            time.sleep(3)
    print("   giving up on this one, moving on")

index = json.load(urllib.request.urlopen(INDEX_URL))
base = index["basedir"]
paths = index["relpaths"][:N_VIDEOS]

for i, p in enumerate(paths):
    stem = os.path.splitext(p)[0]
    name = os.path.basename(stem)
    for ext in (".mp4", ".jsonl"):
        dest = f"{OUT}/{name}{ext}"
        if not os.path.exists(dest):
            print(f"[{i+1}/{len(paths)}] {name}{ext}")
            download(base + stem + ext, dest)
print("done")