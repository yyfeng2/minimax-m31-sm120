#!/usr/bin/env python3
"""校验 /data/MiniMax-M3.1-Flash 是否完整（对比 ModelScope 远端 Size）。"""
import json, os, subprocess, sys

REPO = "MiniMax/MiniMax-M3.1-Flash"
LOCAL = "/data/MiniMax-M3.1-Flash"

def remote_files():
    url = ("https://www.modelscope.cn/api/v1/models/%s/repo/files"
           "?Revision=master&Root=" % REPO)
    out = subprocess.run(["curl", "-s", "--max-time", "30", url],
                         capture_output=True, text=True).stdout
    data = json.loads(out)["Data"]["Files"]
    subs = [f["Path"] for f in data if f["Type"] == "tree"]
    files = {f["Path"]: f.get("Size", 0) or 0 for f in data if f["Type"] != "tree"}
    for s in subs:                       # 递归子目录（docs/ dspark/ figures/）
        sub_url = ("https://www.modelscope.cn/api/v1/models/%s/repo/files"
                   "?Revision=master&Root=%s" % (REPO, s))
        try:
            so = subprocess.run(["curl", "-s", "--max-time", "30", sub_url],
                                capture_output=True, text=True).stdout
            for f in json.loads(so)["Data"]["Files"]:
                if f["Type"] != "tree":
                    files[f["Path"]] = f.get("Size", 0) or 0   # Path 已含目录前缀
        except Exception:
            pass
    return files

def main():
    files = remote_files()
    total = sum(files.values())
    ok = miss = bad = done_bytes = 0
    problems = []
    for path, size in sorted(files.items()):
        lp = os.path.join(LOCAL, path)
        ls = os.path.getsize(lp) if os.path.exists(lp) else -1
        done_bytes += max(ls, 0)
        if ls < 0:
            miss += 1; problems.append("MISSING  %s" % path)
        elif ls != size:
            bad += 1; problems.append("SIZE     %s  local=%d remote=%d" % (path, ls, size))
        else:
            ok += 1
    print("files ok=%d  missing=%d  size-mismatch=%d  of %d" % (ok, miss, bad, len(files)))
    on_disk = 0
    for root, _, fs_ in os.walk(LOCAL):
        for f_ in fs_:
            try:
                on_disk += os.path.getsize(os.path.join(root, f_))
            except OSError:
                pass
    print("on-disk %.2f / %.2f GB  (%.1f%%)   [含进行中的临时分片]" % (
        on_disk/1e9, total/1e9, 100.0*on_disk/total))
    for p in problems[:20]:
        print(" ", p)
    if len(problems) > 20:
        print("  ... 另 %d 项" % (len(problems) - 20))
    return 0 if (ok == len(files)) else 1

sys.exit(main())
