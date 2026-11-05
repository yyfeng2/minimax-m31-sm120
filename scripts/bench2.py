import json, sys, time, threading, urllib.request
URL="http://127.0.0.1:8000/v1/chat/completions"
PROMPT=("Write a long, detailed technical report on paged KV caches and "
        "block-sparse attention in LLM serving engines: memory layout, top-k "
        "block selection, online softmax, split-K reduction, and tensor-core "
        "kernels. Keep going until the length limit.")
def one(n,tag,res):
    body=json.dumps({"model":"MiniMax-M3.1-Flash","temperature":0.0,"max_tokens":n,
        "messages":[{"role":"user","content":PROMPT}]}).encode()
    r=urllib.request.Request(URL,body,{"Content-Type":"application/json"})
    t0=time.time()
    o=json.load(urllib.request.urlopen(r,timeout=900))
    el=time.time()-t0
    u=o.get("usage") or {}
    nt=u.get("completion_tokens") or 0
    tps=(nt-1)/el if el>0 and nt>1 else float("nan")
    print(f"[{tag}] tokens={nt} wall={el:.2f}s tok/s={tps:.2f}",flush=True)
    res.append((nt,el,tps))
def group(k,n,tag):
    res=[]
    ths=[threading.Thread(target=one,args=(n,f"{tag}-{i}",res)) for i in range(k)]
    t0=time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    el=time.time()-t0
    tot=sum(r[0] for r in res)
    print(f"[{tag}] aggregate {tot} tok in {el:.2f}s -> {tot/el:.2f} tok/s",flush=True)
if __name__=="__main__":
    what=sys.argv[1] if len(sys.argv)>1 else "all"
    if what in ("all","solo"): group(1,384,"solo")
    if what in ("all","g4"): group(4,256,"g4")
    if what in ("all","g8"): group(8,256,"g8")
