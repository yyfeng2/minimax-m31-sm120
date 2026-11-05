import json, sys, time, threading, urllib.request

URL = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'MiniMax-M3.1-Flash'
PROMPT = ('Write a detailed technical explanation of how block-sparse attention '
          'works in modern LLM inference engines, covering paging, top-k block '
          'selection, split-K and online softmax. Be thorough.')

def one(n, tag):
    body = json.dumps({'model': MODEL, 'stream': True, 'max_tokens': n,
                       'temperature': 0.0, 'ignore_eos': True,
                       'messages': [{'role': 'user', 'content': PROMPT}]}).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time(); tf = None; nt = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: '): continue
            d = line[6:]
            if d == b'[DONE]': break
            o = json.loads(d)
            ch = o.get('choices') or []
            if ch and (ch[0].get('delta') or {}).get('content'):
                if tf is None: tf = time.time()
                nt += 1
    t1 = time.time()
    ttft = (tf - t0) if tf else float('nan')
    dt = t1 - (tf or t0)
    tps = (nt - 1) / dt if dt > 0 and nt > 1 else float('nan')
    print(f'[{tag}] tokens={nt} ttft={ttft:.3f}s decode={dt:.3f}s tok/s={tps:.2f}', flush=True)
    return tps

def group(k, n, tag):
    res = []
    ths = [threading.Thread(target=lambda i=i: res.append(one(n, f'{tag}-{i}'))) for i in range(k)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    el = time.time() - t0
    tot = k * n
    print(f'[{tag}] aggregate {tot} tok in {el:.3f}s -> {tot/el:.2f} tok/s', flush=True)

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'solo'): one(128, 'solo')
    if what in ('all', 'g4'): group(4, 128, 'g4')
    if what in ('all', 'g8'): group(8, 128, 'g8')
