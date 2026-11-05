import json, sys, urllib.request
URL="http://127.0.0.1:8000/v1/chat/completions"
p=sys.argv[2] if len(sys.argv)>2 else ("Explain how paged KV caches and block-sparse attention interact. "
    "Cover memory layout, top-k block selection and online softmax.")
body=json.dumps({"model":"MiniMax-M3.1-Flash","temperature":0.0,"max_tokens":900,
                 "messages":[{"role":"user","content":p}]}).encode()
r=urllib.request.Request(URL,body,{"Content-Type":"application/json"})
o=json.load(urllib.request.urlopen(r,timeout=600))
m=o["choices"][0]["message"]
u=o.get("usage",{})
print("USAGE", u.get("prompt_tokens"), u.get("completion_tokens"),
      (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
print("REASONING:", (m.get("reasoning") or "")[:200].replace("\n"," "))
print("TEXT_BEGIN")
print(m.get("content"))
print("TEXT_END")
open(sys.argv[1],"w").write((m.get("content") or "") + "\n<<REASON>>\n" + (m.get("reasoning") or ""))
