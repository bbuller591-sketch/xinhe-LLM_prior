import os, json, urllib.request
base=os.environ.get("DEEPSEEK_BASE_URL","https://api.deepseek.com").rstrip("/")
key=os.environ["DEEPSEEK_API_KEY"]
model=os.environ.get("DEEPSEEK_MODEL","deepseek-chat")
body={"model":model,"messages":[{"role":"user","content":"Reply with exactly OK."}],"temperature":1.0,"max_tokens":32,"stream":False,"thinking":{"type":"disabled"}}
req=urllib.request.Request(base+"/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}))
with opener.open(req,timeout=60) as f:
    obj=json.loads(f.read().decode())
print(obj["model"], obj["choices"][0]["message"]["content"])
