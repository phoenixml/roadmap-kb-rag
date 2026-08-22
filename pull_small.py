"""Pull 4 new small models onto the Ollama Cloud endpoint, wait for each to finish."""
import urllib.request, json, socket, time
socket.setdefaulttimeout(600)
OLLAMA = "https://ollama-llama-349218458434.us-west1.run.app"
MODELS = ["qwen2.5:1.5b", "smollm2:1.7b", "gemma2:2b", "llama3.2:3b"]

def pull(m):
    payload = json.dumps({"model": m, "stream": True}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/pull", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    last = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            try:
                s = json.loads(line).get("status", "")
            except Exception:
                continue
            if s != last:
                print(f"  [{m}] {s[:70]}  (+{time.time()-t0:.0f}s)", flush=True)
                last = s
            if s == "success":
                return True
            if "error" in s.lower():
                return False
    return False

for m in MODELS:
    print(f"== pulling {m} ==", flush=True)
    ok = pull(m)
    print(f"== {m}: {'OK' if ok else 'FAILED'} ==", flush=True)

# verify
with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=60) as r:
    have = {x["name"] for x in json.loads(r.read()).get("models", [])}
print("\nNow available:", sorted(have))
print("Target present:", {m: (m in have) for m in MODELS})
