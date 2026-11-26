import json, threading, time, urllib.request, random, string, re, sys
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
OUT = r"C:\LLM\rebase\b\server.out"
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(k))
def post(sess, msgs, tok, res, i, t0s):
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"model":"Inkling-2-Mini","messages":msgs,"max_tokens":tok,"temperature":0}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key,"x-session-affinity":sess})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=3600) as h: d = json.load(h)
    res[i] = (d.get("usage",{}).get("prompt_tokens",0), d.get("usage",{}).get("completion_tokens",0), time.time()-t0)
def engine_done(n=2):
    try: L = open(OUT, encoding="utf-8", errors="replace").read().splitlines()
    except Exception: return []
    got = []
    for ln in reversed(L):
        m = re.search(r"s(\d+) done\s+pp=(\d+)/(\d+)c.*tg=(\d+) ([\d.]+)t/s acc=(\d+)% dl=([\d.]+).*hit=([\d.]+%)", ln)
        if m: got.append(m.groups())
        if len(got) >= n: break
    return got
depth = int(sys.argv[1])
# fill each slot with a unique prompt of about `depth` tokens, then decode prose in both at once
fill = [{"role":"user","content":"Session %s. Read this.\n\n%s" % (words(1), words(max(50, int(depth/3.34))))} for _ in range(2)]
r1 = [None, None]
ts = [threading.Thread(target=post, args=("a" if i == 0 else "b", fill[i], 8, r1, i, None)) for i in range(2)]
for t in ts: t.start()
for t in ts: t.join()
ask = [{"role":"user","content":"Write 400 words of continuous prose about the sea and the weather tonight."}] * 2
r2 = [None, None]
t0 = time.time()
ts = [threading.Thread(target=post, args=("a" if i == 0 else "b", ask, 400, r2, i, None)) for i in range(2)]
for t in ts: t.start()
for t in ts: t.join()
wall = time.time() - t0
tot = sum(x[1] for x in r2 if x)
dl = engine_done(2)
print("  depth %dk: decode %.1f tok/s per stream (aggregate %.1f), acc %s, dl %s, hit %s" % (
    depth//1000, tot/wall/2, tot/wall,
    "/".join(d[5] for d in dl) or "?", "/".join(d[6] for d in dl) or "?", "/".join(d[7] for d in dl) or "?"))
