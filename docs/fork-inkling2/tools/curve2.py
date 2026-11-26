import json, threading, time, urllib.request, random, string
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(k))
def ask(prompt, tok, res, i):
    try:
        r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps({"model":"Inkling-2-Mini","messages":[{"role":"user","content":prompt}],
                             "max_tokens":tok,"temperature":0}).encode(),
            headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
        t0 = time.time()
        with urllib.request.urlopen(r, timeout=3600) as h: d = json.load(h)
        u = d.get("usage", {})
        res[i] = (u.get("prompt_tokens",0), u.get("completion_tokens",0), time.time()-t0)
    except Exception as e:
        res[i] = (-1, -1, 1.0)
        print("    ERR", str(e)[:90])
def pair(prompt_fn, tok):
    res = [None, None]
    ts = [threading.Thread(target=ask, args=(prompt_fn(i), tok, res, i)) for i in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    return res
print("  depth   prefill/stream   decode/stream   (both slots)")
for depth in (30000, 64000, 96000, 120000):
    w = max(64, int(depth / 3.3))
    pp = pair(lambda i: "Session %s.\n\n%s" % (words(1), words(w)), 4)
    rate = [p[0]/p[2] for p in pp if p and p[0] > 0]
    tg = pair(lambda i: "Count from 1 to 900, one number per line. (%s)" % words(1), 400)
    dr = [g[1]/g[2] for g in tg if g and g[1] > 0]
    print("  %6dk  %8.0f t/s   %8.1f t/s   (%s)" % (
        depth//1000, sum(rate)/len(rate) if rate else 0, sum(dr)/len(dr) if dr else 0,
        "failed" if not rate or not dr else "ok"))
