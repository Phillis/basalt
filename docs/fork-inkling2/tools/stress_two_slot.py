import json, threading, time, urllib.request, random, string, sys
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(k))
def ask(prompt, tok):
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"model":"Inkling-2-Mini","messages":[{"role":"user","content":prompt}],"max_tokens":tok,"temperature":0}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
    t0=time.time()
    with urllib.request.urlopen(r, timeout=1800) as h: d=json.load(h)
    u=d.get("usage",{})
    return u.get("prompt_tokens",0), u.get("completion_tokens",0), time.time()-t0
rounds = int(sys.argv[1]) if len(sys.argv)>1 else 3
for rnd in range(rounds):
    res = {}
    def decoder():
        res["a"] = ask("Count from 1 to 900, one number per line, no other text.", 600)
    def prompter():
        time.sleep(0.8)                      # start mid-decode: the crash scenario
        res["b"] = ask("Session %s.\n\n%s" % (words(1), words(6000)), 120)   # ~20k-token prompt + generation
    ts = [threading.Thread(target=decoder), threading.Thread(target=prompter)]
    t0=time.time(); [t.start() for t in ts]; [t.join() for t in ts]; w=time.time()-t0
    a, b = res.get("a"), res.get("b")
    print("  round %d: decode-session %s | prompt-session %s | wall %.1f s" % (
        rnd+1,
        ("%dtok %dtok %.1f tok/s" % (a[0], a[1], a[1]/a[2])) if a else "FAILED",
        ("pp=%d %dtok %.0f t/s" % (b[0], b[1], b[0]/b[2])) if b else "FAILED", w))
