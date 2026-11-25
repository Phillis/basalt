import json, threading, time, urllib.request, random, string, sys, re
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
OUT = r"C:\LLM\rebase\b\server.out"
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(k))
def post(sess, out, i, prompt, tok):
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"model":"Inkling-2-Mini","messages":[{"role":"user","content":prompt}],
                         "max_tokens":tok,"temperature":0}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key,"x-session-affinity":sess})
    t0=time.time()
    with urllib.request.urlopen(r, timeout=3600) as h: d=json.load(h)
    out[i]=(d.get("usage",{}).get("completion_tokens",0), time.time()-t0)
def done_lines(n=2):
    try: L=open(OUT,encoding="utf-8",errors="replace").read().splitlines()
    except Exception: return []
    got=[]
    for ln in reversed(L):
        m=re.search(r"s(\d+) done\s+pp=(\d+)/(\d+)c.*tg=(\d+) ([\d.]+)t/s acc=(\d+)% dl=([\d.]+).*hit=([\d.]+%)", ln)
        if m: got.append(m.groups())
        if len(got)>=n: break
    return got
phase=sys.argv[1]
if phase=="A":
    prompt="Write 400 words of continuous prose about the sea and the weather tonight."
    out=[None,None]; ts=[threading.Thread(target=post,args=("a",out,0,prompt,400)), threading.Thread(target=post,args=("b",out,1,prompt,400))]
    t0=time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall=time.time()-t0
    tot=sum(o[0] for o in out if o)
    dl=done_lines(2)
    accs="/".join(d[5] for d in dl) if dl else "?"
    hits="/".join(d[7] for d in dl) if dl else "?"
    print("  A: %.1f tok/s per stream (aggregate %.1f), acc %s, hit %s" % (tot/wall/2, tot/wall, accs, hits))
elif phase=="B":
    w=int(30000/3.3)
    out=[None,None]
    ts=[threading.Thread(target=post,args=("a",out,0,"Session %s.\n\n%s"%(words(1),words(w)),8)), threading.Thread(target=post,args=("b",out,1,"Session %s.\n\n%s"%(words(1),words(w)),8))]
    t0=time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall=time.time()-t0
    print("  B: prefill %.0f t/s per stream (30k each, aggregate %.0f)" % ((2*30000)/wall/2, (2*30000)/wall))
