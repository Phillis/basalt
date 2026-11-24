import json, time, urllib.request, random, string, sys, re
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
OUT = r"C:\LLM\rebase\b\server.out"
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(k))
def ask(msgs, sess, tok=60):
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"model":"Inkling-2-Mini","messages":msgs,"max_tokens":tok,"temperature":0}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key,"x-session-affinity":sess})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=3600) as h: d = json.load(h)
    m = d["choices"][0]["message"]
    return time.time()-t0, (m.get("content") or ""), (m.get("reasoning_content") or ""), d.get("usage", {})
def last_done():
    try: lines = open(OUT, encoding="utf-8", errors="replace").read().splitlines()
    except Exception: return None
    for ln in reversed(lines):
        m = re.search(r"s(\d+) done\s+pp=(\d+)/(\d+)c", ln)
        if m: return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None
mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
convs = {}
for tag in ("A","B"):
    convs[tag] = [{"role":"user","content":"Session %s. Read this and reply with one short sentence.\n\n%s" % (words(1), words(int(depth/3.3)))}]
prev = {"A":0,"B":0}
print("  harness=%s, ~%dk, real roles (assistant replies sent back), affinity on, 4 turns" % (mode, depth//1000))
for turn in range(1, 5):
    for tag in ("A","B"):
        convs[tag].append({"role":"user","content":"Turn %d: %s" % (turn, words(max(150, random.randint(500,2000)//3)))})
        wall, content, reasoning, u = ask(convs[tag], "conv-"+tag)
        # what a real client sends back
        if mode == "reasoning" and reasoning:
            convs[tag].append({"role":"assistant","content":content,"reasoning_content":reasoning})
        elif mode == "tool":
            convs[tag].append({"role":"assistant","content":content,
                               "tool_calls":[{"id":"c%d" % turn,"type":"function",
                                              "function":{"name":"read_file","arguments":json.dumps({"path":"notes.txt"})}}]})
            convs[tag].append({"role":"tool","tool_call_id":"c%d" % turn,"content":"notes: all good"})
        else:
            convs[tag].append({"role":"assistant","content":content})
        d = last_done(); slot, total, cached = (d[0], d[1], d[2]) if d else (-1,-1,-1)
        print("    turn %d %s: slot %s total %d cached %d  wall %.1f s" % (turn, tag, slot, total, cached, wall))
