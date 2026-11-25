import json, threading, time, urllib.request, random, string, sys
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(k))
def post(msgs, tok, sess, timeout=3600):
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"model":"Inkling-2-Mini","messages":msgs,"max_tokens":tok,"temperature":0}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key,"x-session-affinity":sess})
    with urllib.request.urlopen(r, timeout=timeout) as h: d = json.load(h)
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or ""), d.get("usage", {})
depth = int(sys.argv[1]); tok = int(sys.argv[2]) if len(sys.argv) > 2 else 200
nw = int(depth / 3.34)                      # words for the target token count
marks = [(int(nw*0.1), "alpha" + "".join(random.choice(string.ascii_lowercase) for _ in range(5))),
         (int(nw*0.5), "bravo" + "".join(random.choice(string.ascii_lowercase) for _ in range(5))),
         (int(nw*0.9), "delta" + "".join(random.choice(string.ascii_lowercase) for _ in range(5)))]
parts, last = [], 0
for at, w in marks:
    parts.append(words(at - last)); parts.append("\n\nNOTE: passphrase %s.\n\n" % w); last = at
parts.append(words(nw - last))
parts.append("\n\nQuestion: list the three passphrases, comma separated.")
body = "".join(parts)
out = {}
def dec():
    try: post([{"role":"user","content":"Count from 1 to 900, one number per line."}], 400, "dec")
    except Exception: pass
t = threading.Thread(target=dec); t.start(); time.sleep(0.8)
a, u = post([{"role":"user","content":body}], tok, "read")
got = [w for _, w in marks if w in a.lower()]
print("  %dk: %d/3 needles exact (prompt %s tokens, words %d)" % (depth//1000, len(got), u.get("prompt_tokens"), nw))
if len(got) < 3:
    print("    answer: %s" % a.replace("\n"," ")[:150])
t.join()
