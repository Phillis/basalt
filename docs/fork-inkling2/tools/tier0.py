import json, socket, threading, time, urllib.request, random, string, sys
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
HOST, PORT = "127.0.0.1", 8080
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(k))
def post(msgs, tok=60, sess="t", timeout=3600):
    r = urllib.request.Request("http://%s:%d/v1/chat/completions" % (HOST, PORT),
        data=json.dumps({"model":"Inkling-2-Mini","messages":msgs,"max_tokens":tok,"temperature":0}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key,"x-session-affinity":sess})
    with urllib.request.urlopen(r, timeout=timeout) as h: d = json.load(h)
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or ""), d.get("usage", {})
mode = sys.argv[1]
if mode == "needles":
    for depth in (64000, 120000):
        marks = [(int(depth*0.1), "alpha" + "".join(random.choice(string.ascii_lowercase) for _ in range(5))),
                 (int(depth*0.5), "bravo" + "".join(random.choice(string.ascii_lowercase) for _ in range(5))),
                 (int(depth*0.9), "delta" + "".join(random.choice(string.ascii_lowercase) for _ in range(5)))]
        per = max(1, depth*5 // (len(marks)+1))
        body = ""; last = 0
        for at, w in marks:
            body += words(max(1, (at-last)//6)) + "\n\nNOTE: passphrase %s.\n\n" % w
            last = at
        body += words(max(1, (depth-last)//6)) + "\n\nQuestion: list the three passphrases, comma separated."
        out = {}
        def dec():
            try: post([{"role":"user","content":"Count from 1 to 900, one number per line."}], 400, "dec")
            except Exception as e: out["e"] = str(e)[:40]
        t = threading.Thread(target=dec); t.start(); time.sleep(0.8)
        try:
            a, u = post([{"role":"user","content":body}], 80, "read")
            got = [w for _, w in marks if w in a.lower()]
            print("  %dk: %d/%d needles exact (prompt %s tokens)" % (depth//1000, len(got), len(marks), u.get("prompt_tokens")))
        except Exception as e:
            print("  %dk: ERROR %s" % (depth//1000, str(e)[:70]))
        t.join()
else:
    # cancel: open the socket ourselves, read a little, close it
    def cancel_once(i):
        body = json.dumps({"model":"Inkling-2-Mini","messages":[{"role":"user","content":"Write 800 words about the sea."}],
                           "max_tokens":700,"temperature":0,"stream":True})
        s = socket.create_connection((HOST, PORT), timeout=30)
        s.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\n"
                   "Content-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (HOST, key, len(body), body)).encode())
        time.sleep(1.0)
        s.close()
    out = {}
    def dec():
        try: out["r"] = post([{"role":"user","content":"Count from 1 to 900, one number per line."}], 500, "dec")
        except Exception as e: out["e"] = str(e)[:40]
    t = threading.Thread(target=dec); t.start(); time.sleep(0.5)
    for i in range(5):
        cancel_once(i); time.sleep(0.4)
    t.join()
    ok = "r" in out
    try:
        a, u = post([{"role":"user","content":"What is 17*23?"}], 60, "t")
        after = "391" in a
    except Exception as e:
        after = False
    print("  cancel x5 with slot 0 decoding: decoder %s, next request %s" % ("intact" if ok else "BROKEN", "works" if after else "BROKEN"))
