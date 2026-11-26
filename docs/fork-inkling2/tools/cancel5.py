import http.client, json, threading, time, urllib.request, random, string
key = open(r"C:\LLM\Basalt\api-key.txt").read().strip()
def words(k): return " ".join("".join(random.choice(string.ascii_lowercase) for _ in range(7)) for _ in range(k))
def body(msgs, tok):
    return json.dumps({"model":"Inkling-2-Mini","messages":msgs,"max_tokens":tok,"temperature":0})
def conn(sess):
    return http.client.HTTPConnection("127.0.0.1", 8080, timeout=900)
def send(c, msgs, sess, tok):
    c.request("POST", "/v1/chat/completions", body(msgs, tok),
              {"Content-Type":"application/json","Authorization":"Bearer "+key,"x-session-affinity":sess})
witness = [{"role":"user","content":"Witness session. Read and answer briefly.\n\n" + words(6000)}]
results = {"witness_turns":0, "errors":[]}
stop = threading.Event()
def witness_loop():
    n = 0
    while not stop.is_set():
        n += 1
        witness.append({"role":"user","content":"Witness turn %d: %s" % (n, words(400))})
        try:
            c = conn("conv-W")
            send(c, witness, "conv-W", 40)
            r = c.getresponse()
            if r.status != 200: results["errors"].append("witness HTTP %d" % r.status)
            else:
                d = json.loads(r.read())
                m = d["choices"][0]["message"]
                witness.append({"role":"assistant","content":(m.get("content") or "ok")})
                results["witness_turns"] += 1
            c.close()
        except Exception as e:
            results["errors"].append("witness %s: %s" % (type(e).__name__, e))
            break
        time.sleep(0.5)
t = threading.Thread(target=witness_loop, daemon=True)
t.start()
time.sleep(6)   # let the witness get into steady decode
print("  cancels (client disappears mid-generation), witness decoding throughout:")
for i in range(1, 6):
    victim = [{"role":"user","content":"Victim %d. Read this and write a long essay.\n\n" % i + words(3000)}]
    c = conn("conv-C")
    try:
        send(c, victim, "conv-C", 4000)
        time.sleep(2.0)      # let it start generating
        c.close()            # the client is gone
        print("    cancel %d sent (connection closed after 2 s)" % i, flush=True)
    except Exception as e:
        print("    cancel %d failed to even start: %s" % (i, e))
    time.sleep(1.0)
time.sleep(8)
stop.set(); t.join(timeout=120)
print("  witness turns completed: %d, errors: %s" % (results["witness_turns"], results["errors"][:3]))
# both slots must still answer, cleanly, after the cancels
def ask(sess, msg):
    c = conn(sess); send(c, [{"role":"user","content":msg}], sess, 40)
    r = c.getresponse(); d = json.loads(r.read()); c.close()
    m = d["choices"][0]["message"]
    return (m.get("content") or "") or (m.get("reasoning_content") or "")
print("  after the cancels, slot C:", repr(ask("conv-C", "What is 17*23? Number only.")[:40]))
print("  after the cancels, slot W:", repr(ask("conv-W", "What is 144/12? Number only.")[:40]))
