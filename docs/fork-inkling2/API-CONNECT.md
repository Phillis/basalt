# Connecting an agent to Basalt from another machine

Verified against the running server (`ctx=196608`, `spec=8`, doorbell hand-off) on 2026-11-22.

---

## 1. What you need

| | |
|---|---|
| **Base URL (OpenAI style)** | `http://192.168.11.214:8080/v1` |
| **Base URL (Anthropic style)** | `http://192.168.11.214:8080` |
| **Model id** | `Inkling-2-Mini` |
| **API key** | first line of `C:\LLM\Basalt\api-key.txt` on the server (48 hex chars) |
| **Auth header** | `Authorization: Bearer <key>` **or** `x-api-key: <key>` |
| **Context window** | 196,608 tokens |
| **Bind / firewall** | listens on `0.0.0.0:8080`; inbound 8080 already allowed |

The key gates `/v1/*` **and** `/metrics`. The web UI at `/` and the stats page are unauthenticated â€” fine on a trusted LAN,
not something to expose to the internet.

---

## 2. Endpoints this build answers

| Method | Path | API | Verified |
|---|---|---|---|
| POST | `/v1/chat/completions` | OpenAI Chat Completions | âœ… 200, streaming (`"stream": true`) works |
| POST | `/v1/responses` | OpenAI Responses API | âœ… 200 |
| POST | `/v1/messages` | Anthropic Messages API | âœ… 200 |
| GET | `/v1/models` | model list + status | âœ… 200 |
| GET | `/health` | liveness | âœ… 200 |
| GET | `/metrics` | throughput, experts, GPU (needs key) | âœ… 200 |
| GET | `/slots` | per-slot state | âœ… 200 |
| GET | `/` | chat UI / stats page | âœ… 200 |

`GET /v1/chat/completions` is 404 â€” normal, those are POST-only.

---

## 3. Smoke test from the other machine

```bash
# bash
KEY=<paste the key>
curl -s http://192.168.11.214:8080/v1/models -H "Authorization: Bearer $KEY"

curl -s http://192.168.11.214:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"Inkling-2-Mini",
       "messages":[{"role":"user","content":"Say hello in five words."}],
       "max_tokens":200,"temperature":0}'
```

```powershell
# PowerShell
$K = "<paste the key>"
Invoke-RestMethod 'http://192.168.11.214:8080/v1/chat/completions' -Method Post `
  -Headers @{ Authorization = "Bearer $K" } -ContentType 'application/json' `
  -Body '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"hi"}],"max_tokens":200}'
```

A healthy reply looks like:

```json
{"model":"Inkling-2-Mini",
 "choices":[{"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":26,"completion_tokens":40,"total_tokens":66,
          "prompt_tokens_details":{"cached_tokens":0}}}
```

---

## 4. Wiring it into an agent

**OpenAI Python SDK** â€” pass `base_url` with the `/v1` suffix:

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.11.214:8080/v1", api_key="<key>")
r = client.chat.completions.create(
    model="Inkling-2-Mini",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=2048, temperature=0,
)
```

**Anthropic SDK** â€” no `/v1` suffix (the SDK appends `/v1/messages`):

```python
from anthropic import Anthropic
c = Anthropic(base_url="http://192.168.11.214:8080", api_key="<key>")
m = c.messages.create(model="Inkling-2-Mini", max_tokens=2048,
                      messages=[{"role": "user", "content": "Hello"}])
```

**Anything else** (LangChain, Continue, OpenCode, Claude Code, LM Studio-style clients, `curl`-based tools): point it at
`http://192.168.11.214:8080/v1` with any non-empty API key, model id `Inkling-2-Mini`. It speaks OpenAI Chat
Completions and the Responses API, so most "OpenAI-compatible" integrations work unchanged.

---

## 5. Things worth knowing before an agent hammers it

- **The model writes thinking blocks.** Even at `reasoning_effort: "none"` it emits a short one, so give it room:
  `max_tokens` in the thousands, not tens. Verified: `reasoning_effort:"none"` + `max_tokens:200` returns `"4"` for
  "2+2".
- **`reasoning_effort`** per request: `none`, `minimal`, `low`, `medium` (server default), `high`, `xhigh`, `max`.
  Higher = longer thinking = slower replies.
- **Sampling** accepted per request: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
  `frequency_penalty`, `repetition_penalty`, `penalty_last_n`, `seed`, plus `logit_bias` and
  `response_format` (`json_object` / `json_schema`; the engine holds tool calls and JSON to the schema in-engine).
- **`relax`** (0..1) is this build's extra key: relaxed draft acceptance for sampled requests. Off by default; it did not
  help throughput here, so leave it out.
- **One request at a time.** The engine runs `parallel: 1`, so an agent that fans out concurrent calls will queue them
  behind one another â€” serialise, or expect queuing rather than errors.
- **Speed:** ~63â€“68 tok/s decode, ~54 tok/s sustained on long generations, ~1.5k tok/s prefill. A 16k-token prompt takes
  roughly 10 s to read before the first token.
- **Long context works:** retrieval verified at 130,709 tokens, within the 196,608 window.
- If the server is not up, start it on the host with `C:\LLM\Basalt\Start-Basalt.cmd` (it loads 18.4 GiB, ~40 s).

---

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| `401 missing or wrong API key` | no/incorrect `Authorization: Bearer` (or `x-api-key`) |
| connection refused | server not running, or `host` not `0.0.0.0` in `basalt_config.json` |
| replies that stop early / `finish_reason: "length"` | `max_tokens` too small â€” the thinking block consumed it |
| `400` mentioning the context | prompt + `max_tokens` exceeds 196,608 |
| first token takes seconds | prompt prefill (~1.5k tok/s); many agents re-read the system prompt â€” the server caches it (`prompt-cache: 6`) |
| model id `404` | use exactly `Inkling-2-Mini`; `GET /v1/models` lists what is served |
