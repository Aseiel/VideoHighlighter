# Running Ollama on another machine

The app talks to Ollama over HTTP, so the server does not have to be the machine
you are editing on. A desktop with the GPU can answer for a laptop, and one
server can answer for several people.

## Point the app at it

**LLM panel → LLM Settings → Ollama host.** Type the address and press Enter:

```
192.168.1.118
```

That is enough — a bare address becomes `http://192.168.1.118:11434`, and the
field rewrites itself into the full URL so you can see what was understood. A
different port (`box.lan:12345`), a full URL, or a reverse proxy
(`https://ai.example.com/ollama`) all work; so does pasting an endpoint you had
in a terminal, `http://192.168.1.118:11434/api/generate` — the `/api/...` part is
dropped.

The model dropdown refills from that machine the moment the field is left, so
the tag you pick is one that server actually has. Leave the field empty to go
back to `http://localhost:11434`.

**One setting, whole app.** Chat, the report's model dialog, narration and the
advisor all ask the same place. There is deliberately no per-feature host: the
failure that causes is a model list read from one machine and a run sent to
another, which surfaces minutes later as "model not found" for a model you can
watch running.

## Without the UI

Set `OLLAMA_HOST` (Ollama's own variable — a machine already configured for a
remote server needs nothing said twice), or `VH_OLLAMA_HOST` to point only this
app somewhere:

```
set VH_OLLAMA_HOST=192.168.1.118
```

The host field wins over both, because a field you filled in is a decision you
can see and an inherited variable is not.

## The server side, which is where this usually goes wrong

**Ollama listens on localhost only by default.** Nothing on the network can
reach it, including this app, and the symptom is a connection that is refused
rather than one that hangs — the server is running and is simply not listening
where you are knocking.

On the machine running Ollama:

- **Windows:** set a system environment variable `OLLAMA_HOST` to `0.0.0.0`,
  then restart Ollama (quit it from the tray icon — it does not re-read the
  variable on its own).
- **Linux (systemd):** `sudo systemctl edit ollama.service`, add
  `Environment="OLLAMA_HOST=0.0.0.0"`, then `systemctl daemon-reload` and
  `systemctl restart ollama`.
- **macOS:** `launchctl setenv OLLAMA_HOST 0.0.0.0` and restart the app.

Then let TCP 11434 through that machine's firewall. On Windows that is an
inbound rule; the Windows Defender prompt only appears for programs that ask,
and Ollama does not.

Check it from the machine running VideoHighlighter before touching the app:

```
curl http://192.168.1.118:11434/api/tags
```

A JSON list of models means you are done. A refused connection or a timeout is
one of the two things above, not a problem with this app.

## What to expect over a network

- **Frames go over the wire.** Vision models are sent JPEG-encoded frames, so a
  run's traffic is roughly one image per analysed moment. On a LAN this is
  nothing; over a VPN or the internet it is the slowest part of the run.
- **`0.0.0.0` publishes to everything that can route to that machine**, with no
  authentication of any kind — Ollama has none. That is fine on a home LAN and
  is not fine on a network you do not control. For anything else, put it behind
  a reverse proxy that does the authentication and point the host field at the
  proxy.
- **The GPU that matters is the remote one.** Whatever this machine has is
  irrelevant to generation once the host is set; the model runs where Ollama
  does.
