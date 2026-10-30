"""Stage 4 of smoke.sh: send one tiny image to ollama /api/chat."""
import base64
import io
import json
import sys
import urllib.request

from PIL import Image

MODEL = "qwen3-vl:8b-instruct"
buf = io.BytesIO()
Image.new("RGB", (128, 128), (220, 30, 30)).save(buf, "PNG")
body = {"model": MODEL, "stream": False, "options": {"temperature": 0},
        "messages": [{"role": "user", "content": "What is the dominant colour of this image? One word.",
                      "images": [base64.b64encode(buf.getvalue()).decode()]}]}
req = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        reply = json.loads(r.read())["message"]["content"]
except Exception as e:
    print("VLM ping failed:", repr(e), file=sys.stderr)
    sys.exit(1)
print("VLM reply:", reply.strip()[:100])
