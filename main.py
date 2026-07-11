import json, re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import config

app = FastAPI()

# Rule 2: CORS must be wide open so the grader (calling from a Cloudflare Worker
# or any other origin) is never blocked by the browser's same-origin policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

HEADERS = {
    "Authorization": f"Bearer {config.AIPIPE_TOKEN}",
    "Content-Type": "application/json",
}

# Small in-memory cache: if the grader (or you, while testing) sends the same
# input twice, we don't pay for/re-run the model call a second time.
_CACHE = {}


async def chat(messages, model=None, max_tokens=800, force_json=True):
    """Call AIPipe's chat/completions endpoint with a plain text prompt
    (no image) and return the raw text content of the model's reply."""
    key = json.dumps({"m": model, "msgs": messages}, sort_keys=True, default=str)
    if key in _CACHE:
        return _CACHE[key]

    body = {
        "model": model or config.TEXT_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if force_json:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(f"{config.AIPIPE_BASE}/chat/completions",
                               headers=HEADERS, json=body)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"]

    _CACHE[key] = out
    return out


def parse_json(s: str) -> dict:
    """The model sometimes wraps JSON in ```json ... ``` fences — strip those,
    then fall back to regex-extracting the {...} block if needed."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def normalize_answer(ans) -> str:
    """
    Rule 1: numeric answers must be returned as a bare number string —
    no currency symbols, no thousands separators, no units.
    Text answers (e.g. 'Marketing' as the biggest pie slice) are returned as-is.
    """
    s = str(ans).strip()
    if not s:
        return s

    # Strip spaces/commas/currency symbols to test if this is really a number.
    cleaned = re.sub(r"[,\s]", "", s)
    cleaned = re.sub(r"[₹$€£%]", "", cleaned)
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)

    # Only treat it as numeric if the ORIGINAL string was basically just
    # digits/symbols (so we don't mangle a text answer that happens to
    # contain a number, e.g. "Q3 2024").
    if m and re.fullmatch(r"[^\dA-Za-z]*-?\d[\d,.\s₹$€£%]*", s):
        num = m.group(0)
        if "." in num:              # 240.0 -> 240 (drop trailing zeros/dot)
            num = num.rstrip("0").rstrip(".")
        return num
    return s


@app.get("/")
async def root():
    # Simple health check so you can confirm the deploy is live in a browser.
    return {"ok": True, "email": config.EMAIL}


@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    img_b64 = body.get("image_base64", "")
    question = body.get("question", "")

    prompt = (
        "You read charts, receipts, tables, invoices and pie charts EXACTLY.\n"
        "Work in steps in a 'work' field, then give the final 'answer':\n"
        "1. TRANSCRIBE every relevant label and number you see, one by one "
        "(e.g. each bar's value, each receipt line, each table cell). Read "
        "digits carefully; do not round or estimate.\n"
        "2. If the question needs arithmetic (sum of all bars, grand total, "
        "max/min of a column, total including tax), compute it step by step "
        "and DOUBLE-CHECK the sum by re-adding.\n"
        "3. Final 'answer': if NUMERIC, output ONLY the bare number — no "
        "currency symbol, no thousands separators, no units, no words. Keep "
        "decimals exactly as shown (e.g. a money total 4089.35 stays 4089.35). "
        "If TEXT (e.g. the largest pie category), output it EXACTLY as written "
        "in the image.\n"
        "Return JSON: {\"work\": \"...\", \"answer\": \"...\"}.\n"
        f"Question: {question}"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high",   # high detail so small chart/receipt text is legible
                },
            },
        ],
    }]

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                f"{config.AIPIPE_BASE}/chat/completions",
                headers=HEADERS,
                json={
                    "model": config.VISION_MODEL,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 1200,
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

        out = parse_json(content)
        ans = normalize_answer(out.get("answer", ""))
    except Exception:
        # Never let the endpoint crash / hang — always return valid JSON.
        ans = ""

    return {"answer": str(ans)}


# ================= Q3: /extract (invoice text -> fixed fields) =================
@app.post("/extract")
async def extract(request: Request):
    body = await request.json()
    text = body.get("invoice_text", "")

    prompt = (
        "Extract these fields from the invoice text and return JSON with "
        "EXACTLY these keys: invoice_no, date, vendor, amount, tax, currency.\n"
        "- date: ISO YYYY-MM-DD\n"
        "- amount: the SUBTOTAL before tax, as a plain number (no separators)\n"
        "- tax: the tax amount only, as a plain number\n"
        "- currency: ISO code (INR, USD, EUR...)\n"
        "- use null if a field is not present.\n\n"
        f"TEXT:\n{text}"
    )

    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}]))
    except Exception as e:
        # TEMP DEBUG: show the real error. Remove this except block once fixed.
        keys = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
        result = {k: None for k in keys}
        result["debug_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return result

    keys = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
    return {k: out.get(k) for k in keys}
