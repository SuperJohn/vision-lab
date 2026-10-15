"""Vision Lab redaction pack (Security Agent). Fail-closed: any error => reject."""
import re, json

# ---- App / window denylist: desktop samples matching these are REJECTED (never blurred) ----
APP_DENYLIST = {
    "Messages","Mail","Contacts","FaceTime","Phone","1Password","Keychain Access","Passwords",
    "System Settings","Terminal","iTerm2","Warp","Wallet","Stocks","Numbers","Linear","Calendar",
    "Notes","Signal","WhatsApp","Slack","Photos","Preview","Finder",
}
TITLE_DENY = re.compile(r"(?i)(bank|capital ?one|discover|toyota|chase|wells|schwab|fidelity|venmo|paypal|"
    r"x money|edd|unemployment|ca\.gov|irs|tax|settlement|divorce|custody|attorney|court|kaiser|mychart|"
    r"health|insurance|password|login|sign in|2fa|verification|\.env|secret|token|gmail|inbox|linear|"
    r"x\.com/messages|grok\.com/chat)")

# ---- Text regexes: hits in OCR / captions / transcripts ----
PII_PATTERNS = {
    "phone": r"(?<!\d)(?:\+1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)|\+1\d{10}(?!\d)",
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "card": r"\b(?:\d[ -]?){13,19}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "money": r"\$\s?\d[\d,]*(?:\.\d{2})?",
    "zip_addr": r"\b\d{2,5}\s+\w+(?:\s\w+)?\s(?:st|street|ave|rd|road|dr|drive|ct|way|blvd|ln|pl)\b",
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "gh_token": r"gh[pousr]_[A-Za-z0-9]{36,}",
    "openai_like": r"\b(?:sk|xai|pk|rk)-[A-Za-z0-9_-]{20,}",
    "slack": r"xox[abpr]-[A-Za-z0-9-]{10,}",
    "google_key": r"AIza[0-9A-Za-z_-]{35}",
    "jwt": r"eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "env_assign": r"(?i)\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASS)\s*[=:]\s*\S{6,}",
    "url_token": r"(?i)[?&](?:token|key|code|sig|auth)=[^&\s]{8,}",
    "otp": r"(?i)(?:code|otp|verification)\D{0,10}\b\d{6}\b",
}
PII_RE = {k: re.compile(v, re.I) for k, v in PII_PATTERNS.items()}

# Hard-block terms: any hit in caption/transcript/detections => REJECT item
HARD_BLOCK = re.compile(r"(?i)\b(gun|handgun|pistol|firearm|weapon|revolver|rifle|holster|ammo|ammunition|magazine clip|"
    r"son|kid|child|boy|toddler|minor|school|"
    r"terminated|termination|fired|lawsuit|settlement|divorce|custody|edd)\b")

# Personal names are kept out of the repo: one term per line in data/redaction_terms.txt (local, untracked)
import os as _os
_TERMS_PATH = _os.environ.get("VLAB_REDACT_TERMS") or _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "redaction_terms.txt")
_LOCAL_TERMS = [t.strip() for t in open(_TERMS_PATH).read().splitlines() if t.strip() and not t.startswith("#")] if _os.path.exists(_TERMS_PATH) else []
LOCAL_BLOCK = re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in _LOCAL_TERMS) + r")\b") if _LOCAL_TERMS else None

# HOU-88: personal places/employers/ZIPs are also kept out of the repo: one term per line in
# data/redaction_title_terms.txt (local, untracked). They deny window titles and are masked as [LOCAL_TERM] in text.
_TITLE_TERMS_PATH = _os.environ.get("VLAB_REDACT_TITLE_TERMS") or _os.path.join(_os.path.dirname(_TERMS_PATH), "redaction_title_terms.txt")
_TITLE_TERMS = [t.strip() for t in open(_TITLE_TERMS_PATH).read().splitlines() if t.strip() and not t.startswith("#")] if _os.path.exists(_TITLE_TERMS_PATH) else []
LOCAL_TITLE_DENY = re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in _TITLE_TERMS) + r")\b") if _TITLE_TERMS else None
if LOCAL_TITLE_DENY:
    PII_RE["local_term"] = LOCAL_TITLE_DENY

def scan_text(text: str):
    text = text or ""
    hits = [k for k, r in PII_RE.items() if r.search(text)]
    block = HARD_BLOCK.findall(text) + (LOCAL_BLOCK.findall(text) if LOCAL_BLOCK else [])
    return {"pii": hits, "hard_block": sorted(set(w.lower() for w in block))}

def mask_text(text: str):
    for k, r in PII_RE.items():
        text = r.sub(f"[{k.upper()}]", text)
    return text

WEAPON_CLS = {"knife","gun","handgun","pistol","rifle","weapon","firearm"}

def _strings(obj):
    """Yield string leaves only (numeric JSON fields like timestamps are skipped)."""
    if isinstance(obj, str): yield obj
    elif isinstance(obj, dict):
        for v in obj.values(): yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj: yield from _strings(v)

def classify_event(ev: dict):
    """ev: row from events (kind, text, modality, data JSON). Returns (status, reasons)."""
    try:
        data = ev.get("data") or {}
        if isinstance(data, str): data = json.loads(data)
        reasons = []
        if ev.get("modality") == "desktop":
            app = data.get("app", ""); title = data.get("title", "") or data.get("window_title", "")
            if app in APP_DENYLIST: reasons.append(f"app:{app}")
            if TITLE_DENY.search(title) or (LOCAL_TITLE_DENY and LOCAL_TITLE_DENY.search(title)): reasons.append("title_deny")
        s = scan_text(" ".join([ev.get("text") or "", ev.get("vlm_text") or ""] + list(_strings(data))))
        if s["hard_block"]: reasons += [f"block:{w}" for w in s["hard_block"]]
        if str(ev.get("cls") or "").lower() in WEAPON_CLS: reasons.append(f"det:{ev['cls']}")
        dets = data.get("detections", [])
        if isinstance(dets, str):
            dets = json.loads(dets or "[]")
        for d in dets if isinstance(dets, list) else []:
            lab = str(d.get("cls", d.get("label", d.get("class", "")))).lower()
            if lab in WEAPON_CLS: reasons.append(f"det:{lab}")
            if d.get("age_class") == "child" or (d.get("age_est") or 99) < 18: reasons.append("det:minor")
        if reasons: return "rejected", reasons
        if s["pii"]: return "auto", [f"pii:{p}" for p in s["pii"]]   # redact then human review
        return "pending_review", []
    except Exception as e:
        return "rejected", [f"error:{e!r}"]

MIGRATION_TABLES = ["events", "utterances", "keyframes", "batch_annotations"]
MIGRATION_COLS = [("publishable", "INTEGER NOT NULL DEFAULT 0"),
                  ("redaction_status", "TEXT NOT NULL DEFAULT 'pending'"),
                  ("redaction_reasons", "TEXT")]

def migrate(conn):
    """Idempotent: adds only missing columns. Export rule: publishable=1 AND redaction_status='reviewed'."""
    for t in MIGRATION_TABLES:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
        for c, typ in MIGRATION_COLS:
            if c not in have:
                conn.execute(f"ALTER TABLE {t} ADD COLUMN {c} {typ}")
    conn.commit()
