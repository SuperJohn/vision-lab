"""boot ID + no-cache UI serving so open pages pick up server restarts.

BOOT_ID changes every server start; the page remembers the first one it sees and reloads when it
changes. UI files are served with no-cache and index.html version-busts its asset URLs with BOOT_ID."""
import re
import time
import uuid
from pathlib import Path

from fastapi.responses import HTMLResponse

BOOT_ID = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
NO_CACHE = "no-cache, must-revalidate"


def is_ui_path(path):
    return path == "/" or path.startswith("/static/")


def install_no_cache(app):
    @app.middleware("http")
    async def revalidate_ui(request, call_next):
        response = await call_next(request)
        if is_ui_path(request.url.path):
            response.headers["Cache-Control"] = NO_CACHE
        return response


def render_index(static_dir, boot_id=BOOT_ID):
    html = (Path(static_dir) / "index.html").read_text()
    html = re.sub(r'((?:src|href)="/static/[^"?]+\.(?:js|css))"', rf'\1?v={boot_id}"', html)
    return HTMLResponse(html, headers={"Cache-Control": NO_CACHE})


def with_boot_id(meta_json, boot_id=BOOT_ID):
    """Append boot_id to a JSON-object string without re-parsing it."""
    if not meta_json or not meta_json.rstrip().endswith("}"):
        return meta_json
    s = meta_json.rstrip()
    return s[:-1] + ("," if s[:-1].strip() not in ("{",) else "") + f'"boot_id":"{boot_id}"' + "}"
