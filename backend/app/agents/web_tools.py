"""
Shared web navigation utilities used by the Web Navigator Agent.
"""
import time
import httpx
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from app import timing

FETCH_TIMEOUT = 10.0
_RETRY_DELAYS = [2, 4]  # waits between attempts: try → 2s → try → 4s → try → exit
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _extract_text_and_links(html: str, base_url: str) -> tuple[str, list[str]]:
    from html.parser import HTMLParser

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
            self.links: list[str] = []
            self._in_script = False

        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self._in_script = True
            if tag == "a":
                href = dict(attrs).get("href", "")
                if href and not href.startswith(("javascript:", "#", "mailto:")):
                    self.links.append(urljoin(base_url, href))

        def handle_endtag(self, tag):
            if tag == "script":
                self._in_script = False

        def handle_data(self, data):
            if not self._in_script:
                s = data.strip()
                if s:
                    self.parts.append(s)

    p = _Parser()
    p.feed(html)
    return " ".join(p.parts)[:5000], p.links[:30]


def _fetch_one(url: str) -> dict:
    last_reason = "unknown"
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            with httpx.Client(timeout=FETCH_TIMEOUT, headers=HEADERS, follow_redirects=True) as c:
                r = c.get(url)
                r.raise_for_status()
                text, links = _extract_text_and_links(r.text, url)
                return {"status": "ok", "url": str(r.url), "fetched_at": _now_utc(), "content": text, "links": links}
        except httpx.TimeoutException:
            last_reason = "timeout (>10s)"
        except httpx.HTTPStatusError as e:
            last_reason = f"HTTP {e.response.status_code}"
            break  # HTTP errors won't resolve with retries
        except Exception as e:
            last_reason = str(e)[:120]
    return {"status": "failed", "url": url, "reason": last_reason, "fetched_at": _now_utc()}


def _is_relevant(content: str, goal: str) -> bool:
    goal_words = set(goal.lower().split())
    content_lower = content.lower()
    return sum(1 for w in goal_words if len(w) > 3 and w in content_lower) >= 2


def _find_relevant_links(links: list[str], goal: str) -> list[str]:
    goal_words = goal.lower().split()
    scored = [(sum(1 for w in goal_words if len(w) > 3 and w in link.lower()), link) for link in links]
    return [l for s, l in sorted(scored, reverse=True) if s > 0][:5]


def web_navigate_impl(start_url: str, goal: str) -> dict:
    """
    Navigate the web starting from start_url to find content matching goal.
    Strategy: explicit URL → relevant links → root domain → admit failure with full trace.
    """
    trace = []
    attempted: set[str] = set()

    def try_url(url: str, step: str) -> dict | None:
        if url in attempted:
            return None
        attempted.add(url)
        timing.t(f"web_navigate [{step}]: {url[:70]}")
        result = _fetch_one(url)
        entry = {"step": step, "url": url, "status": result["status"]}
        if result["status"] == "failed":
            entry["reason"] = result["reason"]
        else:
            entry["outcome"] = "relevant" if _is_relevant(result["content"], goal) else "fetched_not_relevant"
            entry["links_found"] = len(result.get("links", []))
        trace.append(entry)
        return result if result["status"] == "ok" else None

    # Step 1: explicit URL
    r = try_url(start_url, "explicit_url")
    if r and _is_relevant(r["content"], goal):
        return {"status": "ok", "url": r["url"], "fetched_at": r["fetched_at"], "content": r["content"], "trace": trace}

    # Step 2: follow relevant links from start page
    if r and r.get("links"):
        for link in _find_relevant_links(r["links"], goal)[:3]:
            r2 = try_url(link, "link_from_start_page")
            if r2 and _is_relevant(r2["content"], goal):
                return {"status": "ok", "url": r2["url"], "fetched_at": r2["fetched_at"], "content": r2["content"], "trace": trace}

    # Step 3: root domain fallback
    parsed = urlparse(start_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root != start_url:
        r3 = try_url(root, "root_domain_fallback")
        if r3 and r3.get("links"):
            for link in _find_relevant_links(r3["links"], goal)[:3]:
                r4 = try_url(link, "link_from_root_page")
                if r4 and _is_relevant(r4["content"], goal):
                    return {"status": "ok", "url": r4["url"], "fetched_at": r4["fetched_at"], "content": r4["content"], "trace": trace}

    return {
        "status": "failed",
        "goal": goal,
        "start_url": start_url,
        "fetched_at": _now_utc(),
        "trace": trace,
        "message": (
            f"Could not find content for '{goal}' after {len(trace)} attempts. "
            f"Tried: {', '.join(t['url'][:50] for t in trace)}. "
            f"Reasons: {'; '.join(t.get('reason', t.get('outcome', '')) for t in trace)}."
        ),
    }
