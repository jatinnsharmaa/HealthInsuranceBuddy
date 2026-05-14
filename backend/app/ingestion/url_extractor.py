"""Extract deferred URLs from policy markdown at ingestion time."""
import re
from typing import TypedDict

# Patterns that indicate the policy is deferring to a live URL
DEFERRAL_PATTERNS = [
    r"(?:visit|refer to|check|available at|available on|see)\s+(?:the\s+)?(?:company['’]?s?\s+)?(?:website\s+at\s+)?(?:https?://|www\.)\S+",
    r"(?:https?://|www\.)\S+",
]

CONTEXT_WINDOW = 200  # chars before URL to extract governing context


class DeferredUrl(TypedDict):
    url: str
    governs: str
    clause: str
    page_number: int


def extract_deferred_urls(pages: list[dict]) -> list[DeferredUrl]:
    """
    Scan parsed page texts for URLs the policy defers to for live data.
    Returns a list of DeferredUrl dicts.
    """
    results: list[DeferredUrl] = []
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    clause_pattern = re.compile(r"(?:Clause|Section|Annexure)\s+[\d.IVX]+", re.IGNORECASE)

    seen_urls: set[str] = set()

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        for match in url_pattern.finditer(text):
            raw_url = match.group(0).rstrip(".,;)")
            # Normalise
            url = raw_url if raw_url.startswith("http") else f"https://{raw_url}"

            if url in seen_urls:
                continue
            seen_urls.add(url)

            start = max(0, match.start() - CONTEXT_WINDOW)
            context = text[start : match.end()]

            # Extract nearest clause reference for citation
            clause_matches = list(clause_pattern.finditer(context))
            clause = clause_matches[-1].group(0) if clause_matches else "Unknown"

            # Summarise what the URL governs from surrounding text
            governs = _infer_governs(context)

            results.append(
                DeferredUrl(
                    url=url,
                    governs=governs,
                    clause=clause,
                    page_number=page_number,
                )
            )

    return results


def _infer_governs(context: str) -> str:
    """Infer what a URL governs from surrounding text."""
    lower = context.lower()
    if "excluded hospital" in lower or "blacklist" in lower:
        return "excluded_hospitals"
    if "network" in lower and "hospital" in lower:
        return "network_hospitals"
    if "cashless" in lower:
        return "cashless_eligibility"
    if "grievance" in lower:
        return "grievance_officer"
    if "ombudsman" in lower:
        return "ombudsman_contacts"
    if "smart select" in lower:
        return "smart_select_network"
    if "irdai" in lower or "irda" in lower:
        return "irdai_regulation"
    return "general"
