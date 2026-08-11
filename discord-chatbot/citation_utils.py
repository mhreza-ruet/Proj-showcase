from typing import List

from storage import Chunk


def format_ref(c: Chunk) -> str:
    m = c.meta or {}

    if m.get("url"):
        return m["url"]

    if m.get("source") == "pdf_text":
        page = m.get("page")
        title = m.get("doc_title") or m.get("file_name") or "PDF"
        return f"{title} (page {page})" if page else f"{title} (PDF)"

    if m.get("source") == "notebooklm":
        title = m.get("doc_title") or m.get("file_name") or "Notes"
        return f"{title} (NotebookLM notes)"

    return "(unknown source)"


def build_refs_from_chunks(chunks: List[Chunk]) -> str:
    seen = set()
    refs = []
    for c in chunks:
        ref = format_ref(c)
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return "\n".join(f"- {r}" for r in refs) if refs else "- (no reference found)"


def _source_group_key(c: Chunk) -> str:
    """
    Group chunks that come from the same underlying source so we do not
    cite three near-duplicates from one message / one PDF page / one note file.
    """
    m = c.meta or {}

    if m.get("message_id"):
        return f"discord:{m.get('message_id')}"

    if m.get("source") == "pdf_text":
        title = m.get("doc_title") or m.get("file_name") or "pdf"
        page = m.get("page") or "?"
        return f"pdf:{title}:page:{page}"

    if m.get("source") == "notebooklm":
        title = m.get("doc_title") or m.get("file_name") or "notes"
        return f"notes:{title}"

    url = m.get("url")
    if url:
        return f"url:{url}"

    return f"fallback:{id(c)}"


def _source_type(c: Chunk) -> str:
    m = c.meta or {}
    if m.get("message_id") or m.get("channel"):
        return "announcements"
    if m.get("source") == "pdf_text":
        return "pdf"
    if m.get("source") == "notebooklm":
        return "notes"
    return "other"


def _citation_priority(c: Chunk, *, prefer_recent_announcements: bool = False) -> tuple:
    """
    Higher priority = better citation candidate.
    Sort key is descending by:
      1) announcement preference (if enabled)
      2) recency among announcements (if enabled)
      3) semantic score
    """
    m = c.meta or {}
    src_type = _source_type(c)
    created_ts = int(m.get("created_ts", 0) or 0)

    announcement_bonus = 1 if (prefer_recent_announcements and src_type == "announcements") else 0
    recency_bonus = created_ts if (prefer_recent_announcements and src_type == "announcements") else 0

    return (announcement_bonus, recency_bonus, c.score)


def select_citations(
    chunks: List[Chunk],
    *,
    prefer_recent_announcements: bool = False,
    max_cites: int = 3,
    cite_min_similarity: float = 0.0,
) -> List[int]:
    """
    Return 1-based chunk indices.

    Goals:
    - keep only chunks above cite_min_similarity
    - prefer recent announcements for deadline/latest queries
    - avoid multiple citations from the same exact source group
    - encourage source diversity before taking duplicates
    """
    strong = [
        (idx, chunk)
        for idx, chunk in enumerate(chunks, start=1)
        if chunk.score >= cite_min_similarity
    ]
    if not strong:
        return []

    ranked = sorted(
        strong,
        key=lambda x: _citation_priority(
            x[1],
            prefer_recent_announcements=prefer_recent_announcements,
        ),
        reverse=True,
    )

    selected: List[int] = []
    used_groups = set()
    used_source_types = set()

    for idx, chunk in ranked:
        if len(selected) >= max_cites:
            break

        gkey = _source_group_key(chunk)
        stype = _source_type(chunk)

        if gkey in used_groups:
            continue

        if stype not in used_source_types:
            selected.append(idx)
            used_groups.add(gkey)
            used_source_types.add(stype)

    for idx, chunk in ranked:
        if len(selected) >= max_cites:
            break

        gkey = _source_group_key(chunk)
        if gkey in used_groups:
            continue

        selected.append(idx)
        used_groups.add(gkey)

    return selected