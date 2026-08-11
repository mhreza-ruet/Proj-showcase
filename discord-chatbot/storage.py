import asyncio
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Dict, List, Optional
import re
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, UPSERT_CONCURRENCY, TOP_K_QA

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

chroma = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

# Existing Discord knowledge collection
collection = chroma.get_or_create_collection(name="discord_knowledge", metadata={"hnsw:space": "cosine"})

# Existing PDF collection from ingest_to_chroma.py
pdf_col = chroma.get_or_create_collection(name="course_files", metadata={"hnsw:space": "cosine"})

# NEW: NotebookLM / cleaned notes collection from ingest_to_chroma.py
notes_col = chroma.get_or_create_collection(name="course_notes", metadata={"hnsw:space": "cosine"})


@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]
    score: float


def jump_url(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def upsert_message(msg) -> None:
    if not getattr(msg, "content", None) or getattr(msg.author, "bot", False):
        return

    text = (msg.content or "").strip()
    if len(text) < 5:
        return

    gid = msg.guild.id if msg.guild else 0
    cid = msg.channel.id
    mid = msg.id
    url = jump_url(gid, cid, mid)

    emb = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

    # timestamps (used for "latest" + time-window overview)
    created = getattr(msg, "created_at", None)
    if created and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created_at = created.isoformat() if created else ""
    created_ts = int(created.timestamp()) if created else 0

    collection.upsert(
        ids=[f"{gid}:{cid}:{mid}"],
        documents=[text],
        metadatas=[{
            "channel": getattr(msg.channel, "name", ""),
            "url": url,
            "author": str(msg.author),
            "message_id": str(mid),
            "created_at": created_at,
            "created_ts": created_ts,
        }],
        embeddings=[emb],
    )


UPSERT_SEM = asyncio.Semaphore(UPSERT_CONCURRENCY)


async def upsert_message_async(msg) -> None:
    async with UPSERT_SEM:
        await asyncio.to_thread(upsert_message, msg)


def _query_collection( col, question: str, top_k: int, where: Optional[Dict[str, Any]] = None ) -> List[Chunk]:
    q = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

    query_kwargs = {
        "query_embeddings": [q],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    res = col.query(**query_kwargs)

    chunks: List[Chunk] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    for text, meta, dist in zip(docs, metas, dists):
        chunks.append(Chunk(text=text, meta=meta or {}, score=1.0 - float(dist)))

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks

def _recent_announcement_chunks(limit: int = 10) -> List[Chunk]:
    """
    Get the most recent announcement messages directly from collection.get(),
    sorted newest first, then trimmed to `limit`.
    """
    res = collection.get(where={"channel": "announcements"}, include=["documents", "metadatas"])
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []

    out: List[Chunk] = []
    for text, meta in zip(docs, metas):
        out.append(Chunk(
            text=(text or "").strip(),
            meta=(meta or {}),
            score=1.0
        ))

    out.sort(key=lambda c: int(c.meta.get("created_ts", 0) or 0), reverse=True)
    return out[:limit]


def _semantic_rerank(question: str, chunks: List[Chunk], top_k: int) -> List[Chunk]:
    """
    Re-score an existing chunk list against the user question.
    Useful when we first fetch 'latest N announcements' and then want
    the most relevant among those latest announcements.
    """
    if not chunks:
        return []

    q = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]

    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    rescored: List[Chunk] = []
    for c, emb in zip(chunks, embs):
        score = float((q * emb).sum())
        rescored.append(Chunk(text=c.text, meta=c.meta, score=score))

    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]

def retrieve_multi(question: str, top_k_each: int = 6, mode: str = "mixed") -> List[Chunk]:
    """
    Retrieval modes:

    - mixed:
        announcements + notes + pdfs, all merged by similarity

    - announcements_first:
        query announcements first; if strong enough results exist, prefer them.
        otherwise back off to mixed retrieval.

    - announcements_only_recent:
        look only at the most recent announcement messages first,
        then re-rank those by semantic relevance.
    """
    question = (question or "").strip()

    if mode == "announcements_only_recent":
        recent = _recent_announcement_chunks(limit=max(10, top_k_each * 2))
        ranked_recent = _semantic_rerank(question, recent, top_k_each)
        return ranked_recent

    if mode == "announcements_first":
        ann = _query_collection( collection, question, top_k=max(top_k_each, 8), where={"channel": "announcements"}, )

        # If announcements already look decent, prefer them heavily.
        # This helps deadline / due / kritik / hw questions stay in announcements.
        if ann and ann[0].score >= 0.35:
            return ann[:top_k_each]

        # Otherwise blend with other sources as fallback
        out: List[Chunk] = []
        out.extend(ann)
        out.extend(_query_collection(notes_col, question, top_k_each, where=None))
        out.extend(_query_collection(pdf_col, question, top_k_each, where=None))
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    # default: mixed
    out: List[Chunk] = []
    out.extend(_query_collection( collection, question, top_k_each, where={"channel": "announcements"} ))
    out.extend(_query_collection(notes_col, question, top_k_each, where=None))
    out.extend(_query_collection(pdf_col, question, top_k_each, where=None))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def get_all_announcements(time_gte_ts: Optional[int] = None) -> List[Chunk]:
    """
    Overview mode: fetch ALL announcements (or all since time_gte_ts).
    Uses collection.get (not vector query), so it won’t miss items.
    """
    where: Dict[str, Any] = {"channel": "announcements"}
    if time_gte_ts is not None:
        where = {
            "$and": [
                {"channel": "announcements"},
                {"created_ts": {"$gte": int(time_gte_ts)}},
            ]
        }

    res = collection.get(where=where, include=["documents", "metadatas"])
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []

    out: List[Chunk] = []
    for text, meta in zip(docs, metas):
        out.append(Chunk(text=(text or "").strip(), meta=(meta or {}), score=1.0))

    out.sort(key=lambda c: int(c.meta.get("created_ts", 0)), reverse=True)
    return out


def get_latest_announcement() -> Optional[Dict[str, str]]:
    res = collection.get(where={"channel": "announcements"}, include=["documents", "metadatas"])
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []
    if not docs or not metas:
        return None

    best_i = 0
    best_ts = -1
    for i, m in enumerate(metas):
        ts = int((m or {}).get("created_ts", 0) or 0)
        if ts > best_ts:
            best_ts = ts
            best_i = i

    return {
        "text": (docs[best_i] or "").strip(),
        "url": (metas[best_i] or {}).get("url", ""),
    }


def _contains_any_term(text: str, terms: List[str]) -> bool:
    tl = (text or "").lower()
    return any(term.lower() in tl for term in terms if term)


def _normalize_item_terms(item_hint: str) -> List[str]:
    q = (item_hint or "").lower().strip()
    terms = []

    if "kritik" in q:
        terms.extend(["kritik", "brainstorming"])
    if "hw" in q or "homework" in q:
        terms.extend(["hw", "homework"])
    if "quiz" in q:
        terms.extend(["quiz"])
    if "assignment" in q:
        terms.extend(["assignment"])
    if "project" in q:
        terms.extend(["project"])
    if "lab" in q:
        terms.extend(["lab"])
    if "exam" in q or "test" in q or "midterm" in q:
        terms.extend(["exam", "test", "midterm"])

    if not terms and q:
        terms.append(q)

    # dedupe
    out = []
    seen = set()
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def find_latest_announcement_match(
    item_hint: str,
    require_deadline: bool = True,
) -> List[Chunk]:
    """
    Scan announcements newest -> oldest.
    Return candidate announcements mentioning the requested item.
    If require_deadline=True, only return announcements that likely contain
    a deadline / due-date signal.
    """
    anns = get_all_announcements(time_gte_ts=None)  # already newest first
    item_terms = _normalize_item_terms(item_hint)

    deadline_patterns = [
        r"\bdue\b",
        r"\bdeadline\b",
        r"\bdeadlines\b",
        r"\bsubmit\b",
        r"\bsubmission\b",
        r"\bsubmitted\b",
        r"\bdeliverable\b",
        r"\bfeedback is due\b",
        r"\bcreation is due\b",
        r"\bevaluation is due\b",
        r"\bby\s+\d{1,2}:\d{2}\s*(am|pm)\b",
        r"\bnext week\b",
        r"\bmonday\b|\btuesday\b|\bwednesday\b|\bthursday\b|\bfriday\b|\bsaturday\b|\bsunday\b",
        r"\bjan\b|\bfeb\b|\bmar\b|\bapr\b|\bmay\b|\bjun\b|\bjul\b|\baug\b|\bsep\b|\boct\b|\bnov\b|\bdec\b",
        r"\bjanuary\b|\bfebruary\b|\bmarch\b|\bapril\b|\bjune\b|\bjuly\b|\baugust\b|\bseptember\b|\boctober\b|\bnovember\b|\bdecember\b",
    ]

    out: List[Chunk] = []
    for c in anns:
        text = (c.text or "").lower()

        if item_terms and not _contains_any_term(text, item_terms):
            continue

        if require_deadline:
            if not any(re.search(p, text, re.I) for p in deadline_patterns):
                continue

        out.append(c)

    return out

def find_latest_matching_announcement(item_hint: str) -> Optional[Chunk]:
    """
    Scan announcements newest -> oldest and return the first announcement
    that matches the item hint and contains likely deadline language.
    """
    anns = get_all_announcements(time_gte_ts=None)  # newest first
    item_hint_l = (item_hint or "").lower().strip()

    item_terms = []
    if "kritik" in item_hint_l:
        item_terms = ["kritik", "brainstorming"]
    elif "hw" in item_hint_l or "homework" in item_hint_l:
        item_terms = ["hw", "homework"]
    elif "quiz" in item_hint_l:
        item_terms = ["quiz"]
    elif "assignment" in item_hint_l:
        item_terms = ["assignment"]
    elif "project" in item_hint_l:
        item_terms = ["project"]
    elif "lab" in item_hint_l:
        item_terms = ["lab"]
    elif "exam" in item_hint_l or "test" in item_hint_l or "midterm" in item_hint_l:
        item_terms = ["exam", "test", "midterm"]
    else:
        item_terms = [item_hint_l] if item_hint_l else []

    deadline_terms = [
        "due",
        "deadline",
        "deadlines",
        "submit",
        "submission",
        "deliverable",
        "creation is due",
        "evaluation is due",
        "feedback is due",
        "next week",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august",
        "sep", "sept", "september", "oct", "october", "nov", "november", "dec", "december",
    ]

    for c in anns:
        text_l = (c.text or "").lower()

        if item_terms and not any(term in text_l for term in item_terms):
            continue

        if not any(term in text_l for term in deadline_terms):
            continue

        return c

    return None