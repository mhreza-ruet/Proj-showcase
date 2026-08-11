import re
from typing import Optional, Tuple

LATEST_WORDS = [
    "latest", "last", "newest", "most recent", "recent", "currently", "upcoming"
]

ANNOUNCEMENT_PRIORITY_WORDS = [
    "deadline", "deadlines",
    "due", "due date", "due dates",
    "kritik",
    "quiz", "quizzes",
    "hw", "homework",
    "assignment", "assignments",
    "task", "tasks",
    "creation",
    "evaluation",
    "project", "projects",
    "lab", "labs",
    "exam", "exams",
    "submission", "submissions",
    "submit", "submitted",
    "deliverable", "deliverables",
    "grading", "grading policy",
    "late", "late policy",
    "recording", "recordings",
    "slides",
    "policy", "policies",
]

OVERVIEW_WORDS = [
    "summarize", "summary", "overview",
    "as of today",
    "list all",
    "what are the various",
    "everything due",
    "what is due",
    "upcoming",
    "coming week",
    "next week",
    "next two weeks",
    "last week",
    "last two weeks",
]

TOPIC_HINTS = ["about", "regarding", "on", "re:", "related to", "with respect to"]


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower()).strip()


def has_latest_word(q: str) -> bool:
    ql = _norm(q)
    return any(w in ql for w in LATEST_WORDS)


def is_announcement_priority_query(q: str) -> bool:
    ql = _norm(q)
    return any(w in ql for w in ANNOUNCEMENT_PRIORITY_WORDS)


def is_deadline_query(q: str) -> bool:
    ql = _norm(q)
    deadline_words = [
        "deadline", "deadlines",
        "due", "due date", "due dates",
        "submit", "submission", "submissions",
        "deliverable", "deliverables",
        "when is", "when's",
    ]
    return any(w in ql for w in deadline_words)


def parse_latest_intent(question: str) -> Tuple[bool, Optional[str]]:
    q = _norm(question)
    if not has_latest_word(q):
        return (False, None)

    for h in TOPIC_HINTS:
        m = re.search(rf"\b{re.escape(h)}\b\s+(.+)$", q)
        if m:
            topic = m.group(1).strip()
            topic = re.split(r"[?.!]", topic)[0].strip()
            return (True, topic or None)

    # also support patterns like:
    # "latest kritik deadlines"
    # "recent homework"
    # "last quiz due date"
    if is_announcement_priority_query(q):
        return (True, q)

    return (True, None)


def looks_like_followup(message, question: str) -> bool:
    ql = _norm(question)
    if not ql:
        return False
    if getattr(message, "reference", None) and getattr(message.reference, "message_id", None):
        return True
    if len(ql) <= 80 and any(w in ql for w in [
        "what about",
        "what does that mean",
        "clarify",
        "explain",
        "why",
        "how come",
        "and",
        "what else",
        "which one",
        "when is that",
        "when is it due",
    ]):
        return True
    return False


def looks_like_overview(question: str) -> bool:
    ql = _norm(question)
    return any(p in ql for p in OVERVIEW_WORDS)