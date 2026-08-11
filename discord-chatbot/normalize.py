import re

ORDINAL_TO_NUM = {
    "first": "1", "1st": "1",
    "second": "2", "2nd": "2",
    "third": "3", "3rd": "3",
    "fourth": "4", "4th": "4",
    "fifth": "5", "5th": "5",
    "sixth": "6", "6th": "6",
    "seventh": "7", "7th": "7",
    "eighth": "8", "8th": "8",
    "ninth": "9", "9th": "9",
    "tenth": "10", "10th": "10",
}

ITEM_TYPES = [
    "quiz", "hw", "homework", "assignment", "project", "lab",
    "midterm", "exam", "test", "kritik", "task", "tasks",
]

LATEST_WORDS = [
    "latest", "last", "newest", "recent", "most recent", "upcoming", "current", "currently"
]

DEADLINE_WORDS = [
    "deadline", "deadlines", "due", "due date", "due dates",
    "submission", "submissions", "submit", "deliverable", "deliverables"
]


def _contains_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def normalize_question(q: str) -> str:
    if not q:
        return ""

    t = q.lower().strip()

    # basic cleanup
    t = re.sub(r"[“”\"']", "", t)
    t = re.sub(r"[(),;:?!]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # canonicalize common variants
    t = re.sub(r"\bhome\s*work\b", "homework", t)
    t = re.sub(r"\bhomework\b", "hw", t)
    t = re.sub(r"\boh\b", "office hours", t)
    t = re.sub(r"\brecordings?\b", "recording", t)
    t = re.sub(r"\blecture\s*videos?\b", "recording", t)
    t = re.sub(r"\bdue\s*date\b", "deadline", t)
    t = re.sub(r"\bdue\s*dates\b", "deadlines", t)
    t = re.sub(r"\bgrades?\b", "grading", t)
    t = re.sub(r"\bassignments?\b", "assignment", t)
    t = re.sub(r"\bprojects?\b", "project", t)
    t = re.sub(r"\blabs?\b", "lab", t)
    t = re.sub(r"\bquizzes\b", "quiz", t)
    t = re.sub(r"\bexams\b", "exam", t)
    t = re.sub(r"\btests\b", "test", t)
    t = re.sub(r"\bsubmissions?\b", "submission", t)
    t = re.sub(r"\bdeliverables?\b", "deliverable", t)

    # normalize numbered items like hw2 -> hw 2, quiz #3 -> quiz 3
    for typ in ["quiz", "hw", "assignment", "project", "lab", "midterm", "exam", "test", "kritik"]:
        t = re.sub(rf"\b{typ}\s*#?\s*(\d+)\b", rf"{typ} \1", t)
        t = re.sub(rf"\b{typ}(\d+)\b", rf"{typ} \1", t)

    # normalize ordinals like first quiz -> quiz 1
    for ordw, num in ORDINAL_TO_NUM.items():
        for typ in ["quiz", "hw", "assignment", "project", "lab", "midterm", "exam", "test", "kritik"]:
            t = re.sub(rf"\b{ordw}\s+{typ}\b", f"{typ} {num}", t)
            t = re.sub(rf"\b{typ}\s+{ordw}\b", f"{typ} {num}", t)

    extra_terms = []

    # latest / recent intent expansion
    if _contains_any(t, LATEST_WORDS):
        extra_terms.extend(["latest", "recent", "newest", "last"])

    # detect deadline / due-date style questions more robustly
    deadline_patterns = [
        "when is",
        "when's",
        "when do",
        "when does",
        "when should",
        "what is due",
        "what's due",
        "what is due next",
        "what is due this",
        "due next",
        "due this week",
        "due next week",
        "due date",
        "due dates",
        "deadline",
        "deadlines",
        "submit",
        "submission",
        "turn in",
        "hand in",
        "deliverable",
    ]

    deadline_detected = False

    # direct keyword detection
    if _contains_any(t, DEADLINE_WORDS):
        deadline_detected = True

    # pattern detection
    if any(p in t for p in deadline_patterns):
        deadline_detected = True

    # item + timing detection (e.g. "quiz 3 when", "kritik 2 due")
    if any(item in t for item in ITEM_TYPES) and any( w in t for w in ["when", "due", "deadline", "submit"]):
        deadline_detected = True

    if deadline_detected:
        extra_terms.extend(["deadline", "due", "submission", "submit", "announcement", "announcements"])
    
    # upcoming / future tasks
    if "next" in t or "upcoming" in t:
        extra_terms.extend(["upcoming", "deadline", "due", "announcement"])

    # announcement-priority item expansions
    if "kritik" in t:
        extra_terms.extend(["kritik", "deadline", "submission", "announcement"])
    if "quiz" in t:
        extra_terms.extend(["quiz", "deadline", "announcement"])
    if "hw" in t:
        extra_terms.extend(["hw", "homework", "deadline", "announcement"])
    if "assignment" in t:
        extra_terms.extend(["assignment", "deadline", "announcement"])
    if "project" in t:
        extra_terms.extend(["project", "deadline", "announcement"])
    if "lab" in t:
        extra_terms.extend(["lab", "deadline", "announcement"])
    if "exam" in t or "test" in t or "midterm" in t:
        extra_terms.extend(["exam", "announcement"])
    if "recording" in t:
        extra_terms.extend(["recording", "announcement"])
    if "slides" in t:
        extra_terms.extend(["slides", "announcement"])
    if "grading" in t:
        extra_terms.extend(["grading", "policy", "announcement"])
    if "policy" in t:
        extra_terms.extend(["policy", "announcement"])

    # if clearly course-related, lightly bias toward announcements
    if any(w in t for w in [
        "kritik", "quiz", "hw", "assignment", "project", "lab",
        "exam", "midterm", "test", "deadline", "due", "submission",
        "recording", "slides", "grading", "policy"
    ]):
        extra_terms.append("announcements")

    # deduplicate while preserving order
    seen = set()
    final_terms = []
    for token in (t.split() + extra_terms):
        if token not in seen:
            seen.add(token)
            final_terms.append(token)

    return " ".join(final_terms).strip()