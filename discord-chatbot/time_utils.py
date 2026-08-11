import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List
from zoneinfo import ZoneInfo
from config import LOCAL_TZ_NAME
from storage import Chunk

LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

TIME_QUERY_CUES = [
    "due", "deadline", "by ", "before ", "after ", "next week", "this week", "coming week",
    "last week", "last two weeks", "past week", "past two weeks", "as of today", "in the next",
    "in next", "within", "upcoming", "coming two weeks", "next two weeks",
]

def is_time_sensitive_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(c in ql for c in TIME_QUERY_CUES)

def parse_time_window_from_query(q: str, now: Optional[datetime] = None) -> Optional[timedelta]:
    """
    Returns a timedelta window if user asks "last week", "last two weeks", etc.
    If none, returns None (meaning: all announcements).
    """
    ql = (q or "").lower()
    now = now or datetime.now(tz=LOCAL_TZ)

    if "last two weeks" in ql or "past two weeks" in ql or "previous two weeks" in ql:
        return timedelta(days=14)
    if "last week" in ql or "past week" in ql or "previous week" in ql:
        return timedelta(days=7)
    if "last 2 weeks" in ql:
        return timedelta(days=14)
    if "last 7 days" in ql:
        return timedelta(days=7)
    if "last 14 days" in ql:
        return timedelta(days=14)

    return None

def fmt_local(iso_or_empty: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_or_empty.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception:
        return "unknown time"

def posted_dt_local(meta: Dict[str, Any]) -> Optional[datetime]:
    iso = (meta or {}).get("created_at", "")
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ)
    except Exception:
        return None

def _next_weekday(base: datetime, weekday: int) -> datetime:
    days_ahead = (weekday - base.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return base + timedelta(days=days_ahead)

def extract_all_due_datetimes_from_text(text: str, posted: datetime) -> List[Dict[str, Any]]:
    """
    Extract all due-date candidates from one announcement.

    Returns a list like:
    [
        {"label": "Creation", "due_dt": datetime(...)},
        {"label": "Evaluation", "due_dt": datetime(...)},
        {"label": "Feedback", "due_dt": datetime(...)},
    ]
    """
    t = (text or "")
    tl = t.lower()
    out: List[Dict[str, Any]] = []

    # Pattern 1: "Creation is due on Tuesday, March 10 at 11:59 pm"
    pattern_abs = re.finditer(
        r"(?P<label>[A-Za-z][A-Za-z _/-]{0,40}?)\s+is\s+due\s+(?:on\s+)?"
        r"(?:(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday),\s+)?"
        r"(?P<month>jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+"
        r"(?P<day>\d{1,2})(?:,\s*(?P<year>\d{4}))?"
        r"(?:\s+at\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>am|pm))?",
        t,
        re.I,
    )
    for m in pattern_abs:
        label = (m.group("label") or "").strip(" :-")
        mon = MONTHS[m.group("month").lower()]
        day = int(m.group("day"))
        year = int(m.group("year")) if m.group("year") else posted.year

        if m.group("hour") and m.group("minute") and m.group("ampm"):
            hour = int(m.group("hour"))
            minute = int(m.group("minute"))
            ampm = m.group("ampm").lower()
            if ampm == "pm" and hour != 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
        else:
            hour, minute = 23, 59

        try:
            due_dt = datetime(year, mon, day, hour, minute, tzinfo=LOCAL_TZ)
            out.append({"label": label, "due_dt": due_dt, "raw": m.group(0)})
        except Exception:
            pass

    # Pattern 2: "Creation is due by Friday"
    pattern_weekday = re.finditer(
        r"(?P<label>[A-Za-z][A-Za-z _/-]{0,40}?)\s+is\s+due\s+(?:by|on)\s+"
        r"(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        tl,
        re.I,
    )
    for m in pattern_weekday:
        label = (m.group("label") or "").strip(" :-")
        wd = WEEKDAYS[m.group("weekday").lower()]
        due_dt = _next_weekday(posted, wd).replace(hour=23, minute=59, second=0, microsecond=0)
        out.append({"label": label, "due_dt": due_dt, "raw": m.group(0)})

    # Pattern 3: generic "due next week Wednesday"
    pattern_next_weekday = re.finditer(
        r"(?P<label>[A-Za-z][A-Za-z _/-]{0,40}?)\s+is\s+due\s+(?:on\s+)?next week\s+"
        r"(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        tl,
        re.I,
    )
    for m in pattern_next_weekday:
        label = (m.group("label") or "").strip(" :-")
        wd = WEEKDAYS[m.group("weekday").lower()]
        base = posted + timedelta(days=7)
        due_dt = _next_weekday(base, wd).replace(hour=23, minute=59, second=0, microsecond=0)
        out.append({"label": label, "due_dt": due_dt, "raw": m.group(0)})

    return out

def choose_latest_due_from_announcement(text: str, posted: datetime) -> Optional[Dict[str, Any]]:
    """
    From a single announcement, return the latest due-date mentioned.
    """
    items = extract_all_due_datetimes_from_text(text, posted)
    if not items:
        return None
    items.sort(key=lambda x: x["due_dt"], reverse=True)
    return items[0]

def format_latest_due_answer(item: Dict[str, Any], now: Optional[datetime] = None) -> str:
    now = now or datetime.now(tz=LOCAL_TZ)
    due_dt = item["due_dt"]
    label = (item.get("label") or "").strip()

    due_str = due_dt.strftime("%A, %B %d at %I:%M %p %Z")

    if label:
        base = f"The latest listed deadline is for {label}, due {due_str}."
    else:
        base = f"The latest listed deadline is {due_str}."

    if now > due_dt:
        return f"{base} It appears to be past due."

    return base

@dataclass
class DueCheck:
    due_dt: datetime
    posted_dt: datetime
    source_url: str
    source_text: str

def find_past_due_items(chunks: List[Chunk], now: Optional[datetime] = None) -> List[DueCheck]:
    now = now or datetime.now(tz=LOCAL_TZ)
    out: List[DueCheck] = []

    for c in chunks:
        posted = posted_dt_local(c.meta)
        if not posted:
            continue

        all_due = extract_all_due_datetimes_from_text(c.text, posted)
        for item in all_due:
            due = item["due_dt"]
            if now > due:
                out.append(DueCheck(
                    due_dt=due,
                    posted_dt=posted,
                    source_url=(c.meta or {}).get("url", ""),
                    source_text=(c.text or "")[:400],
                ))

    out.sort(key=lambda x: x.due_dt, reverse=True)
    return out

def warning_for_answer(question: str, answer: str, cited_chunks: List[Chunk]) -> Optional[str]:
    """
    Attach warning ONLY when:
      - user asked something time-sensitive AND
      - we can infer at least one due date AND
      - query time is past that due date AND
      - answer is not a refusal
    """
    if not is_time_sensitive_question(question):
        return None
    if "couldn't find" in (answer or "").lower():
        return None

    overdue = find_past_due_items(cited_chunks)
    if not overdue:
        return None

    top = overdue[0]
    due_str = top.due_dt.strftime("%Y-%m-%d")
    return f"⚠️ Note: at least one deadline in the referenced announcements appears past-due (e.g., due around {due_str}). Please check the most recent announcements for updates."