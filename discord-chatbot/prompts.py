from datetime import datetime
from typing import List

from zoneinfo import ZoneInfo
from config import LOCAL_TZ_NAME, MAX_CONTEXT_CHARS_QA, MAX_CONTEXT_CHARS_OVERVIEW
from storage import Chunk
from time_utils import fmt_local

LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)

def build_smalltalk_prompt(user_text: str) -> str:
    return f"""You are a friendly course Discord assistant.
Reply naturally to the message below (1-2 sentences).
Then add one short sentence offering course help.

User message:
{user_text}
"""

def build_rag_prompt_qa(question: str, chunks: List[Chunk]) -> str:
    now_str = datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p %Z")

    ctx_parts = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        created_str = fmt_local((c.meta or {}).get("created_at", ""))
        source_name = (c.meta or {}).get("channel") or (c.meta or {}).get("doc_title") or (c.meta or {}).get("file_name") or "unknown"
        line = f"[Source {i} | Origin: {source_name} | Posted: {created_str}] {c.text}\n"
        if total + len(line) > MAX_CONTEXT_CHARS_QA:
            break
        ctx_parts.append(line)
        total += len(line)
    ctx = "".join(ctx_parts)

    return f"""You are a course Discord assistant.

Today (local): {now_str}

Rules:
- Answer ONLY using the provided sources.
- Do NOT use outside knowledge.
- Do NOT invent deadlines, dates, policies, links, submission instructions, or clarifications.
- If the answer is not explicitly supported by the sources, say so clearly.
- For deadline, due-date, quiz, homework, assignment, Kritik, project, lab, exam, submission, or deliverable questions:
  - only report a date/time if it is explicitly stated in the sources
  - if no explicit due date is stated, say: "I could not find an explicit due date in the provided sources."
- Prefer the most relevant and recent source when sources disagree.
- Keep the answer short and specific (1-4 sentences).

Question:
{question}

Sources:
{ctx}

Output requirements:
- Answer directly.
- If helpful, mention the item name (for example: Kritik, homework, quiz, project).
- Do not mention any source number in the answer.
"""

def build_overview_prompt(question: str, chunks: List[Chunk]) -> str:
    now_str = datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p %Z")

    ctx_parts = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        created_str = fmt_local((c.meta or {}).get("created_at", ""))
        source_name = (c.meta or {}).get("channel") or (c.meta or {}).get("doc_title") or (c.meta or {}).get("file_name") or "unknown"
        line = f"[Source {i} | Origin: {source_name} | Posted: {created_str}] {c.text}\n"
        if total + len(line) > MAX_CONTEXT_CHARS_OVERVIEW:
            break
        ctx_parts.append(line)
        total += len(line)
    ctx = "".join(ctx_parts)

    return f"""You are a course Discord assistant.

Today (local): {now_str}

Task:
Provide an overview using ONLY the provided sources.

Include all distinct course items that are explicitly mentioned in the sources, especially:
- homework / hw
- assignments
- quizzes
- projects
- labs
- Kritik tasks
- exams / tests
- deadlines / due items / submissions

For each item, include:
- item name/title
- due date/time if explicitly present
- where to submit, only if explicitly present

Rules:
- Do NOT invent items, dates, or submission instructions.
- If a due date is not explicitly stated, write: "Due date not explicitly stated."
- If multiple sources refer to the same item, merge them into one bullet using the most complete supported details.
- Prefer more recent sources when they conflict.

User request:
{question}

Sources:
{ctx}

Output format:
- Bullet list
- One bullet per distinct item
"""