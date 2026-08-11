import asyncio
import re
from typing import Dict

import discord
from discord.ext import commands

from bot_helpers import (
    with_model_footer,
    strip_leading_answer_tag,
    _root_channel_name,
    _is_reply_to_bot,
    _mention_present)
from citation_utils import build_refs_from_chunks, select_citations
from announcement_sync import initial_backfill, sync_announcements_before_answer
from query_router import looks_like_course_query, choose_retrieve_mode

from config import (
    DISCORD_TOKEN,
    OLLAMA_MODEL,
    SOURCE_CHANNEL_NAMES,
    QUESTION_CHANNEL_NAMES,
    REPLY_ONLY_WHEN_MENTIONED,
    ALLOWED_MENTION_ROLE_IDS,
    STARTUP_HISTORY,
    MIN_SIMILARITY,
    CITE_MIN_SIMILARITY,
    MAX_CITES,
    OVERVIEW_INCLUDE_REFERENCES,
    OVERVIEW_MAX_REF_LINKS,
)

from logging_setup import setup_logging
from ollama_client import ollama_generate
from storage import ( upsert_message_async, retrieve_multi, get_latest_announcement, get_all_announcements, find_latest_matching_announcement, Chunk)
from normalize import normalize_question
from intents import (
    parse_latest_intent,
    looks_like_followup,
    looks_like_overview,
    is_announcement_priority_query,
    is_deadline_query,
    has_latest_word,
)
from prompts import build_smalltalk_prompt, build_rag_prompt_qa, build_overview_prompt
from time_utils import ( parse_time_window_from_query, warning_for_answer, posted_dt_local, choose_latest_due_from_announcement, format_latest_due_answer)

log = setup_logging()


# Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# per channel anchor (text+url)
LAST_ANCHOR: Dict[int, Dict[str, str]] = {}
LAST_SYNCED_ANNOUNCE_ID: Dict[int, int] = {}


def extract_item_hint_from_query(q: str) -> str:
    ql = (q or "").lower()
    for item in [
        "kritik",
        "hw",
        "homework",
        "quiz",
        "quizzes",
        "assignment",
        "project"
    ]:
        if item in ql:
            return item
    return ql

@bot.command()
async def ping(ctx: commands.Context):
    await ctx.reply(with_model_footer("pong ✅", OLLAMA_MODEL))


@bot.event
async def on_ready():
    log.info("Logged in as %s (id=%s)", bot.user, getattr(bot.user, "id", None))
    bot.loop.create_task(
        initial_backfill(
            bot,
            SOURCE_CHANNEL_NAMES,
            STARTUP_HISTORY,
            LAST_SYNCED_ANNOUNCE_ID,
        )
    )
    log.info("Initial backfill started.")


@bot.event
async def on_message(message: discord.Message):
    try:
        await bot.process_commands(message)

        if message.author.bot:
            return

        # Real-time indexing of announcements
        if hasattr(message.channel, "name") and message.channel.name in SOURCE_CHANNEL_NAMES:
            try:
                await upsert_message_async(message)
                LAST_SYNCED_ANNOUNCE_ID[message.channel.id] = max(
                    LAST_SYNCED_ANNOUNCE_ID.get(message.channel.id, 0),
                    message.id)
            except Exception as e:
                log.exception("Indexing error: %s", e)

        root = _root_channel_name(message)
        mention_tok = _mention_present(message, bot, ALLOWED_MENTION_ROLE_IDS)
        reply_tok = await _is_reply_to_bot(message, bot)

        # Strip mentions (bot + allowed roles) EARLY so replies/followups work
        raw = message.content or ""
        if bot.user:
            raw = re.sub(rf"<@!?{bot.user.id}>", "", raw)
        for rid in ALLOWED_MENTION_ROLE_IDS:
            raw = re.sub(rf"<@&{rid}>", "", raw)
        raw = raw.strip()

        if REPLY_ONLY_WHEN_MENTIONED and not (mention_tok or reply_tok):
            if not root or root.lower() not in {n.lower() for n in QUESTION_CHANNEL_NAMES}:
                return
            log.info(
                "IGNORED: root=%r mention=%s reply=%s content=%r",
                root, mention_tok, reply_tok, raw[:160])
            return

        if not root or root.lower() not in {n.lower() for n in QUESTION_CHANNEL_NAMES}:
            log.info("IGNORED(not question channel): root=%r content=%r", root, raw[:160])
            return

        log.info(
            "ACCEPTED: root=%r mention=%s reply=%s content=%r",
            root, mention_tok, reply_tok, raw[:160])

        # Sync first
        if message.guild:
            await sync_announcements_before_answer(
                message.guild,
                SOURCE_CHANNEL_NAMES,
                LAST_SYNCED_ANNOUNCE_ID)

        # Latest announcement path
        is_latest, topic = parse_latest_intent(raw)
        if is_latest and topic is None:
            latest = get_latest_announcement()
            if latest and latest["text"]:
                LAST_ANCHOR[message.channel.id] = {
                    "text": latest["text"],
                    "url": latest["url"],
                }
                await message.reply(
                    with_model_footer(
                        f"{latest['text']}\n\nLink: {latest['url']}",
                        OLLAMA_MODEL,
                    )
                )
            else:
                await message.reply(
                    with_model_footer(
                        "I couldn't find any messages indexed in **#announcements** yet.",
                        OLLAMA_MODEL,
                    )
                )
            return

        # Followup path (reply chain) — works on replies WITHOUT mention
        if reply_tok or looks_like_followup(message, raw):
            anchor = LAST_ANCHOR.get(message.channel.id, {})
            anchor_text = anchor.get("text", "")
            if anchor_text:
                log.info("FOLLOWUP: using LAST_ANCHOR for channel_id=%s", message.channel.id)
                prompt = f"""You are a course Discord assistant. Use ONLY the announcement text below. If it is not relevant, say: "Couldn't find it in #announcements."

Announcement text:
{anchor_text}

Question:
{raw}

Answer in 1-3 sentences.
"""
                ans = strip_leading_answer_tag(await asyncio.to_thread(ollama_generate, prompt))
                url = anchor.get("url", "")
                if url and "couldn't find" not in ans.lower():
                    await message.reply(
                        with_model_footer(
                            f"{ans}\n\n**Reference:**\n- {url}",
                            OLLAMA_MODEL,
                        )
                    )
                else:
                    await message.reply(with_model_footer(ans, OLLAMA_MODEL))
                return
            else:
                log.info("FOLLOWUP: no anchor yet; falling back to normal QA flow")

        # OVERVIEW / SUMMARY mode
        if looks_like_overview(raw):
            window = parse_time_window_from_query(raw)  # 7 days / 14 days / None
            if window is None:
                anns = get_all_announcements(time_gte_ts=None)
            else:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                from config import LOCAL_TZ_NAME

                tz = ZoneInfo(LOCAL_TZ_NAME)
                now = datetime.now(tz=tz)
                gte_dt = now - window
                gte_ts = int(gte_dt.timestamp())
                anns = get_all_announcements(time_gte_ts=gte_ts)

            if not anns:
                await message.reply(
                    with_model_footer(
                        "I couldn't find any messages indexed in **#announcements** yet.",
                        OLLAMA_MODEL,
                    )
                )
                return

            prompt = build_overview_prompt(raw, anns)
            ans = strip_leading_answer_tag(await asyncio.to_thread(ollama_generate, prompt))

            warn = warning_for_answer(raw, ans, anns)
            if warn:
                ans = f"{ans}\n\n{warn}"

            if OVERVIEW_INCLUDE_REFERENCES:
                seen = set()
                links = []
                for c in anns:
                    u = (c.meta or {}).get("url", "")
                    if u and u not in seen:
                        seen.add(u)
                        links.append(u)
                    if len(links) >= OVERVIEW_MAX_REF_LINKS:
                        break
                if links:
                    refs = "\n".join(f"- {u}" for u in links)
                    await message.reply(
                        with_model_footer(f"{ans}\n\n**References:**\n{refs}", OLLAMA_MODEL)
                    )
                    return

            await message.reply(with_model_footer(ans, OLLAMA_MODEL))
            return

        # SMALLTALK path
        norm_q = normalize_question(raw)
        looks_course = looks_like_course_query(norm_q)

        announcement_priority = is_announcement_priority_query(norm_q)
        deadline_query = is_deadline_query(norm_q)
        latest_query = has_latest_word(norm_q)

        if not looks_course:
            prompt = build_smalltalk_prompt(raw)
            ans = strip_leading_answer_tag(await asyncio.to_thread(ollama_generate, prompt))
            await message.reply(with_model_footer(ans, OLLAMA_MODEL))
            return

        # Deterministic latest-item-deadline path
        if latest_query and deadline_query and announcement_priority:
            item_hint = extract_item_hint_from_query(norm_q)
            match = find_latest_matching_announcement(item_hint)

            if match:
                posted = posted_dt_local(match.meta or {})
                if posted:
                    latest_due = choose_latest_due_from_announcement(match.text, posted)
                    if latest_due:
                        ans = format_latest_due_answer(latest_due)

                        LAST_ANCHOR[message.channel.id] = {
                            "text": match.text,
                            "url": (match.meta or {}).get("url", ""),
                        }

                        refs = build_refs_from_chunks([match])
                        await message.reply(
                            with_model_footer(f"{ans}\n\n**References:**\n{refs}", OLLAMA_MODEL)
                        )
                        return

            await message.reply(
                with_model_footer(
                    "I checked the announcements from newest to oldest, but I couldn't find a matching item with an explicit deadline.",
                    OLLAMA_MODEL,
                )
            )
            return

        # QA mode (targeted)
        retrieve_mode = choose_retrieve_mode( latest_query, announcement_priority, deadline_query, )

        chunks = retrieve_multi(norm_q, top_k_each=6, mode=retrieve_mode)
        best = chunks[0].score if chunks else 0.0

        if not chunks or best < MIN_SIMILARITY:
            if retrieve_mode in {"announcements_first", "announcements_only_recent"}:
                await message.reply(
                    with_model_footer(
                        "I checked the indexed announcements, but I couldn't find an explicit due date or deadline matching that query.",
                        OLLAMA_MODEL,
                    )
                )
            else:
                await message.reply(
                    with_model_footer(
                        "Couldn't find a reliable match in the indexed course sources.",
                        OLLAMA_MODEL,
                    )
                )
            return

        cite_ids = select_citations(
            chunks,
            prefer_recent_announcements=(announcement_priority or deadline_query or latest_query),
            max_cites=MAX_CITES,
            cite_min_similarity=CITE_MIN_SIMILARITY,
        )

        if not cite_ids:
            if retrieve_mode in {"announcements_first", "announcements_only_recent"}:
                await message.reply(
                    with_model_footer(
                        "I checked the indexed announcements, but I couldn't find an explicit due date or deadline matching that query.",
                        OLLAMA_MODEL,
                    )
                )
            else:
                await message.reply(
                    with_model_footer(
                        "Couldn't find a reliable match in the indexed course sources.",
                        OLLAMA_MODEL,
                    )
                )
            return

        cited = [chunks[i - 1] for i in cite_ids]
        prompt = build_rag_prompt_qa(raw, cited)
        ans = strip_leading_answer_tag(await asyncio.to_thread(ollama_generate, prompt))

        warn = warning_for_answer(raw, ans, cited)
        if warn:
            ans = f"{ans}\n\n{warn}"

        # Anchor (so the next reply works as followup)
        LAST_ANCHOR[message.channel.id] = {
            "text": cited[0].text,
            "url": (cited[0].meta or {}).get("url", ""),
        }

        refs = build_refs_from_chunks(cited)
        await message.reply(
            with_model_footer(f"{ans}\n\n**References:**\n{refs}", OLLAMA_MODEL)
        )

    except Exception as e:
        log.exception("on_message crashed: %s", e)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)