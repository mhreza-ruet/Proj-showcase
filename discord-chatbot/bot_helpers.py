import re
from typing import Dict, Optional

import discord
from discord.ext import commands


def with_model_footer(text: str, model_name: str) -> str:
    return f"{text}\n\n_Model: {model_name}_"


def strip_leading_answer_tag(s: str) -> str:
    return re.sub(r"(?is)^\s*Answer:\s*", "", (s or "")).strip()


def _root_channel_name(message: discord.Message) -> Optional[str]:
    if isinstance(message.channel, discord.Thread):
        parent = message.channel.parent
        return parent.name if parent else None
    return getattr(message.channel, "name", None)


async def _is_reply_to_bot(message: discord.Message, bot: commands.Bot) -> bool:
    """
    True if this message is a reply to a message sent by THIS bot.
    Works even when reference isn't resolved by fetching the referenced message.
    """
    if not bot.user:
        return False
    ref = getattr(message, "reference", None)
    if not ref or not getattr(ref, "message_id", None):
        return False
    resolved = getattr(ref, "resolved", None)
    if resolved and getattr(resolved, "author", None):
        return getattr(resolved.author, "id", None) == bot.user.id
    try:
        parent_msg = await message.channel.fetch_message(ref.message_id)
        return parent_msg.author.id == bot.user.id
    except Exception:
        return False


def _mention_present(message: discord.Message, bot: commands.Bot, allowed_mention_role_ids) -> bool:
    content = message.content or ""
    if not bot.user:
        return False
    if f"<@{bot.user.id}>" in content or f"<@!{bot.user.id}>" in content:
        return True
    for rid in allowed_mention_role_ids:
        if f"<@&{rid}>" in content:
            return True
    return False


def should_respond(
    message: discord.Message,
    mention_tok: bool,
    reply_tok: bool,
    question_channel_names,
    reply_only_when_mentioned: bool,
) -> bool:
    if message.author.bot:
        return False

    root = _root_channel_name(message)
    if not root:
        return False

    if root.lower() not in {n.lower() for n in question_channel_names}:
        return False

    if reply_only_when_mentioned:
        return mention_tok or reply_tok

    return True