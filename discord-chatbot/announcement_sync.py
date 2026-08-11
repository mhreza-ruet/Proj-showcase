import discord

from logging_setup import setup_logging
from storage import upsert_message_async

log = setup_logging()


async def initial_backfill(bot, source_channel_names, startup_history, last_synced_announce_id):
    await bot.wait_until_ready()
    for guild in bot.guilds:
        for ch in guild.text_channels:
            if ch.name in source_channel_names:
                try:
                    newest_id = 0
                    async for msg in ch.history(limit=startup_history, oldest_first=True):
                        await upsert_message_async(msg)
                        newest_id = max(newest_id, msg.id)
                    if newest_id:
                        last_synced_announce_id[ch.id] = newest_id
                except Exception as e:
                    log.exception("Indexing error in #%s: %s", ch.name, e)


async def sync_announcements_before_answer(guild: discord.Guild, source_channel_names, last_synced_announce_id) -> None:
    try:
        for ch in guild.text_channels:
            if ch.name not in source_channel_names:
                continue

            after_id = last_synced_announce_id.get(ch.id, 0)
            after_obj = discord.Object(id=after_id) if after_id else None
            newest_id = after_id

            async for msg in ch.history(limit=200, oldest_first=True, after=after_obj):
                await upsert_message_async(msg)
                newest_id = max(newest_id, msg.id)

            if newest_id:
                last_synced_announce_id[ch.id] = newest_id
    except Exception as e:
        log.exception("Sync error: %s", e)