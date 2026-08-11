import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]
CHROMA_DIR = os.getenv("CHROMA_DIR", "/home/exouser/course-ai/chroma_db")

# Channels
SOURCE_CHANNEL_NAMES = {"announcements"}
QUESTION_CHANNEL_NAMES = {"queries", "beta-testers"}

# Mention behavior
REPLY_ONLY_WHEN_MENTIONED = True
ALLOWED_MENTION_ROLE_IDS = {1473386632483831954}

# Retrieval defaults (QA mode)
TOP_K_QA = 10
MIN_SIMILARITY = 0.45
CITE_MIN_SIMILARITY = 0.50
MAX_CITES = 3

# Overview behavior
# In overview/summary mode we fetch ALL announcements in the window using collection.get()
# No citations by default (you can turn on if you later want them)
OVERVIEW_INCLUDE_REFERENCES = False
OVERVIEW_MAX_REF_LINKS = 5

# Indexing
STARTUP_HISTORY = 400
UPSERT_CONCURRENCY = 2

# LLM output control
MAX_CONTEXT_CHARS_QA = 9000
MAX_CONTEXT_CHARS_OVERVIEW = 22000  # overview needs more room

# Timezone
LOCAL_TZ_NAME = os.getenv("LOCAL_TZ", "America/New_York")
