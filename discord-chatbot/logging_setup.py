import logging

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("coursebot")
    logging.getLogger("discord").setLevel(logging.INFO)
    return log