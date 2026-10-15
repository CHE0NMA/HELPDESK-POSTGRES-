"""Shared bot-message → HTML formatting.

Used by both the live /chat POST endpoint and the /chat/<uuid> page-load
path, so a message renders identically whether it just arrived or is
being restored from chat_history after a refresh.
"""
import re

_IMG_TAG = (
    r'<img src="/\1" style="max-width:200px;border-radius:8px;'
    r'margin:10px 0;display:block;" loading="lazy">'
)


def format_bot_message(answer: str, show_images: bool = True) -> str:
    """Turn a raw answer into clean chat HTML.

    Supports on ALL answers (not only lyrics):
      - **bold**  -> <strong>
      - *italic*  -> <em>
      - blank lines -> paragraph spacing
      - single newlines -> <br> (lists, steps, lyrics stay readable)
      - lines starting with - or • -> bullet marker
      - [Image: path] -> <img>
    """
    if not answer:
        return ''

    text = answer

    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)

    # Normalize bullet prefixes
    text = re.sub(r'(?m)^[\-\u2022]\s+', '• ', text)

    # Paragraph breaks (blank line) vs line breaks
    parts = re.split(r'\n\s*\n', text)
    blocks = [p.replace('\n', '<br>') for p in parts]
    formatted = '<br><br>'.join(blocks)

    if show_images:
        formatted = re.sub(r'\[Image: (.*?)\]', _IMG_TAG, formatted)
    else:
        formatted = re.sub(r'\[Image: .*?\]', '', formatted)

    return formatted