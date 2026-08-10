"""Shared bot-message → HTML formatting.

Used by both the live /chat POST endpoint and the /chat/<uuid> page-load
path, so a message renders identically whether it just arrived or is
being restored from chat_history after a refresh. Having two separate
copies of this logic (one in routes/chat.py, one implicitly "missing" on
reload) was why images disappeared on refresh — the reload path never
converted the stored "[Image: path]" marker into an <img> tag at all.
"""
import re

_IMG_TAG = (
    r'<img src="/\1" style="max-width:200px;border-radius:8px;'
    r'margin:10px 0;display:block;" loading="lazy">'
)


def format_bot_message(answer: str, show_images: bool = True) -> str:
    """Turn a raw stored/generated answer into the HTML the chat bubble
    expects: **bold** -> <strong>, newlines -> <br>, and [Image: path]
    markers -> actual <img> tags (or stripped out if show_images=False).
    """
    if not answer:
        return ''
    formatted = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', answer)
    formatted = formatted.replace('\n', '<br>')
    if show_images:
        formatted = re.sub(r'\[Image: (.*?)\]', _IMG_TAG, formatted)
    else:
        formatted = re.sub(r'\[Image: .*?\]', '', formatted)
    return formatted