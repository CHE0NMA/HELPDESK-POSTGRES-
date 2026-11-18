"""Shared bot-message → HTML formatting.

Used by both the live /chat POST endpoint and the /chat/<uuid> page-load
path, so a message renders identically whether it just arrived or is
being restored from chat_history after a refresh.
"""
import html
import re
from urllib.parse import quote, urlsplit


_IMAGE_RE = re.compile(r"\[Image: (.*?)\]")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+(.+)$")


def _image_tag(match: re.Match) -> str:
    image_path = match.group(1).strip()
    parsed = urlsplit(image_path)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme != "https" or parsed.hostname != "res.cloudinary.com":
            return ""
        src = image_path
    else:
        local_path = image_path.lstrip("/")
        if not local_path.startswith("uploads/") or ".." in local_path.split("/"):
            return ""
        src = f"/{quote(local_path, safe='/')}"
    return (
        '<span class="bot-image-frame">'
        '<span class="bot-image-placeholder" role="status">Loading attached image…</span>'
        f'<img src="{html.escape(src, quote=True)}" alt="FAQ image" '
        'style="max-width:200px;border-radius:8px;margin:10px 0;display:block;" '
        'loading="lazy">'
        '</span>'
    )


def _format_inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        r"<em>\1</em>",
        escaped,
    )


def format_bot_message(answer: str, show_images: bool = True) -> str:
    """Render a small, safe Markdown subset for readable chat answers.

    Supports bold and italic text, headings, paragraphs, bullet and numbered
    lists, and validated FAQ image references.
    """
    if not answer:
        return ''

    images = []

    def preserve_image(match: re.Match) -> str:
        image_html = _image_tag(match) if show_images else ''
        if not image_html:
            return ''
        images.append(image_html)
        return f"\x00IMAGE{len(images) - 1}\x00"

    text = _IMAGE_RE.sub(preserve_image, answer.replace("\r\n", "\n").replace("\r", "\n"))
    lines = text.split("\n")
    blocks = []
    paragraph = []
    list_type = None
    list_items = []

    def flush_paragraph() -> None:
        if paragraph:
            body = "<br>".join(_format_inline(line) for line in paragraph)
            blocks.append(f"<p>{body}</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_type, list_items
        if list_items:
            tag = "ul" if list_type == "bullet" else "ol"
            items = "".join(f"<li>{_format_inline(item)}</li>" for item in list_items)
            blocks.append(f"<{tag}>{items}</{tag}>")
        list_type = None
        list_items = []

    for line in lines:
        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            blocks.append(
                f"<h{level + 1}>{_format_inline(heading.group(2).strip())}</h{level + 1}>"
            )
            continue

        bullet = _BULLET_RE.match(line)
        numbered = _NUMBERED_RE.match(line)
        current_list_type = "bullet" if bullet else "numbered" if numbered else None
        if current_list_type:
            flush_paragraph()
            if list_type and list_type != current_list_type:
                flush_list()
            list_type = current_list_type
            list_items.append((bullet or numbered).group(1).strip())
            continue

        flush_list()
        paragraph.append(line)

    flush_paragraph()
    flush_list()
    formatted = "".join(blocks)

    def restore_image(match: re.Match) -> str:
        return images[int(match.group(1))]

    return re.sub(r"\x00IMAGE(\d+)\x00", restore_image, formatted)