"""Research-only opt-in HTML byte decoding prototype.

NOT the public CodecCat API, not a complete WHATWG HTML parser. Generic
codeccat.detect/decode and frozen native model remain unchanged.

Requires webencodings only in this research environment. HTMLParser finds
real meta tags, not a regex; parsing still has documented limits for malformed
HTML and raw-text modes. No scripts run or resources are fetched.
"""
from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
from typing import Literal

import webencodings


class HtmlDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class HtmlDecodeResult:
    text: str
    encoding: str
    source: Literal["bom", "override", "transport", "meta", "fallback"]
    used_replacement: bool
    meta_encoding: str | None
    transport_encoding: str | None
    warning: str | None


def _web_label(value: str | None) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 100:
        return None
    try:
        found = webencodings.lookup(value.strip())
    except (ValueError, TypeError):
        return None
    return found.name if found else None


def _http_charset(header: str | None) -> str | None:
    if header is None:
        return None
    if not isinstance(header, str):
        raise TypeError("content_type must be a string or None")
    if len(header) > 2048 or "\r" in header or "\n" in header:
        raise HtmlDecodeError("unsafe content_type length or line break")
    m = Message()
    m["content-type"] = header
    if m.get_content_type() not in ("text/html", "application/xhtml+xml"):
        return None
    params = m.get_params(header="content-type") or []
    matches = [str(v) for k, v in params[1:] if k.lower() == "charset"]
    if len(matches) != 1:
        return None
    return _web_label(matches[0])


class _MetaFinder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.encoding: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag != "meta" or self.encoding:
            return
        fields = {}
        for key, value in attrs:
            if key not in fields and value is not None:
                fields[key] = value
        label = fields.get("charset")
        if label is None and fields.get("http-equiv", "").lower().strip() == "content-type":
            content = fields.get("content", "")
            m = Message()
            m["content-type"] = content
            label = m.get_param("charset", header="content-type")
        if label is None:
            return
        canon = _web_label(str(label))
        if canon in ("utf-16be", "utf-16le"):
            canon = "utf-8"
        if canon:
            self.encoding = canon


def _meta_charset(data: bytes, max_scan_bytes: int) -> str | None:
    finder = _MetaFinder()
    finder.feed(data[:max_scan_bytes].decode("latin-1"))
    finder.close()
    return finder.encoding


def decode_html_prototype(
    data: bytes,
    *,
    content_type: str | None = None,
    override_encoding: str | None = None,
    fallback_encoding: str = "windows-1252",
    errors: Literal["strict", "replace"] = "replace",
    max_bytes: int = 8_000_000,
    max_scan_bytes: int = 1_048_576,
) -> HtmlDecodeResult:
    """Decode explicitly identified HTML bytes, not arbitrary byte strings.

    BOM > explicit trusted caller override > HTTP Content-Type > HTML meta >
    fallback. The output is HTML source Unicode, not rendered text/DOM.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) > max_bytes:
        raise HtmlDecodeError("HTML input exceeds byte limit")
    if not 0 < max_scan_bytes <= max_bytes:
        raise ValueError("invalid scan limit")
    if errors not in ("strict", "replace"):
        raise ValueError("invalid errors policy")
    if override_encoding is not None and _web_label(override_encoding) is None:
        raise HtmlDecodeError("unknown explicit encoding override")
    if _web_label(fallback_encoding) is None:
        raise HtmlDecodeError("unknown fallback encoding")
    transport = _http_charset(content_type)
    meta = None
    if data.startswith(b"\xef\xbb\xbf"):
        chosen,source = "utf-8","bom"
    elif data.startswith(b"\xff\xfe"):
        chosen,source = "utf-16le","bom"
    elif data.startswith(b"\xfe\xff"):
        chosen,source = "utf-16be","bom"
    elif override_encoding is not None:
        chosen,source = _web_label(override_encoding),"override"
    elif transport:
        chosen,source = transport,"transport"
    else:
        meta = _meta_charset(data,max_scan_bytes)
        if meta:
            chosen,source = meta,"meta"
        else:
            chosen,source = _web_label(fallback_encoding),"fallback"
    try:
        text,actual = webencodings.decode(data,chosen,errors=errors)
    except UnicodeError as exc:
        raise HtmlDecodeError("decoding failed under chosen encoding") from exc
    return HtmlDecodeResult(
        text=text, encoding=actual.name, source=source,
        used_replacement=("\ufffd" in text), meta_encoding=meta,
        transport_encoding=transport,
        warning="prototype_only_not_whatwg_conformant" if source in ("meta","fallback") else None,
    )
