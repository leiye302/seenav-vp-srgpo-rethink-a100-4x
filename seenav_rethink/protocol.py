"""One reasoning span, one optional pre-action marker, one native move."""

import re

MARKER = "<RETHINK>"
EOS = "<|im_end|>"
_BOUNDARY = r"[\s\u00ad]*"
_PATTERN = re.compile(
    rf"\s*<think>(?P<reasoning>.*?)</think>{_BOUNDARY}"
    rf"(?:(?P<marker>{MARKER}){_BOUNDARY})?"
    r"<action>\s*(?P<action>[0-7])\s*</action>\s*\Z",
    re.S,
)
_CONTROL = re.compile(r"</?think>|</?action>|<RETHINK>|<\|[^<>]*\|>", re.I)
_BETWEEN_TAGS = re.compile(r"(</think>|<RETHINK>)([\s\u00ad]+)(?=<RETHINK>|<action>)")


def parse_decision(text):
    """Do not infer a marker from reasoning text, or route malformed responses."""
    text = re.sub(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>))+\s*\Z", "", text)
    match = _PATTERN.fullmatch(text)
    if not match:
        return None
    reason = match["reasoning"].strip()
    if not reason or _CONTROL.search(reason):
        return None
    return reason, bool(match["marker"]), int(match["action"])


def for_execution(text):
    """Treat SHY only at the tag boundaries as transparent; do not repair moves."""
    return _BETWEEN_TAGS.sub(lambda m: m[1] + m[2].replace("\u00ad", ""), text)
