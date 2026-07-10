# Based on https://github.com/theislab/scanpy/blob/master/docs/extensions/typed_returns.py
# Adjusted so the Returns section renders as a bullet list that matches the
# Parameters list: one bullet per return value, with the name in bold, the type
# as a cross-reference, and the description after an en-dash.
from __future__ import annotations

import re
from collections.abc import Iterable

from sphinx.application import Sphinx
from sphinx.ext.napoleon import NumpyDocstring


def _render_type(type_: str) -> str:
    """Render a return type: a simple dotted name as a cross-reference, anything
    else (``dict[...]``, ``tuple[...]``, ...) as an inline literal."""
    type_ = type_.strip()
    return f":class:`~{type_}`" if re.fullmatch(r"[\w.]+", type_) else f"``{type_}``"


def _returns_bullets(lines: Iterable[str]) -> list[str]:
    """Group a numpy ``Returns`` block into one bullet per return value.

    A non-indented ``name : type`` (or bare ``name``) line starts a new entry;
    the indented lines beneath it are its description.
    """
    entries: list[list] = []  # each: [name, rendered_type | None, [desc lines]]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indented = line[:1].isspace()
        if not indented and (m := re.fullmatch(r"(?P<name>\w+)\s*:\s*(?P<type>.*)", stripped)):
            type_ = m["type"].strip()
            entries.append([m["name"], _render_type(type_) if type_ else None, []])
        elif not indented and re.fullmatch(r"\w+", stripped):
            entries.append([stripped, None, []])
        elif entries:
            entries[-1][2].append(stripped)

    bullets = []
    for name, rendered, desc in entries:
        head = f"**{name}** ({rendered})" if rendered else f"**{name}**"
        text = " ".join(desc).strip()
        bullets.append(f"* {head} -- {text}" if text else f"* {head}")
    return bullets


def _parse_returns_section(self: NumpyDocstring, section: str) -> list[str]:
    lines_raw = self._dedent(self._consume_to_next_section())
    if lines_raw and lines_raw[0] == ":":
        del lines_raw[0]
    bullets = _returns_bullets(lines_raw)
    lines = self._format_block(":returns: ", bullets) if bullets else []
    if lines and lines[-1]:
        lines.append("")
    return lines


def setup(app: Sphinx):
    """Set app."""
    NumpyDocstring._parse_returns_section = _parse_returns_section
    # Only monkeypatches a read-time docstring transform, so it is safe under
    # Sphinx's parallel build (`sphinx -j auto`, as Read the Docs uses).
    return {"parallel_read_safe": True, "parallel_write_safe": True}
