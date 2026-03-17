from __future__ import annotations

from typing import Any, Iterable


def _line(ch: str = "=", width: int = 92) -> str:
    return ch * width


def preview_text(value: Any, max_chars: int = 400) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def preview_code(code: str, max_lines: int = 18, max_line_chars: int = 180) -> str:
    lines = code.splitlines()
    shown = lines[:max_lines]
    rendered = []
    for i, line in enumerate(shown, start=1):
        if len(line) > max_line_chars:
            line = line[:max_line_chars] + " ..."
        rendered.append(f"{i:>3} | {line}")
    if len(lines) > max_lines:
        rendered.append(f"... ({len(lines) - max_lines} more lines)")
    return "\n".join(rendered)


def print_block(title: str) -> None:
    print("\n" + _line("="))
    print(f"[ {title} ]")
    print(_line("="))


def print_kv(key: str, value: Any) -> None:
    print(f"- {key}: {value}")


def print_list(label: str, items: Iterable[Any], max_items: int = 8) -> None:
    items_list = list(items)
    print(f"- {label}: {len(items_list)}")
    for item in items_list[:max_items]:
        print(f"  * {item}")
    if len(items_list) > max_items:
        print(f"  * ... ({len(items_list) - max_items} more)")


def print_subheader(title: str) -> None:
    print("\n" + _line("-"))
    print(f"{title}")
    print(_line("-"))
