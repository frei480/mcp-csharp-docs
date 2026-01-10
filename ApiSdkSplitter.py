import re
from typing import Any

RE_H1 = re.compile(r"^#\s+(.*?)\s+-\s+класс", re.MULTILINE)
RE_NAMESPACE = re.compile(r"\*\*Пространство.*?:\*\*\s*\[(.*?)\]")
RE_ASSEMBLY = re.compile(r"\*\*Сборка:\*\*\s*(.*?)\s+Версия")
RE_CODE_BLOCK = re.compile(r"```.*?\n.*?```", re.DOTALL)
RE_MD_LINK = re.compile(r"\[(?P<name>.*?)\]\((?P<link>[^)]+\.md)\)")


def extract_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    blocks: dict[str, str] = {}

    def repl(m: re.Match[str]) -> str:
        key = f"__CODE_BLOCK_{len(blocks)}__"
        blocks[key] = m.group(0)
        return key

    clean = RE_CODE_BLOCK.sub(repl, text)
    return clean, blocks


def restore_code_blocks(text: str, blocks: dict[str, str]) -> str:
    for k, v in blocks.items():
        text = text.replace(k, v)
    return text


def parse_markdown_table(table: str) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []

    lines = [l.strip() for l in table.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("| ---"):
            continue
        if not line.startswith("|"):
            continue

        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 3:
            continue

        name_cell = cols[1]
        desc = cols[2]

        md_match = RE_MD_LINK.search(name_cell)

        if md_match:
            name = md_match.group("name")
            md_link = md_match.group("link")
        else:
            name = name_cell
            md_link = None

        member_type: str | None = None
        if md_link:
            if md_link.startswith("P_"):
                member_type = "property"
            elif md_link.startswith("M_"):
                member_type = "method"
            elif md_link.startswith("F_"):
                member_type = "field"
            elif md_link.startswith("E_"):
                member_type = "event"
            elif "__ctor" in md_link:
                member_type = "constructor"

        rows.append(
            {
                "name": name,
                "description": desc,
                "md_link": md_link,
                "member_type": member_type,
            }
        )

    return rows


class ApiSdkChunkSplitter:
    def split(self, markdown: str) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        # --- metadata уровня класса ---
        class_name = self._find(RE_H1, markdown)
        namespace = self._find(RE_NAMESPACE, markdown)
        assembly = self._find(RE_ASSEMBLY, markdown)

        # --- code blocks ---
        clean_text, code_blocks = extract_code_blocks(markdown)

        # --- описание класса ---
        header_end = clean_text.find("\n---")
        description = clean_text[:header_end] if header_end != -1 else clean_text

        chunks.append(
            {
                "text": restore_code_blocks(description, code_blocks),
                "metadata": {
                    "type": "class",
                    "class": class_name,
                    "namespace": namespace,
                    "assembly": assembly,
                },
            }
        )

        # --- таблицы ---
        tables = clean_text.split("\n\n|")
        for table in tables[1:]:
            table = "|" + table
            rows = parse_markdown_table(table)

            for row in rows:
                chunks.append(
                    {
                        "text": f"{row['name']} — {row['description']}",
                        "metadata": {
                            "type": "member",
                            "class": class_name,
                            "member": row["name"],
                            "member_type": row["member_type"],
                            "source_md": row["md_link"],
                        },
                    }
                )

        return chunks

    def _find(self, pattern: re.Pattern[str], text: str) -> str | None:
        m = pattern.search(text)
        return m.group(1).strip() if m else None
