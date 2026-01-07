import re
from typing import List, Dict

CODE_BLOCK_PATTERN = re.compile(
    r"```.*?\n.*?```|~~~.*?\n.*?~~~",
    re.DOTALL
)

def extract_code_blocks(text: str):
    """
    Заменяет code blocks плейсхолдерами и возвращает:
    - очищенный текст
    - словарь placeholder -> code block
    """
    code_blocks: Dict[str, str] = {}

    def replacer(match):
        placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks[placeholder] = match.group(0)
        return placeholder

    clean_text = CODE_BLOCK_PATTERN.sub(replacer, text)
    return clean_text, code_blocks

def restore_code_blocks(text: str, code_blocks: Dict[str, str]):
    for placeholder, code in code_blocks.items():
        text = text.replace(placeholder, code)
    return text

class RecursiveMarkdownSplitter:
    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size

        # ВАЖНО: от крупных смысловых к мелким
        self.separators = [
            "\n\n# ",
            "\n\n## ",
            "\n\n### ",
            "\n\n#### ",
            "\n\n##### ",
            "\n\n###### ",            
            "\n\n***\n\n",
            "\n\n",
            "\n",
            " ",
            ""
        ]

    def split_text(self, text: str) -> List[str]:
        clean_text, code_blocks = extract_code_blocks(text)
        chunks = self._recursive_split(clean_text, self.separators)
        return [{"text": restore_code_blocks(chunk, code_blocks).strip()} for chunk in chunks if chunk.strip()]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.max_chunk_size:
            return [text]

        if not separators:
            return self._force_split(text)

        sep = separators[0]

        # Последний fallback — символьный split
        if sep == "":
            return self._force_split(text)

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part

            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(
                        self._recursive_split(current, separators[1:])
                    )
                current = part

        if current:
            chunks.extend(
                self._recursive_split(current, separators[1:])
            )

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """
        Последний уровень — жёсткое деление по длине
        """
        return [
            text[i:i + self.max_chunk_size]
            for i in range(0, len(text), self.max_chunk_size)
        ]
