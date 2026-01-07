import re
from pathlib import Path
from tqdm import tqdm

folder_md = Path(r"d:\work\Obsidian\TFlexCAD API")


def remove_fragments(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="ignore")    
    old_len = len(text)
    patterns = [
        r"```vb[\s\S]*?```",
        r"```cpp[\s\S]*?```",
        "\n---  \n",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    if len(text) < old_len:
        file_path.write_text(text, encoding="utf-8")
    
files_md = list(folder_md.rglob("*.md"))
for file_path in tqdm(files_md, desc="Cleaning MD files"):
    remove_fragments(file_path)