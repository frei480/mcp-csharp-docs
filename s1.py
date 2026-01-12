import re
from pathlib import Path

pattern = re.compile(
    r"<OBJECT[^>]*?>.*?"
    r'<param name="Name" value="(.*?)">.*?'
    r'<param name="Local" value="(.*?)">.*?'
    r"</OBJECT>",
    re.S | re.I,
)
files = {
    "TFlexAPI.hhc": "index1.html",
    "TFlexAPI.hhk": "search1.html",
}
header = """<HTML>
    <HEAD>
    <TITLE>T-FLEX CAD Open API</TITLE>
    <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=utf-8">
  </HEAD>"""
for file, html_file in files.items():
    text = Path(file).read_text(encoding="cp1251", errors="ignore")
    text = pattern.sub(r'<a href="\2">\1</a>', text)
    text = re.sub("<HTML>", header, text)
    Path(html_file).write_text(text, encoding="utf-8")
