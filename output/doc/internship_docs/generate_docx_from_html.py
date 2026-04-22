from __future__ import annotations

import html
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


DOCX_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"


def w_tag(name: str) -> str:
    return f"{{{DOCX_NS}}}{name}"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def paragraph_xml(
    text: str,
    *,
    align: str | None = None,
    bold: bool = False,
    italic: bool = False,
    size: int = 28,
    before: int = 0,
    after: int = 120,
) -> str:
    if not text:
        text = ""
    ppr_parts: list[str] = []
    if align:
        ppr_parts.append(f'<w:jc w:val="{align}"/>')
    if before or after:
        ppr_parts.append(f'<w:spacing w:before="{before}" w:after="{after}"/>')
    ppr = f"<w:pPr>{''.join(ppr_parts)}</w:pPr>" if ppr_parts else ""

    run_props = [
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:cs="Times New Roman" w:eastAsia="Times New Roman"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>',
    ]
    if bold:
        run_props.append("<w:b/>")
    if italic:
        run_props.append("<w:i/>")
    rpr = f"<w:rPr>{''.join(run_props)}</w:rPr>"

    segments = text.split("\n")
    run_body: list[str] = []
    for index, segment in enumerate(segments):
        if index > 0:
            run_body.append("<w:br/>")
        if segment:
            run_body.append(f"<w:t xml:space=\"preserve\">{esc(segment)}</w:t>")
        else:
            run_body.append("<w:t xml:space=\"preserve\"></w:t>")
    return f"<w:p>{ppr}<w:r>{rpr}{''.join(run_body)}</w:r></w:p>"


def table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    if max_cols == 0:
        return ""
    grid_cols = "".join('<w:gridCol w:w="2400"/>' for _ in range(max_cols))
    body_rows: list[str] = []
    for row_index, row in enumerate(rows):
        cells: list[str] = []
        for cell_text in row:
            text = normalize_text(cell_text)
            if not text:
                text = " "
            paragraphs = [paragraph_xml(line, bold=row_index == 0, size=28, after=0) for line in text.split("\n")]
            cell = (
                "<w:tc>"
                "<w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f"{''.join(paragraphs)}"
                "</w:tc>"
            )
            cells.append(cell)
        while len(cells) < max_cols:
            cells.append(
                "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f"{paragraph_xml(' ', after=0)}"
                "</w:tc>"
            )
        body_rows.append(f"<w:tr>{''.join(cells)}</w:tr>")
    tbl_pr = (
        "<w:tblPr>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
        "</w:tblBorders>"
        "</w:tblPr>"
    )
    return f"<w:tbl>{tbl_pr}<w:tblGrid>{grid_cols}</w:tblGrid>{''.join(body_rows)}</w:tbl>"


def extract_body_tree(html_text: str) -> ET.Element:
    match = re.search(r"<body[^>]*>(.*)</body>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("No <body> found in HTML source.")
    body = match.group(1)
    body = body.replace("<br>", "<br />").replace("<br/>", "<br />")
    wrapped = f"<root>{body}</root>"
    return ET.fromstring(wrapped)


def inline_text(node: ET.Element) -> str:
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        if child.tag == "br":
            parts.append("\n")
        else:
            parts.append(inline_text(child))
        if child.tail:
            parts.append(child.tail)
    return normalize_text("".join(parts))


def list_items(node: ET.Element, ordered: bool) -> list[str]:
    items: list[str] = []
    for index, child in enumerate(node.findall("li"), start=1):
        text = inline_text(child)
        prefix = f"{index}. " if ordered else "• "
        items.append(prefix + text)
    return items


def table_rows(node: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in node.findall("tr"):
        cells: list[str] = []
        for cell in list(tr):
            if cell.tag not in {"th", "td"}:
                continue
            lines: list[str] = []
            if cell.text and cell.text.strip():
                lines.append(normalize_text(cell.text))
            for child in cell:
                if child.tag == "br":
                    lines.append("")
                else:
                    text = inline_text(child)
                    if text:
                        lines.append(text)
                if child.tail and child.tail.strip():
                    lines.append(normalize_text(child.tail))
            cell_text = "\n".join(line for line in lines if line is not None).strip()
            cells.append(cell_text)
        rows.append(cells)
    return rows


def blocks_from_tree(root: ET.Element) -> list[str]:
    blocks: list[str] = []
    for child in root:
        if child.tag == "p":
            text = inline_text(child)
            align = "center" if "center" in child.attrib.get("class", "") else None
            blocks.append(paragraph_xml(text, align=align, after=120))
        elif child.tag == "h1":
            blocks.append(paragraph_xml(inline_text(child), align="center", bold=True, size=36, before=120, after=120))
        elif child.tag == "h2":
            blocks.append(paragraph_xml(inline_text(child), align="center", bold=True, size=30, before=160, after=120))
        elif child.tag == "h3":
            blocks.append(paragraph_xml(inline_text(child), align="center", bold=True, size=28, before=120, after=120))
        elif child.tag == "ul":
            for item in list_items(child, ordered=False):
                blocks.append(paragraph_xml(item, after=80))
        elif child.tag == "ol":
            for item in list_items(child, ordered=True):
                blocks.append(paragraph_xml(item, after=80))
        elif child.tag == "table":
            blocks.append(table_xml(table_rows(child)))
        elif child.tag == "div":
            if "spacer" in child.attrib.get("class", ""):
                blocks.append(paragraph_xml(" ", after=200))
            else:
                nested = blocks_from_tree(child)
                blocks.extend(nested)
        else:
            text = inline_text(child)
            if text:
                blocks.append(paragraph_xml(text, after=120))
    return blocks


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def rels_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL_NS}">
  <Relationship Id="rId1" Type="{DOC_REL_NS}/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="{DOC_REL_NS}/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="{DOC_REL_NS}/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def document_rels_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL_NS}"></Relationships>"""


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{DOCX_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/>
        <w:sz w:val="28"/>
        <w:szCs w:val="28"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="120"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
    </w:rPr>
  </w:style>
</w:styles>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>"""


def core_xml(title: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="{CP_NS}" xmlns:dc="{DC_NS}" xmlns:dcterms="{DCTERMS_NS}" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="{XSI_NS}">
  <dc:title>{esc(title)}</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def document_xml(blocks: Iterable[str]) -> str:
    body = "".join(blocks)
    sect_pr = """
<w:sectPr>
  <w:pgSz w:w="11906" w:h="16838"/>
  <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
</w:sectPr>"""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{DOCX_NS}">
  <w:body>{body}{sect_pr}</w:body>
</w:document>"""


@dataclass
class DocSpec:
    html_name: str
    output_name: str
    title: str


DOCS = [
    DocSpec(
        "04_syllabus_nlp_methods_for_disinformation_detection.html",
        "Научная стажировка - задание 4 - Силлабус.docx",
        "Силлабус - NLP Methods for Disinformation Detection",
    ),
    DocSpec(
        "05_article_title_abstract_journals.html",
        "Научная стажировка - задание 5 - Статья и журналы.docx",
        "Задание 5 - Статья и журналы",
    ),
    DocSpec(
        "06_scopus_zotero_bibliography.html",
        "Научная стажировка - задание 6 - Scopus и библиография.docx",
        "Задание 6 - Scopus и библиография",
    ),
    DocSpec(
        "07_project_passport.html",
        "Научная стажировка - задание 7 - Паспорт проекта.docx",
        "Задание 7 - Паспорт проекта",
    ),
    DocSpec(
        "updated_internship_report.html",
        "Отчет о научной стажировке - обновленный.docx",
        "Отчет о научной стажировке",
    ),
]


def write_docx(html_path: Path, out_path: Path, title: str) -> None:
    root = extract_body_tree(html_path.read_text(encoding="utf-8"))
    blocks = blocks_from_tree(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", rels_xml())
        zf.writestr("word/document.xml", document_xml(blocks))
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml())
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("docProps/core.xml", core_xml(title))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "src_html"
    for spec in DOCS:
        html_path = src_dir / spec.html_name
        out_path = base_dir / spec.output_name
        write_docx(html_path, out_path, spec.title)
        print(out_path)


if __name__ == "__main__":
    main()
