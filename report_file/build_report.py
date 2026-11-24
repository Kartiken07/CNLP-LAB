import html
import os
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame,
                                Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle,
                                XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

BASE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(BASE, "content")
OUTPUT = os.path.join(BASE, "lab_report.pdf")

BLACK = colors.HexColor("#000000")
PALE = colors.HexColor("#F0F0F0")

SECTION_LABELS = {
    "Abstract": "Abstract",
    "Description": "Description of code",
    "Code": "Implemented code",
    "Tables": "Results and comparison tables",
    "Figures": "Figures",
}

styles = getSampleStyleSheet()
S = {
    "toc_title": ParagraphStyle("toc_title", parent=styles["Heading1"], fontSize=17,
                                leading=21, textColor=BLACK, spaceAfter=16),
    "chapter": ParagraphStyle("chapter", parent=styles["Heading1"], fontSize=16,
                              leading=20, textColor=BLACK, spaceBefore=4, spaceAfter=14),
    "section": ParagraphStyle("section", parent=styles["Heading2"], fontSize=12.5,
                              leading=16, textColor=BLACK, spaceBefore=14, spaceAfter=6),
    "caption": ParagraphStyle("caption", parent=styles["Heading3"], fontSize=10,
                              leading=13, textColor=BLACK, spaceBefore=10, spaceAfter=5),
    "body": ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                           fontSize=9.6, leading=13.4, textColor=BLACK,
                           alignment=TA_LEFT, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", parent=styles["BodyText"], fontName="Helvetica",
                             fontSize=9.6, leading=13.4, textColor=BLACK,
                             leftIndent=14, bulletIndent=4, spaceAfter=3),
    "code": ParagraphStyle("code", parent=styles["Code"], fontName="Courier",
                           fontSize=7.1, leading=9.1, textColor=BLACK,
                           backColor=PALE, borderPadding=5, leftIndent=4,
                           spaceBefore=2, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.4, leading=9.2,
                           textColor=BLACK),
    "cell_head": ParagraphStyle("cell_head", fontName="Helvetica-Bold", fontSize=7.6,
                                leading=9.4, textColor=BLACK),
    "figcap": ParagraphStyle("figcap", fontName="Helvetica-Oblique", fontSize=8.5,
                             leading=11, textColor=BLACK, alignment=TA_CENTER,
                             spaceBefore=4, spaceAfter=12),
    "toc": ParagraphStyle("toc", fontName="Helvetica", fontSize=10, leading=15,
                          textColor=BLACK),
}


def clean_inline(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return html.escape(text)


def ascii_only(text):
    return "".join(ch if ord(ch) < 256 else "?" for ch in text)


def parse_md(path):
    data = {"title": "", "abstract": [], "description": [], "code": [],
            "tables": [], "figures": []}
    section = None
    subsection = ""
    para = []
    code_lines = []
    in_code = False
    table_rows = []

    def flush_para():
        nonlocal para
        if para:
            text = " ".join(x.strip() for x in para).strip()
            if text and section in ("Abstract", "Description"):
                data[section.lower()].append(text)
            para = []

    def flush_table():
        nonlocal table_rows
        if table_rows:
            rows = []
            for row in table_rows:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    continue
                rows.append(cells)
            if rows:
                if section == "Figures":
                    for row in rows[1:] if any("path" in c.lower() for c in rows[0]) else rows:
                        if len(row) >= 2 and row[0].lower() != "none":
                            data["figures"].append((row[0].strip(), row[1].strip()))
                else:
                    data["tables"].append((subsection, rows))
            table_rows = []

    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if line.startswith("```"):
            if in_code:
                data["code"].append((subsection, "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                flush_para()
                flush_table()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("# "):
            data["title"] = line[2:].strip()
        elif line.startswith("## "):
            flush_para()
            flush_table()
            key = line[3:].strip()
            section = key if key in SECTION_LABELS else ("Description" if key.startswith("Description") else key)
            subsection = ""
        elif line.startswith("### "):
            flush_para()
            flush_table()
            subsection = line[4:].strip()
        elif line.startswith("|"):
            flush_para()
            table_rows.append(line)
        elif not line.strip():
            flush_para()
            flush_table()
        else:
            flush_table()
            para.append(line)
    flush_para()
    flush_table()
    if in_code and code_lines:
        data["code"].append((subsection, "\n".join(code_lines)))
    return data


def wrap_code(code, width=104):
    out = []
    for line in code.split("\n"):
        line = ascii_only(line.rstrip())
        if len(line) <= width:
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, width=width, subsequent_indent="    ",
                                     break_long_words=True, break_on_hyphens=False))
    return "\n".join(out)


def chapter(data, number, story):
    title = re.sub(r"^(?:EXP|Experiment)?[\s-]*\d+\s*[\.\):-]?\s*", "", data["title"],
                   flags=re.IGNORECASE).strip()
    heading = f"Experiment {number}: {title}" if title else f"Experiment {number}"
    key = f"exp{number}"

    story.append(PageBreak())
    chapter_para = Paragraph(clean_inline(heading), S["chapter"])
    chapter_para.toc_level = 0
    chapter_para.toc_key = key
    story.append(chapter_para)

    def section(title_text, key_suffix):
        sec_para = Paragraph(clean_inline(title_text), S["section"])
        sec_para.toc_level = 1
        sec_para.toc_key = key + "_" + key_suffix
        story.append(sec_para)

    if data["abstract"]:
        section("Abstract", "abstract")
        for para in data["abstract"]:
            story.append(Paragraph(clean_inline(para), S["body"]))

    if data["description"]:
        section("Description of code", "description")
        for para in data["description"]:
            story.append(Paragraph(clean_inline(para), S["body"]))

    if data["code"]:
        section("Implemented code", "code")
        for caption, code in data["code"]:
            if caption:
                story.append(Paragraph(clean_inline(caption), S["caption"]))
            story.append(XPreformatted(wrap_code(code), S["code"]))

    if data["tables"]:
        section("Results and comparison tables", "tables")
        for caption, rows in data["tables"]:
            story.append(CondPageBreak(90))
            if caption:
                story.append(Paragraph(clean_inline(caption), S["caption"]))
            ncols = max(len(r) for r in rows)
            rows = [r + [""] * (ncols - len(r)) for r in rows]
            weights = []
            for col in range(ncols):
                longest = max(len(r[col]) for r in rows)
                weights.append(min(max(longest, 10), 46))
            total = float(sum(weights))
            usable = A4[0] - 4.2 * cm
            col_widths = [usable * w / total for w in weights]
            table_data = []
            for ri, row in enumerate(rows):
                style = S["cell_head"] if ri == 0 else S["cell"]
                table_data.append([Paragraph(clean_inline(c), style) for c in row])
            table = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.45, BLACK),
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(table)
            story.append(Spacer(1, 10))

    if data["figures"]:
        section("Figures", "figures")
        for path, caption in data["figures"]:
            if not os.path.exists(path):
                story.append(Paragraph(f"Missing image: {os.path.basename(path)}", S["body"]))
                continue
            max_w = A4[0] - 4.5 * cm
            max_h = 12.7 * cm
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                iw, ih = im.size
            scale = min(max_w / iw, max_h / ih, 1.6)
            image = Image(path, width=iw * scale, height=ih * scale)
            image.hAlign = "CENTER"
            story.append(CondPageBreak(image.drawHeight + 60))
            story.append(KeepTogether([image, Paragraph(clean_inline(caption), S["figcap"])]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLACK)
    canvas.setLineWidth(0.5)
    canvas.line(2.1 * cm, 1.35 * cm, A4[0] - 2.1 * cm, 1.35 * cm)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(BLACK)
    canvas.drawRightString(A4[0] - 2.1 * cm, 0.95 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


def build():
    def after_flowable(flowable):
        level = getattr(flowable, "toc_level", None)
        if level is not None:
            text = flowable.getPlainText()
            key = getattr(flowable, "toc_key", None)
            if key:
                doc.canv.bookmarkPage(key)
                doc.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            doc.notify("TOCEntry", (level, text, doc.page, key))

    doc = BaseDocTemplate(OUTPUT, pagesize=A4, title="CLNLP Lab Report",
                          author="CLNLP Lab", leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                          topMargin=1.9 * cm, bottomMargin=1.9 * cm,
                          afterFlowable=after_flowable)
    doc.afterFlowable = after_flowable
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])

    story = []
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="Helvetica-Bold", fontSize=10.5, leading=19,
                       textColor=BLACK, leftIndent=0),
        ParagraphStyle("toc1", fontName="Helvetica", fontSize=9.2, leading=15,
                       textColor=BLACK, leftIndent=18),
    ]
    story.append(Paragraph("Contents", S["toc_title"]))
    story.append(toc)

    for n in range(1, 11):
        chapter(parse_md(os.path.join(CONTENT, f"exp{n}.md")), n, story)

    doc.multiBuild(story)
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    build()
