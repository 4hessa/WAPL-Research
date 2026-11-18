"""Render the stage-59 Arabic Markdown report as an editable Word document."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_DIRECTION
from docx.shared import Cm, Pt, RGBColor

from tools.write_pud_policy_report import rtl_text

STEM = Path("reports/59_pud_policy_profile_ar")


def write_table(document, lines):
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    header, _, *values = rows
    table = document.add_table(rows=0, cols=len(header))
    table.table_direction = WD_TABLE_DIRECTION.RTL
    table.style = "Light Shading Accent 1"
    table.autofit = False
    widths = [Cm(9), Cm(4), Cm(4)]
    for column, width in zip(table.columns, widths, strict=True):
        column.width = width
    for row_index, row in enumerate([header, *values]):
        cells = table.add_row().cells
        for i, (cell, text) in enumerate(zip(cells, row, strict=True)):
            cell.width = widths[i]
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(4)
            rtl_text(paragraph, text, bold=row_index == 0, size=11, rtl=i == 0 or row_index == 0)


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(1.8)
    section.top_margin = section.bottom_margin = Cm(1.6)
    style = doc.styles["Normal"].paragraph_format
    style.space_after, style.line_spacing = Pt(8), 1.2
    content = STEM.with_suffix(".md").read_text().strip().split("\n\n")
    for block in content:
        if block.startswith("|"):
            write_table(doc, block.splitlines())
            continue
        level = len(block) - len(block.lstrip("#"))
        paragraph = doc.add_heading(level=level) if level else doc.add_paragraph()
        text = block.lstrip("# ")
        if text == "طريقة القياس وحدوده":
            paragraph.paragraph_format.page_break_before = True
        run = rtl_text(paragraph, text, size={1: 22, 2: 16}.get(level, 12), bold=bool(level))
        if level:
            run.font.color.rgb = RGBColor.from_string("174B68")
    doc.core_properties.title = content[0].lstrip("# ")
    doc.save(STEM.with_suffix(".docx"))


if __name__ == "__main__":
    main()
