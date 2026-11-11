"""Convert research Markdown to an RTL Word report without external services."""
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def convert(src, dst):
    doc=Document();section=doc.sections[0]
    section.top_margin=section.bottom_margin=Inches(.75)
    style=doc.styles['Normal'];style.font.name='Arial';style.font.size=Pt(12)
    style.paragraph_format.space_after=Pt(7)
    for line in Path(src).read_text().splitlines():
        if not line.strip():continue
        level=len(line)-len(line.lstrip('#'))
        p=doc.add_heading(line[level:].strip(),min(level,3)) if level else doc.add_paragraph(line)
        bidi=OxmlElement('w:bidi');bidi.set(qn('w:val'),'1');p._p.get_or_add_pPr().append(bidi)
        for r in p.runs:
            r.font.name='Arial'
            rtl=OxmlElement('w:rtl');rtl.set(qn('w:val'),'1');r._r.get_or_add_rPr().append(rtl)
            fonts=OxmlElement('w:rFonts');fonts.set(qn('w:cs'),'Arial');r._r.get_or_add_rPr().append(fonts)
    doc.core_properties.title=Path(src).stem
    doc.core_properties.subject='WAPLV — تقرير بحثي باللغة العربية'
    doc.save(dst)
    reopened=Document(dst)
    assert len(reopened.paragraphs)>5
    print(dst,len(reopened.paragraphs),'paragraphs')
if __name__=='__main__':convert(*sys.argv[1:])
