"""Build the Arabic stage-58 report from the audited experiment records."""
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_DIRECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


STEM = Path("reports/58_pud_policy_results_ar")
METHODS = "كيف أُجريت المقارنة؟"


def report_lines(result, rows, audit):
    outputs = result["outputs"]
    comparisons = result["comparisons"]
    timed_ar = outputs["ar"]["time"]
    fixed_ar = outputs["ar"]["updates"]
    timed_en = outputs["en"]["time"]
    fixed_en = outputs["en"]["updates"]
    table = [
        "| اللغة | الميزانية | خسارة العشوائي | خسارة المحاذاة | فرق الخسارة | فاصل 95% للفرق |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for language, label in [("ar", "العربية"), ("en", "الإنجليزية")]:
        for mode, budget in [("time", "3 ثوانٍ"), ("updates", "128 تحديثًا")]:
            random_loss = outputs[language][mode]["random"]["final_loss"]
            aligned_loss = outputs[language][mode]["alignment"]["final_loss"]
            contrast = comparisons[language][mode]
            low, high = contrast["ci95"]
            table.append(
                f"| {label} | {budget} | {random_loss:.4f} | {aligned_loss:.4f} | "
                f"{contrast['difference']:+.5f} | [{low:+.5f}, {high:+.5f}] |"
            )
    time_rows = [row for row in rows if row["mode"] == "time"]
    metadata = result["metadata"]
    ar_ratio = fixed_ar["alignment"]["actual_seconds"] / fixed_ar["random"]["actual_seconds"]
    target_chunks = {
        key: sum(fold["target_chunks"] for fold in coverage)
        for key, coverage in result["coverage"].items()
    }
    return [
        "# محاذاة التحديث تحت مهلة ثلاث ثوانٍ",
        "التقرير 58 من مشروع WAPLV  |  18 نوفمبر 2026",
        "في اختبار العربية، حقق الاختيار العشوائي خسارة أقل عندما مُنحت كل سياسة ثلاث ثوانٍ للتدريب. "
        f"أكمل العشوائي في المتوسط {timed_ar['random']['updates']:.0f} تحديثًا، وأكملت المحاذاة "
        f"{timed_ar['alignment']['updates']:.0f} تحديثًا. وعندما حصلت السياستان على 128 تحديثًا لكل منهما، "
        "حققت المحاذاة خسارة أقل قليلًا. ظهر النمط نفسه في الإنجليزية.",
        "## الخسارة النهائية حسب ميزانية التدريب",
        "الخسارة هي متوسط سالب اللوغاريتم الطبيعي للاحتمال الذي أعطاه النموذج للبايت التالي الصحيح. القيمة الأصغر أفضل. "
        "يطرح عمود الفرق خسارة العشوائي من خسارة المحاذاة. يشير الفرق الموجب إلى تفوق العشوائي، والسالب إلى تفوق المحاذاة.",
        *table,
        "المقارنة العربية ذات الثلاث ثوانٍ هي النتيجة الرئيسية المحددة في بروتوكول المرحلة 57. "
        "تصف الصفوف الثلاثة الأخرى المقارنات الثانوية. يُحسب كل فرق قبل تقريب الأرقام المعروضة.",
        "## ماذا كلّف اختيار المرشح؟",
        "تجرّب المحاذاة أربعة تحديثات وتحسب تدرج خسارة نصّ توجيهي عند كل قرار، ثم تحتفظ بالتحديث الذي تتجه "
        "معاملاته أكثر نحو خفض تلك الخسارة. أما الاختيار العشوائي فينفّذ تحديث المرشح المسحوب مباشرة.",
        f"استغرقت التحديثات العربية الـ128 في المتوسط {fixed_ar['alignment']['actual_seconds']:.2f} ثانية للمحاذاة "
        f"مقابل {fixed_ar['random']['actual_seconds']:.2f} ثانية للعشوائي، أي نحو {ar_ratio:.1f} ضعف الزمن. "
        "كانت فائدة المحاذاة الصغيرة في خفض الخسارة عند تثبيت عدد التحديثات مصحوبة بكلفة جعلت العشوائي يتقدم بفارق كبير عند تثبيت الوقت.",
        f"في الإنجليزية، أكمل العشوائي {timed_en['random']['updates']:.0f} تحديثًا خلال المهلة، مقابل "
        f"{timed_en['alignment']['updates']:.0f} للمحاذاة. واستغرقت التحديثات الـ128 نحو "
        f"{fixed_en['random']['actual_seconds']:.2f} ثانية للعشوائي و{fixed_en['alignment']['actual_seconds']:.2f} ثانية للمحاذاة.",
        f"## {METHODS}",
        "استُخدمت ألف جملة متوازية بالعربية والإنجليزية من PUD، إصدار Universal Dependencies ‏r2.15. "
        "في كل دورة من عشر دورات، استُخدمت ثماني طيات للتدريب، وأول 64 مقطعًا من طية تاسعة للتوجيه، "
        f"والطية الباقية للتقييم. بلغ مجموع مقاطع التقييم عبر الطيات {target_chunks['ar']} للعربية و{target_chunks['en']} للإنجليزية، "
        "في كل مقطع 64 انتقالًا بين البايتات.",
        "بدأ نموذج ByteTransformer ذو 35,776 معاملًا بتهيئة عشوائية. بعد 32 تحديث إحماء، بدأت السياستان من حالة النموذج والمحسّن نفسها، "
        "واستخدمتا جدول المرشحين نفسه لكل طية وبذرة. حُسبت درجة المحاذاة من التغير الفعلي في المعاملات بعد تحديث AdamW، "
        "مع طرح المتوسط من فروق معاملات رأس الإخراج، ومن تدرج الخسارة على نص التوجيه. قيس الأداء النهائي على نص الطية المخصصة للتقييم.",
        f"اكتملت {len(rows)} عملية تدريب في {result['seconds']/60:.1f} دقيقة. شملت كل لغة عشر طيات وثماني بذور، وتجربتين زمنيتين لكل سياسة "
        "وتجربة واحدة ثابتة التحديثات. جُمعت الخسائر بحساب متوسط التكرارات ثم متوسط الطيات لكل بذرة. "
        "تعطي إعادة المعاينة المقترنة لفروق البذور الثماني، 10,000 مرة، فاصل الثقة المعروض لكل مقارنة.",
        "## حدود القراءة",
        "غطى المؤقت تجارب المرشحين وتدرج التوجيه والاختيار والتحديث. وقع الإحماء وتوليد الجداول والتقييم خارج المهلة. "
        "احتفظ التنفيذ بآخر قرار اكتمل داخل ثلاث ثوانٍ، وسجّل زمن المحاولة المتأخرة ضمن الزمن الفعلي. "
        f"بلغ أكبر تجاوز مسجّل {max(row['overrun_seconds'] for row in time_rows):.3f} ثانية. "
        "نُسخت حالة النموذج لدى السياستين للسماح برفض التحديث المتأخر.",
        "حصلت المحاذاة على معلومات إضافية من نص التوجيه، فتقيس التجربة أثر سياسة الاختيار كاملة. "
        "ترتبط فواصل الثقة بهذا المصدر والبذور الثماني، وتتشارك الطيات بعض نصوص التدريب. "
        "سبق استخدام PUD في المشروع، وتتكون النسختان العربية والإنجليزية من نصوص متوازية. "
        "تقتصر النتيجة الزمنية على هذا التنفيذ والنموذج والجهاز.",
        "## التحقق وإمكان المراجعة",
        f"نجح {audit['tests']} اختبارًا قبل بدء التجربة. أُعيد حساب الخسائر المجمعة وفروق البذور وفواصل الثقة من سجلات التشغيل، "
        "وتحقّق التدقيق من البصمات وترتيب الحالات والاختيارات العشوائية والالتزام بالمهلة. فُحصت متوسطات العربية بعد اكتمالها "
        "وأثناء استمرار الإنجليزية، واستمر التشغيل وفق التصميم المنشور نفسه.",
        f"جرى التنفيذ على معالج {metadata['cpu'].split(':')[-1].strip()} بخيط PyTorch واحد، "
        f"باستخدام PyTorch {metadata['torch']} وNumPy {metadata['numpy']} وPython {metadata['python']}.",
        "يسجل الملف results/pud-policy.json فروق البذور الثماني وتغطية الطيات. يحفظ results/pud-policy-records.jsonl "
        "الخسائر والتحديثات والأزمنة لكل تشغيل، وترد نتيجة التدقيق في results/pud-policy-audit.json.",
    ]


def rtl_text(paragraph, text, *, size=12, bold=False, rtl=True):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT if rtl else WD_ALIGN_PARAGRAPH.CENTER
    if rtl:
        # "start" follows the paragraph's reading direction in Word and LibreOffice.
        paragraph._p.get_or_add_pPr().find(qn("w:jc")).set(qn("w:val"), "start")
    bidi = OxmlElement("w:bidi")
    bidi.set(qn("w:val"), "1" if rtl else "0")
    paragraph._p.get_or_add_pPr().append(bidi)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(size)
    props = run._r.get_or_add_rPr()
    props.rFonts.set(qn("w:cs"), "Arial")
    size_cs = OxmlElement("w:szCs")
    size_cs.set(qn("w:val"), str(int(2 * size)))
    props.append(size_cs)
    if bold:
        props.append(OxmlElement("w:bCs"))
    return run


def write_word(lines):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(1.8)
    section.top_margin = section.bottom_margin = Cm(1.6)
    normal = doc.styles["Normal"].paragraph_format
    normal.space_after, normal.line_spacing = Pt(8), 1.25
    table = None
    for line in lines:
        if line.startswith("|"):
            if line.startswith("| ---"):
                continue
            header = table is None
            if header:
                table = doc.add_table(rows=0, cols=6)
                table.table_direction = WD_TABLE_DIRECTION.RTL
                table.style = "Light Shading Accent 1"
                table.autofit = False
            cells = table.add_row().cells
            for index, (value, cell) in enumerate(zip(line.strip("|").split("|"), cells, strict=True)):
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(3)
                rtl_text(paragraph, value.strip(), bold=header, size=10, rtl=header or index < 2)
            continue
        level = len(line) - len(line.lstrip("#"))
        text = line.lstrip("# ")
        paragraph = doc.add_heading(level=level) if level else doc.add_paragraph()
        if text == METHODS:
            paragraph.paragraph_format.page_break_before = True
        run = rtl_text(paragraph, text, size={1: 23, 2: 16}.get(level, 12), bold=bool(level))
        if level:
            run.font.color.rgb = RGBColor.from_string("174B68")
    doc.core_properties.title = lines[0].lstrip("# ")
    doc.save(STEM.with_suffix(".docx"))


def main():
    source = Path("results/pud-policy.json")
    audit = json.loads(Path("results/pud-policy-audit.json").read_text())
    if audit["status"] != "passed" or hashlib.sha256(source.read_bytes()).hexdigest() != audit["result_sha256"]:
        raise ValueError("Expected the audited experiment summary")
    result = json.loads(source.read_text())
    journal = Path("results/pud-policy-records.jsonl").read_bytes()
    if hashlib.sha256(journal).hexdigest() != result["records_sha256"]:
        raise ValueError("Run journal differs from the audited record")
    rows = [json.loads(row) for row in journal.splitlines()]
    lines = report_lines(result, rows, audit)
    markdown = "\n".join(
        line if line.startswith("|") else f"\n{line}\n" for line in lines
    ).strip()
    STEM.with_suffix(".md").write_text(markdown + "\n")
    write_word(lines)


if __name__ == "__main__":
    main()
