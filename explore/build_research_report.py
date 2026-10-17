from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "report_work" / "assets"
OUTPUT = ROOT / "Study on Global Multi-Assessment Dynamic Strategies   Chinese.docx"

RED = "B82626"
DARK_RED = "8F1D2C"
NAVY = "2B549C"
LIGHT_BLUE = "E9F0F8"
LIGHT_RED = "F8EBED"
MID_GRAY = "D9D9D9"
LIGHT_GRAY = "F2F2F2"
DARK = "222222"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run_font(run, size=10.5, bold=False, color=DARK, latin="Times New Roman", east_asia="Song.") -> None:
    run.font.name = latin
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    r_fonts.set(qn("w:ascii"), latin)
    r_fonts.set(qn("w:hAnsi"), latin)
    r_fonts.set(qn("w:eastAsia"), east_asia)


def set_paragraph_spacing(paragraph, before=0, after=5, line=1.25) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line


def add_text(doc, text, *, bold=False, color=DARK, size=10.5, align=None, first_line=True, after=5):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if first_line:
        p.paragraph_format.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color)
    set_paragraph_spacing(p, after=after)
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.25)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(item)
        set_run_font(run, size=10.2)


def add_section_heading(doc, title, level=1):
    p = doc.add_paragraph()
    p.style = doc.styles[f"Heading {level}"]
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(5)
    if level == 1:
        marker = p.add_run("■ ")
        set_run_font(marker, size=14, bold=True, color=RED, east_asia="Bold")
        run = p.add_run(title)
        set_run_font(run, size=15, bold=True, color=DARK, east_asia="Bold")
        p_pr = p._p.get_or_add_pPr()
        pbdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "12")
        bottom.set(qn("w:space"), "3")
        bottom.set(qn("w:color"), RED)
        pbdr.append(bottom)
        p_pr.append(pbdr)
    else:
        run = p.add_run(title)
        set_run_font(run, size=12.5 if level == 2 else 11, bold=True, color=DARK, east_asia="Bold")
    return p


def add_equation(doc, equation: str, number: str | None = None):
    from report_math import add_professional_equation
    return add_professional_equation(doc, number)

def add_table(doc, headers, rows, widths=None, font_size=8.8, first_col_bold=False):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    table.style = "Table Grid"
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for i, value in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_shading(cell, RED)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(value))
        set_run_font(r, size=font_size, bold=True, color="FFFFFF", east_asia="Bold")
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if ridx % 2:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(value))
            set_run_font(r, size=font_size, bold=(first_col_bold and i == 0))
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    set_run_font(r, size=9.4, bold=True, color=DARK, east_asia="Bold")
    p_pr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "2")
    bottom.set(qn("w:color"), RED)
    pbdr.append(bottom)
    p_pr.append(pbdr)


def add_figure(doc, filename, caption, width=6.65, source="Source: Yahoo Finance, MAS, by author"):
    add_caption(doc, caption)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run()
    run.add_picture(str(ASSETS / filename), width=Inches(width))
    s = doc.add_paragraph()
    s.paragraph_format.space_after = Pt(4)
    r = s.add_run(source)
    set_run_font(r, size=8, color="666666")


def add_key_box(doc, title, lines):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    cell = table.cell(0, 0)
    set_cell_shading(cell, LIGHT_RED)
    set_cell_margins(cell, top=120, start=140, bottom=120, end=140)
    p = cell.paragraphs[0]
    r = p.add_run(title)
    set_run_font(r, size=11, bold=True, color=DARK_RED, east_asia="Bold")
    for line in lines:
        p = cell.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run("• " + line)
        set_run_font(r, size=9.8)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])
    set_run_font(run, size=8, color="666666")


def add_toc_field(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = 'TOC \\o "1-1" \\h \\z \\u'
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "Select Update Field Generation Directory in Word"
    fld_char3 = OxmlElement("w:fldChar")
    fld_char3.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char, instr_text, fld_char2, text, fld_char3])
    set_run_font(run, size=10)


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.75)
    section.bottom_margin = Cm(1.65)
    section.left_margin = Cm(1.75)
    section.right_margin = Cm(1.75)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Song.")
    for style_name, size, bold in (("Title", 24, True), ("Heading 1", 15, True), ("Heading 2", 12.5, True), ("Heading 3", 11, True)):
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = bold
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Bold")

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = header.add_run("Global multi-asset dynamic defence strategy study October 2026")
    set_run_font(r, size=8, color="777777")
    footer = section.footer.paragraphs[0]
    left = footer.add_run("Study on financial engineering")
    set_run_font(left, size=8, color="777777")
    add_page_number(footer)


def main():
    doc = Document()
    configure_document(doc)

    # Cover
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Cm(4.8)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Global multi-asset dynamic defence strategy study")
    set_run_font(r, size=25, bold=True, color=DARK, east_asia="Bold")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Self-adaptation configuration based on momentum Top10, fully dynamic double-layer risk budget and EWMA risk parity")
    set_run_font(r, size=14, bold=True, color=DARK_RED, east_asia="Bold")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Cm(1.0)
    r = p.add_run("The final product strategy: Ji-chun Shih-hyun's attack")
    set_run_font(r, size=13, bold=True, color=RED, east_asia="Bold")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Cm(4.2)
    r = p.add_run("October 2026, ZXQ0QZ, Financial Engineering Course Project Group")
    set_run_font(r, size=11, color="555555")
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Cm(2.0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Disclaimer: This report is used only for financial engineering research and teaching and does not constitute an investment proposal. History does not represent future performance.")
    set_run_font(r, size=8.8, color="777777")
    doc.add_page_break()

    # Executive summary
    add_section_heading(doc, "Summary and core findings", 1)
    add_text(doc, "The central question studied in the present report is how to combine the fragmentation of multiple assets globally, the continuity of trends and risk budget management without using nominal leverage and relying on a single stock market, so that the portfolio can participate in strong trends while reducing risk exposure in times of market deterioration. The study, based on the average range and classic risk parities, gradually added to the top10, EWMA band, a fully dynamic double-tier risk budget and 0%-30% self-adapted attack kit, which eventually led to the product strategy of \"Swissing the Ball, Gent and Spirit.\"")
    add_section_heading(doc, "Studies completed in the present report", 2)
    add_text(doc, "First, the report established a unified multi-asset research platform. Assets pools cover United States, Japanese, British, European, Mainland China, Hong Kong, China, and Singapore equities, United States and Chinese Treasury debt, gold, crude oil and industrial metals, United States dollars, yen and euro, and bitcoin. All assets are uniformly converted to SGD proceeds and SORA is the calibre of the cash and risk-free interest rate. The code integrates the time zone, date of transaction, foreign exchange direction, signal lag and implementation point, making the strategic results comparable in different markets.")
    add_text(doc, "Second, the report describes the selection of 12 formal strategies by four study phases. The allocation threshold compares 18 assets, power, etc., with the full-assets ERC; the risk model phase compares Top10 against volatility, classic ERC, EWMA-ERC, row down and GARC; the budget phase compares single-layer, static double layer with full-dynamic double layer; and the offensive phase is fixed at 30 to 30 per cent. At each stage, the limitations of the previous programme, the adaptation of ideas and the additional costs are explained in conjunction with the indicator and the funding weight. The risk model is a parallel candidate and is not 12 linear upgrades; nor is the eventual offensive/executed modification abated by pure single parameters.")
    add_text(doc, "Thirdly, the report expands robust evidence to cover the audience. The 12 official strategies listed the results of the sample, the validation period and the complete study, with all categories of weight maps used to compare the configuration mechanism; historical pressures retained nine windows against seven, assuming that shock, general and crisis weights were compared to three core products. Cost multipliers, additional delay in implementation, annual Walk-forward and Deflated Sharpe testing implementation and selection risks. Sensitivity of parameters and the de-BTC experiment are directed at the basic EWMA study path and cannot be considered directly to have been fully adopted; random Top10 and 60/40 do not produce a complete output of the same calibre as that currently applicable.")
    add_text(doc, "Fourth, the report splits returns into four categories of performance. The three modules Shapley further reveal the interaction between the momentum, EWMA and the dynamic budget; factor attribution adds layer by layer, gold, BTC, momentum and status variables, distinguishing between external risk interpretation and internal rules; and real-cost waterfalls singles commissions, swaps, price differentials, market shocks and withholding taxes. Each attribution retains its scope and limits and avoids directly equating the difference in proceeds with independent Alpha.")
    add_section_heading(doc, "Main research findings", 2)
    add_bullets(doc, [
        "The final strategy was to achieve cumulative returns of 441.52 per cent, annualized returns of 16.19 per cent, annualized fluctuations of 10.77 per cent, Sharp 1.32 and maximum retreats of 14.98 per cent. It is not a minimal exit programme, but rather one that best meets current product objectives between elasticity, risk and hand-over.",
        "Pressure performance supports the choice of a dynamic offensive relative to a fixed attack: the epidemic window yield +0.79 per cent, the maximum withdrawal - 5.17 per cent, the 2022 stock debt double-kill gain - 0.43 per cent, and the recovery - 5.18 per cent; and the risk of a crisis-weighted Bootstream loss of 13.50 per cent, down from 17.30 per cent for a fixed attack and 13.86 per cent for ES in 99 per cent of the year, down from 17.49 per cent. However, rapid rebound participation is insufficient, and some of the end indicators remain weaker than the robust core.",
        "20.05-01-01 to 2026-07-31 cumulative proceeds towards the certification period 37.08 per cent, Sharp 1.61, maximum retreat - 8.02 per cent. From 2026-08-01 to 2026-10-01, the lock-in was 44 trading days, only as an observation for which data were not available and not used to prove long-term Alpha.",
        "A copy of the cost of the actual difference is included in the sub-market commissions, swaps, price differentials, ADV square root shocks and withholding taxes, resulting in cumulative gains of 364.12 per cent and Sharp 1.20. Realism weakened the cumulative return of about 77.40 percentage points, but did not change the main ranking of the strategy relative to the benchmark.",
        "The proceeds are not evenly derived from all assets. Equity, bitcoin and commodities contributed 164.58, 149.62 and 129.50 percentage points, respectively; the largest source of the target level was the Bitcoin, Gold and Channel Times Index. Therefore, access to BTC and investmentable asset substitution tests are important for productization.",
        "The simplification of the four factor model is 11.47 per cent for Alpha; 3.45 per cent for external markets and style interpretation only (p=0.06), and 2.85 per cent for re-entry status and internal budget rules (p=0.01). Internal rules have increased the level of interpretation but do not constitute independent external Alpha evidence. Gains should be understood as a combination of market exposure, trends, rule responses and statistical residues.",
        "Sound evidence requires objective interpretation. After doubling costs, Sharp remained at 1.29 and after a one-day delay in implementation at 1.31; however, the probability of Deflad Sharpe under 12 formal strategies was 44.56 per cent, which was not sufficient to rule out multiple testing and sample accident.",
    ])
    add_table(doc, ["Indicators", "GMI-ADG", "GMI-FOA", "GMI-AHRB", "18 Rights, etc."], [
        ["Cumulative proceeds", "441.52%", "362.55%", "270.99%", "211.15%"],
        ["Annualized proceeds", "16.19%", "14.57%", "12.35%", "10.61%"],
        ["Yearic fluctuations", "10.77%", "11.19%", "8.49%", "9.26%"],
        ["Sharpe Ratio", "1.32", "1.15", "1.26", "0.99"],
        ["Maximum Drawdown", "-14.98%", "-15.51%", "-9.75%", "-17.87%"],
        ["The yearning unilateral change of hands", "342.17%", "446.31%", "397.89%", "13.11%"],
    ], font_size=9.1, first_col_bold=True)
    add_text(doc, "Note: The above table shows the results of the study of complete zones, SGD, unilateral 0.05 per cent fixed transaction costs from 2015-09-07 to 2026-07-31. The strategy is judged not only by the ranking of the entire sample, but also by the results of forward validation, stress testing, realistic costs and attribution.", size=8.3, color="666666", first_line=False)
    directory_heading = add_section_heading(doc, "Contents", 1)
    directory_heading.style = doc.styles["Normal"]
    add_toc_field(doc)
    add_text(doc, "Structure of reporting", bold=True, first_line=False)
    add_table(doc, ["Part", "Theme", "Core issues"], [
        ["Part I", "From average range to risk parity", "Why to move from forecasting gains to managing risk contributions"],
        ["Part II", "Policy-level construction", "How to combine the motor, EWMA, double layer risk budget with attack module"],
        ["Part III", "The final strategy and the real deal.", "Performance and product positioning under fixed and real costs"],
        ["Part IV", "Pressure testing and robustness", "History, assumptions, Bootstream, VaR/ES, parameter sensitivity and Walker-Forward"],
        ["Part V", "Attribution of performance", "Four tiers of interpretation for assets, modules, factors and transactions"],
    ], font_size=9.2, widths=[2.2, 4.6, 10.4])
    doc.add_page_break()

    # Part 1
    add_section_heading(doc, "Part I, from average to risk level Price", 1)
    add_text(doc, "The value of large asset allocation is not simply an increase in holding stock, but a fragmentation of sources of income and risk to different economic drivers. Equities mainly reflect growth and risk preferences, national debt reflects interest rates and deflation risks, gold, energy and industrial metals reflect inflation, supply shocks and physical demand, and foreign exchange brings spreads, capital flows and benchmark currency risks into the portfolio. How these assets are combined is the starting point for all subsequent design of the strategy.")
    add_section_heading(doc, "1.1 Average variance establishes a modern asset allocation framework", 2)
    add_text(doc, "For the first time, the Markowitz mean variance model has translated the combination selection system into a “return-risk” trade-off. The weight of the n asset is w, the expected gain is m, the uniting matrix is zirconium, and the expected gain is wTmi and the combination is wTir. The typical form of maximizing the effects of the mean margin without being bound to sell is:")
    add_equation(doc, "max_w  wᵀμ − (γ/2) wᵀΣw,   s.t.  1ᵀw = 1,  w ≥ 0", "(1)")
    add_text(doc, "Risk aversion factor gamma determines which combination investors choose along the effective front line. When gamma is smaller, the model is more concerned with the expected benefits, and weights are concentrated in high-expected income assets; when gamma is larger, the combination is gradually close to the smallest variance combination. The important contribution of this framework is to make it clear that assets should not be evaluated in isolation, and that the value of an asset to a portfolio depends at the same time on its own earnings, fluctuations and its confluence with other assets.")
    add_text(doc, "The same question may also be written as minimizing the group deviation when given the target gain μ0:")
    add_equation(doc, "min_w  wᵀΣw,   s.t.  wᵀμ ≥ μ₀,  1ᵀw = 1,  w ≥ 0", "(2)")
    add_text(doc, "If the target revenue constraint is removed, the smallest global variance group is obtained. It is not sensitive to expected gains, but it is highly sensitive to the coordination matrix and can easily concentrate its weights on a small number of low-variant and related assets. Therefore, the \"unpredicted gain\" does not automatically mean that the combination is more robust, and the estimated error is simply a shift from the millimetre to the calf.")
    add_section_heading(doc, "1.2 The biggest obstacle to the average variance is input error", 2)
    add_text(doc, "In the sample, the optimizer can find a very precise set of weights, but this accuracy is often derived from over-comparisons to historical averages and historical relevance. For active sample length T and number of assets n, the estimated noise in the AC matrix increases rapidly when n/T rises; the average expected gain is more difficult to stabilize than the rate of fluctuations. Changes in parameters for small ranges in the sample may result in significant changes in the best weights, leading to centralized holding, high hand change and a decline in performance outside the sample.")
    add_text(doc, "There are additional problems with the global ETF asset pool: there are different listing dates for different assets, different original currencies, different trading time zones and sudden increases in the relevance of the crisis phase. If the whole sample average and the co-conciliation differentials are replaced directly by the average variance model, the optimized results can easily reflect the random structure of the historical sample rather than stable economic patterns. This is why the rolling mean margin is not directly at the centre of the final strategy in this report.")
    add_table(doc, ["Configure Method", "Reliance on expected benefits", "Whether to use relevance", "Main strengths", "Main limitations"], [
        ["Equal rights", "Yes", "Yes", "Simple, stable, low-handed.", "Fragmentation of funds does not amount to risk dispersion"],
        ["Average variance", "Yes.", "Yes.", "Directly optimized benefit risk balance in theory", "It's highly sensitive to the error estimates for the millimetres and thorium."],
        ["Minimal variance", "Yes", "Yes.", "Avoid directly estimating expected benefits", "It can concentrate on a few low-variant assets"],
        ["Volatility Last", "Yes", "Yes", "Transparent, calculated and stable", "Ignore the correlation between assets"],
        ["Risk parity", "Yes", "Yes.", "Visible risk management contribution", "Relying on historical risk parameters and asset pools"],
    ], font_size=7.9)
    doc.add_page_break()

    add_section_heading(doc, "1.3 Inverse volatility rate as a first step towards the same risk", 2)
    add_text(doc, "The simplest risk-adjusted weight is the inverse method of volatility. If the annualized volatility of the asset i is considered, the weight is:")
    add_equation(doc, "wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)", "(3)")
    add_text(doc, "The intuitive intuitiveness of the volatility rate is to allow high-volatile assets to hold smaller financial weights, thus avoiding fluctuations in the dominant portfolios of equities, crude oil and bitcoin. If the correlation coefficients between all assets are the same, the last fluctuation rate is close to the equivalent; however, in a real multi-asset portfolio, the internal relevance of equities ETF tends to be significantly higher than that of equities to gold, bonds or foreign exchange. Looking at single-asset volatility alone ignores this resonance, and therefore the bottom-up is a useful benchmark, not the final answer.")
    add_section_heading(doc, "1.4 Risk parities shift management from financial weights to risk contributions", 2)
    add_text(doc, "The idea at the core of risk parities is that the fragmentation of the portfolio should not depend solely on how much money is invested in each asset, but rather on how much each asset contributes to total portfolio fluctuations. Group fluctuations:")
    add_equation(doc, "σₚ = √(wᵀΣw)", "(4)")
    add_text(doc, "MRRI:")
    add_equation(doc, "MRCᵢ = ∂σₚ/∂wᵢ = (Σw)ᵢ/σₚ", "(5)")
    add_text(doc, "The absolute risk contribution of the asset i is equal to the weight multiplied by the marginal risk contribution. According to the Euler breakdown, the sum of the RCi of all assets equals the portfolio fluctuations:")
    add_equation(doc, "RCᵢ = wᵢ(Σw)ᵢ/σₚ,   ΣᵢRCᵢ = σₚ", "(6)")
    add_text(doc, "Where the target risk budget is b, strict risk budgeting requires that the risk share for each asset be in line with b. When bi = 1/n, the risk contribution combination is rigorous:")
    add_equation(doc, "RCᵢ/σₚ = bᵢ,   Σᵢbᵢ = 1", "(7)")
    add_text(doc, "Risk parity does not equal the same financial weight for each asset. Low volatility and related assets usually require higher funding weights to contribute to the same risk; high volatility assets provide the same risk with lower financial weights. This prevents the apparent portfolio of power from becoming a “stock+bitcoin” portfolio at risk.")
    add_figure(doc, "cell 049 img 02 value trends and correlations of assets only show does not save pictures in  Notebook .png", "Figure 1: Matrix of all assets SGD-day gains-related coefficients")
    add_text(doc, "A matrix of relevant coefficients illustrates two facts. On the one hand, there is a clear positive correlation between SPY, QQQ and IWM, as well as multiple regional stocks ETF, and an increase in the number of ZXXQ4QXZ shares cannot be linearly increased and decentralized; on the other hand, there are significant differences between the related structures of bonds, gold, foreign exchange and bitcoin and traditional stocks. This is why a full matrix of differences is used rather than a single-asset volatility.")
    doc.add_page_break()

    add_section_heading(doc, "1.5 Single-tier risk parities may still imply category bias", 2)
    add_text(doc, "Strict ERC divides risks at a single asset level, but it does not guarantee a large asset risk balance when the asset pool is asymmetric in the number of classes. For example, ZXQ1QZ, which is the subject of this study, is significantly larger than bonds and commodities ETF. Even if each ETF makes the same risk contribution, the combined risk contribution of all equities ETF may be much higher than that of bonds or commodities. This is an implied risk budget that is generated by the volume of assets rather than by an economic perspective.")
    add_text(doc, "The hierarchical risk budget splits the budget into two tiers: first, the category budget for equities, bonds, commodities, foreign exchange and encrypted assets, and then the asset budget within each category. The code multiplied the two by the target risk share for each asset, followed by a joint risk budget resolution using the complete EWMA ACQ matrix. It is not aggregating weights after ERC in the category. The category target budget, which is determined by the rules, can reduce the bias resulting from the distribution of risks based solely on the number of assets, but remains subject to changes in Top10 ' s actual selection category.")
    add_table(doc, ["Level", "Targets for decision-making", "Treatment of the study", "Resolved deviations"], [
        ["Category Layer", "Equity, bonds, commodities, foreign exchange, encrypted assets", "Setting the baseline budget, adjusted by ranking and trend dynamics", "Avoiding risk in determining asset types"],
        ["Asset level", "Single ETF or target in Top10", "Joint risk resolution budget after budget within category multiplied", "Map the target risk share to fund rights Heavy"],
        ["Attack level", "Maximum 3 controlled assets with strong trends and relevance", "Change between 0% and 30% according to breadth and crisis state", "To compensate for the lack of upward elasticity of net risk parities"],
    ], font_size=8.1)
    add_section_heading(doc, "1.6 Advantages of static risk parities and boundaries", 2)
    add_text(doc, "There are three clear advantages to risk parities relative to the mean difference. First, it does not need to estimate the expected long-term benefits and reduces the most volatile inputs. Second, it transforms risk contributions into explicit constraints and avoids excessive dominance of highly volatile assets in the portfolio. Finally, risk budgeting has a clear economic meaning that can be used to express investor tolerance for different types of risk.")
    add_text(doc, "However, static risk parities are not a complete yield model. It defaults that all assets are worth long-term holdings, with only risk-adjusted weights. When bonds, equities or a commodity enters a long-term negative trend, it continues to allocate risk budgets; when a strong trend market emerges, it may lose its upward elasticity by lowering its volatile asset weights. In addition, when using a long window sample, the strategy reacts slowly to increasing volatility and relevance during a crisis period.")
    add_section_heading(doc, "1.7 Extension of the study from static risk parities to dynamic offensives", 2)
    add_text(doc, "In response to the above-mentioned boundary, the study does not abandon risk parities, but locates them as a combination of “risk distribution engines” and inserts a state of return and crisis management before and after the engines. The whole strategy can be understood as four cross-cutting issues:")
    add_bullets(doc, [
        "Motion Top10 answers “What assets should be held at the moment”. It does not directly forecast the next day ' s gains, but rather excludes relatively vulnerable assets and reduces the drag of static decentralization on long-term decline.",
        "EWMA co-ordinated the answer \"What is the current risk structure?\" A higher weighting of closer-to-receipt gains allows fluctuations to enter risk estimates faster.",
        "The full dynamic double budget answers “how risks are allocated between and within categories”. The category budget is no longer a fixed constant, but is based on ranking and trend status.",
        "0-30% of self-adaptation responses: \"When is it worth taking the risk of expansion?\" Increased offensives when strong trends increase in breadth, proactively contracting when combinations retreat deeper or VIX enter a state of crisis.",
    ])
    add_text(doc, "This evolutionary path preserves the interpretability of risk parities, while allowing asset selection, risk parameters and overall risk exposure to change according to market conditions. The second part would expand the formula, the rules of implementation and the empirical results in that order.")
    doc.add_page_break()

    # Part 2
    from report_revision_content import write_revised_body
    write_revised_body(doc)
    doc.save(OUTPUT)
    print(OUTPUT)
    return

    add_section_heading(doc, "Part Two: Layer by layer construction of the policy", 1)
    add_section_heading(doc, "2.1 Asset pool, data timeliness point constraints", 2)
    add_text(doc, "The research asset pool consists of 18 risky assets such as global equities, US-China Treasury debt, gold, crude oil, industrial metals, foreign exchange and bitcoin, plus SORA cash earnings. All prices are converted to SGD total return calibre. Signals and risk parameters use only data available on and before the date of decision-making, and target weights are effective for the next transaction, thus avoiding pre-vision deviations.")
    add_table(doc, ["Item", "Set"], [
        ["Area of study", "2015-09-07 to 2026-07-31"],
        ["In Sample", "2015-09-07 to 2024-12-31"],
        ["Forward Authentication", "2025-01-01 to 2026-07-31"],
        ["Lock observation", "2026-08-01 to 2026-10-01,44 transactions Day"],
        ["Relocation check.", "15 trading days"],
        ["Official return cost", "Unilateral 0.05 per cent"],
        ["Risk-Free Rate", "SORA Daily frequency, one day behind"],
    ], font_size=9.2, first_col_bold=True)
    add_section_heading(doc, "2.2 First tier: 12–1 monthly momentum Top10", 2)
    add_text(doc, "The strategy uses 12-1 months to skip the most recent month's kinetic signal to reduce short-term inverting and microstructure noise:")
    add_equation(doc, "MOMᵢ,t¹²⁻¹ = Pᵢ,t₋₂₁ / Pᵢ,t₋₂₅₂ − 1", "(8)")
    add_text(doc, "Each inspection day is arranged in MOM12-1 descending order to select Top10 to enter the optimizer. Top10 is not a direct view, but instead limits the risk budget to relatively strong assets. In order to prevent an excessive bias in the ranking of the original proceeds towards high-volatile assets, the second ranking of the offensive candidates uses the combined momentum adjusted for volatility.")
    add_section_heading(doc, "2.3 Second tier: EWMA risk estimate", 2)
    add_text(doc, "EWMA:")
    add_equation(doc, "Σₜ = λΣₜ₋₁ + (1−λ) rₜ₋₁rₜ₋₁ᵀ", "(9)")
    add_text(doc, "EWMA reacts more quickly to fluctuations during the crisis than the sample band; however, too low is expected to lead to frequent weight swings and therefore needs to be designed in conjunction with the frequency of inspections, partial adjustments and trade strips.")
    add_section_heading(doc, "2.4 Third tier: fully dynamic two-tiered risk budget", 2)
    add_text(doc, "The two-tiered structure allocates category risks between stocks, bonds, commodities, foreign exchange, encrypted assets, and then solves the target within the category. The base budget for the category is 40/35/15/15/5/5 per cent, with a ranking and trend multiplier:")
    add_equation(doc, "Bₖ,t ∝ Bₖ⁰ × RankMultₖ,t × TrendMultₖ,t", "(10)")
    add_text(doc, "Of which RankMult takes a ranking of 0.75+0.50x, TrendMult takes 1.5, 1.0 or 0.5 based on 6 months of earnings and MA200 status. This design allows risk budgets to vary according to market conditions, but does not directly use income projections to solve financial weights.")
    add_section_heading(doc, "2.5 Fourth floor: 0%-30% self-adaptation", 2)
    add_text(doc, "The core group is superimposed with a dynamic attack kit. The attack rate is based on the breadth of assets that simultaneously meet Top10, 6 months positive momentum, prices above MA200 and short-term fluctuations without significant expansion, taking 0%, 10%, 20% or 30%.")
    add_equation(doc, "wₜ = (1−aₜ) wₜᶜᵒʳᵉ + aₜ wₜᵃᵗᵗᵃᶜᵏ,   aₜ ∈ {0,0.1,0.2,0.3}", "(11)")
    add_text(doc, "An offensive combination is constructed from up to three assets with the highest score and uses a correlation ceiling of 0.75 to reduce homogenization exposure. If the combination retreat crossed -5/-8/-8/-12% or VIX crossed 30, the offensive silos were restricted or Z0; VIX was reduced to less than 25 to be released from crisis status.")
    doc.add_page_break()

    add_section_heading(doc, "2.6 Implementation and cost constraints", 2)
    add_text(doc, "The strategy is checked every 15 trading days. If Top10 changes in membership, complete replacement is carried out; if membership does not change, only the target difference of 50% is implemented and 2% is used to filter small adjustments without trading belts. The return of the signal and weight calculated using the proceeds of the previous day is measured to ensure that the proceeds of implementation are well received.")
    add_equation(doc, "rₜᴾ = wₜ₋₁ᵀ rₜ − Costₜ + Cashₜ", "(12)")
    add_text(doc, "Formally compare the use of a unilateral 0.05 per cent cost to ensure that all strategic calibres are consistent. An independent copy of the real cost is created at the same time, but it is not transposed to reduce the internality between economic assumptions and strategic choices.")
    add_figure(doc, "cell_035_img_11_CORE_COMPARISON_STRATEGIES____.png", "Figure 2: Changes in the weight of larger categories of assets protected by inspiration")
    add_text(doc, "The figure shows that group weights are not static 40/35/15/5/5, but rather change in trend rankings, fluctuations and the combined effect of offensive cylinders. This is the core difference between this strategy and the traditional static all-weather.", first_line=True)
    doc.add_page_break()

    add_section_heading(doc, "2.7 Optimized path from baseline to final strategy", 2)
    add_text(doc, "To avoid displaying only the best results, the study retained 12 formal strategies, including equal rights, full-asset classic ERC, Top10 inverse fluctuations, classic/ERC/EWMA/downline/GARCH risk parity, single-layer dynamic budget, static and full-dynamic double-layer budget, and two offensive combinations. These intermediate strategies are used to identify “what levels of proceeds come from” rather than simply cross-section selection of Sharp's top.")
    add_figure(doc, "cell_075_img_01_FINAL_EXPECTED_STRATEGY_COUNT___12.png", "Figure 3:12 Cumulative gains from formal strategies and market benchmarks")
    add_text(doc, "The cumulative proceeds of an attack are the highest, but not the end value alone. Reactions, hand changes, forward annual stability and realistic costs need to be examined together.")
    add_figure(doc, "cell_075_img_02_FINAL_EXPECTED_STRATEGY_COUNT___12.png", "Figure 4: 12 official policy retreat curves")
    doc.add_page_break()

    # Part 3
    add_section_heading(doc, "Part Three: Final strategy and real trading", 1)
    add_section_heading(doc, "- Why did you choose Ji-chun?", 2)
    add_text(doc, "The final choice is not because it ranks first on all indicators, but because it provides a higher yield target for product positioning, while keeping annualized fluctuations at about 11% and maximum retreat control at 15%. Robust retreats are lower and fit for a more conservative client; constant and increased offensive positions are fixed, with little flexibility in weak markets. Psychic attacks turn the offensive ratio into a state function, which is more in line with the positioning of \"growth-based absolute-return products\".")
    add_table(doc, ["Policy", "Target population", "Identity of proceeds", "Main costs"], [
        ["GMI-AHRB", "Conservative, retreating sensitive clients.", "Low volatility, strong pullback.", "It's less elastic."],
        ["GMI-FOA", "Long-term capital value-added clients", "Fixed 30%. Attack.", "Weak markets can't shrink."],
        ["GMI-ADG", "Medium-risk clients seeking elasticity", "0% to 30% dynamic attack, containment switch.", "It's complicated, it's more dependent on state recognition."],
    ], font_size=8.8)
    add_section_heading(doc, "3.2 Fixed 0.05 per cent cost to actual cost", 2)
    add_table(doc, ["caliber", "Cumulative proceeds", "Annualized proceeds", "Yearic fluctuations", "Sharp.", "Maximum Drawdown"], [
        ["Spiritual attack (fixed at 0.05%)", "441.52%", "16.19%", "10.77%", "1.32", "-14.98%"],
        ["Psychic attack (real differences in costs)", "364.12%", "14.61%", "10.79%", "1.20", "-15.44%"],
        ["18 Rights, etc. (fixed 0.05 per cent)", "211.15%", "10.61%", "9.26%", "0.99", "-17.87%"],
        ["VT global equities (SGD)", "219.50%", "10.87%", "17.46%", "0.60", "-31.09%"],
        ["ZXQ0XZSI Channel Times (SGD)", "190.93%", "9.95%", "12.85%", "0.70", "-33.77%"],
    ], font_size=8.4)
    add_text(doc, "Actual differential costs use sub-market commissions, branch exchange costs, sub-market price differentials/slip-point agents, square root impact models that lag behind ADV and volatility rates, and asset-grade annual withholding tax agents. It is a conservative scenario, and it does not mean that every history has a real deal.")
    add_figure(doc, "cell_041_img_01_REALISTIC_SCENARIO_NAMES____.png", "Figure 5: Cumulative gains under fixed-cost versus real-time costs")
    doc.add_page_break()

    add_section_heading(doc, "3.3 Inline, forward validation and lock observation of samples", 2)
    add_table(doc, ["Area", "Cumulative proceeds", "Annualized proceeds", "Yearic fluctuations", "Sharp.", "Maximum Drawdown"], [
        ["within samples 2015-09-07 to 2024-12-31", "295.04%", "15.33%", "10.64%", "1.27", "-14.98%"],
        ["Forward authentication 2025-01-01 to 2026-07-31", "37.08%", "21.39%", "11.51%", "1.61", "-8.02%"],
        ["Targeted observation 2026-08-01 to 2026-10-01", "1.72%", "10.23%", "6.62%", "1.32", "-1.91%"],
    ], font_size=9)
    add_figure(doc, "cell_075_img_03_FINAL_EXPECTED_STRATEGY_COUNT___12.png", "Figure 6: Cumulative proceeds forward to certification area")
    add_key_box(doc, "Lock set to interpret boundary", [
        "There are only 44 trading days in the lockout and Alpha statistics are not effective enough to support long-term conclusions.",
        "If the rules are changed after looking at the zone, the area is no longer a real sample and needs to be clearly disclosed in the report.",
        "Formally judge the year-by-year reliance on freezing rules, cost sensitivity and stress scenarios.",
    ])
    doc.add_page_break()
    add_section_heading(doc, "3.4 Current hold and risk exposure", 2)
    add_text(doc, "The end-of-cycle real weights were concentrated in equities and commodities, with no bonds, foreign exchange or bitcoin holding stock. This does not mean that the strategy necessarily favours equities over the long term, but rather the dynamics and dynamic budget results of the intersectional point.")
    add_table(doc, ["Subject", "Asset Name", "End of period weights"], [
        ["ES3.SI", "Channel Times Index", "19.43%"], ["USO", "Energy (oil)", "15.61%"],
        ["IWM", "Russell 2000", "14.26%"], ["ISF.L", "British time 100", "12.97%"],
        ["DBB", "Industrial metals", "12.88%"], ["510300.SS", "300 deep.", "5.89%"],
        ["1308.T", "TOPIX", "5.89%"], ["SPY", "Pistol 500.", "5.39%"],
        ["GLD", "Gold", "4.29%"], ["QQQ", "NASDAQ 100", "3.39%"],
    ], font_size=8.8)
    add_text(doc, "The cross-section category weights were approximately 67.22 per cent for equities and 32.78 per cent for commodities. As equity assets remain regionally relevant, the main short-term risks of the portfolio are global stock market synchronization, energy and industrial metal fallback, and ZXQ0QZ conversion magnifying or offsetting gains from overseas assets.")
    doc.add_page_break()

    # Part 4
    add_section_heading(doc, "Part IV. Pressure testing and robustness", 1)
    add_section_heading(doc, "4.1 Historical Pressure Window", 2)
    add_table(doc, ["Scenario", "The cumulative proceeds of a spirit attack", "Maximum Drawdown", "Cumulative proceeds of rights such as assets", "Waiting for maximum retreat"], [
        ["Impact of the 2020 epidemic", "+0.79%", "-5.17%", "-16.84%", "-17.87%"],
        ["2022 stock, double-kill.", "-0.43%", "-5.18%", "-12.78%", "-15.66%"],
    ], font_size=8.8)
    add_figure(doc, "cell_043_img_01_stress_windows____.png", "Figure 7: Cumulative benefits and historical stress phases of the core product strategy")
    add_text(doc, "During the 2020 epidemic, the withdrawal of the elitist offensive was clearly less than the equivalent of the equity benchmark; in the 2022 stock double-kill, trend screening and dynamic risk budgets reduced exposure to the combined decline in traditional bonds and equities. However, strategies may still suffer losses in the rapid reversal of trends and leapfrogging.")
    add_section_heading(doc, "4.2 Assumptions of combined shock", 2)
    add_table(doc, ["Scenario", "GMI-AHRB", "GMI-FOA", "GMI-ADG"], [
        ["Asset and exchange rate shocks in Asia", "-11.08%", "-11.75%", "-11.68%"],
        ["Global liquidity crisis", "-27.19%", "-27.23%", "-26.86%"],
        ["Global recession and risk aversion", "-22.85%", "-23.80%", "-23.24%"],
        ["Rapid post-crisis risk retreat", "+17.19%", "+17.23%", "+16.86%"],
        ["Inflationary stock debt double kill", "-8.22%", "-6.86%", "-6.22%"],
    ], font_size=8.7)
    add_text(doc, "Assuming that the shock directly affects the real hold at the end of 2026-07-31, it is a \" hold-up loss immediately\" rather than allowing the model to rebalance the path. In the global liquidity crisis, the three strategic losses were close to 27 per cent, suggesting that dynamic budgets could not eliminate the risk of flight and simultaneous deleveraging.")
    add_section_heading(doc, "4.3 Parameter Neighbourly stability", 2)
    add_figure(doc, "cell_083_img_01_parameter_tests____.png", "Figure 8: Top N, momentum window, EWMA λ sensitivity to frequency of inspection")
    add_text(doc, "Parameter tests target the base EWMA-ERC path rather than the final policy of repeated adjustments. Top N = 8–10, Dynamic Window 189–231 days, λ = 0.94–0.97 and 15 days of inspection are relatively acceptable; however Top N from 10 to 6 increases in returns will be maximized to -23.57 per cent, which is not consistent with product risk positioning.")
    doc.add_page_break()

    add_section_heading(doc, "4.4 Walker-Forward and Multiple Tests", 2)
    add_figure(doc, "cell_087_img_01_WALK_FORWARD_STRATEGIES____.png", "Figure 9: Freezing rules Walker-Forward ratio for each full year")
    add_table(doc, ["Policy", "Medium annual Sharp", "Min", "Positive annual rate of return", "Annual ratio of Sharpes above parity"], [
        ["GMI-AHRB", "1.48", "-0.66", "80%", "60%"],
        ["GMI-ADG", "1.23", "-0.97", "80%", "60%"],
        ["Global Momentum EWMA Risk Parity (GMI-ERP)", "1.18", "-0.82", "80%", "60%"],
        ["GMI-FOA", "1.05", "-1.07", "80%", "50%"],
        ["18 Rights, etc.", "1.23", "-1.57", "80%", "0%"],
    ], font_size=8.5)
    add_text(doc, "When the 12 formal strategies were compared, the probability of a deflad Sharpe attack was 44.56 per cent, which did not meet strong statistical expectations. The result is that although common Sharp is 1.32, historical rankings should not be interpreted as proven stability Alpha after considering non-normal gains and multiple strategic options.")
    add_section_heading(doc, "4.5 Cost, delay in implementation and tail risk", 2)
    add_table(doc, ["Test", "Cumulative proceeds", "Sharp.", "Maximum Drawdown", "Conclusions"], [
        ["0.5 x fixed cost", "452.05%", "1.34", "-14.94%", "Optimistic scenario"],
        ["1.0 x fixed cost", "441.52%", "1.32", "-14.98%", "Official caliber"],
        ["2.0 x fixed cost", "421.04%", "1.29", "-15.05%", "Performance remains valid"],
        ["Implementation delay of one day", "425.01%", "1.31", "-14.60%", "Cumulative gains decreased by 16.51 percentage points"],
    ], font_size=8.6)
    add_table(doc, ["End indicator", "95%", "99%"], [
        ["1 Day History", "0.97% / SGD 9,711.75", "1.91% / SGD 19,120.13"],
        ["1 Day History ES", "1.55% / SGD 15,459.00", "2.56% / SGD 25,622.29"],
        ["252-day crisis weights", "4.92% / SGD 49,157.23", "11.64% / SGD 116,370.02"],
        ["252 crisis weighted ES", "8.96% / SGD 89,599.96", "13.86% / SGD 138,576.55"],
    ], font_size=8.7)
    add_text(doc, "Bootstrap uses 1000, 252 day block sample paths. In the weighted scenario of crisis blocks, the probability of loss at the end of a psychic attack is 13.50 per cent, with a maximum retreat of 5 per cent at -14.19 per cent. The result proves that the strategy is not the product, and that there is still a need for sufficient risk tolerance for extreme pathways.")
    # Part 5
    add_section_heading(doc, "Part V Performance attribution", 1)
    add_section_heading(doc, "5.1 Contribution of asset and class proceeds", 2)
    add_figure(doc, "cell_095_img_01_linked_asset_contribution___geometrically_link_contributions_.png", "Figure 10: Invigorating against cumulative revenue contribution (by large asset class)")
    add_table(doc, ["Asset Class", "Cumulative proceeds contribution"], [
        ["Equities", "+162.58 percentage points"], ["Crypto", "+149.62 percentage points"],
        ["Commodities", "+129.50 percentage points"], ["FX", "+10.50 percentage points"],
        ["Bonds", "2.93 percentage points"], ["Trading Friction", "- 9.76 percentage points"],
    ], font_size=9.0)
    add_text(doc, "At the target level, bitcoin contributes +149.62 percentage points, gold +85.93, ES3.SI +52.53, industrial metals +28.82, QQQ +24.76. Thus, the high cumulative benefits of the strategy are not evenly derived from 18 assets, but are significantly dependent on a few long-cyclical strong trends, particularly bitcoin, gold and some stock markets. This conclusion also indicates that the BTC digestion is a necessary robust examination and cannot be shown only the best results with BTC.")
    add_section_heading(doc, "5.2 Strategy module incremental contribution", 2)
    add_figure(doc, "cell 097 img 01 ATTRIBUTION MOMENTUM EQUAL NAME is dedicated to  12-1 month kinetic Top10, etc.  .png", "Figure 11: The incremental contribution of the strategy modules")
    add_table(doc, ["Build Phase", "Cumulative proceeds at this stage", "Relatively forward step increment"], [
        ["GMA-EW", "211.15%", "+211.15%"],
        ["Dynamic Top10 Filter", "330.55%", "+119.40%"],
        ["EWMA Strict risk parities", "208.73%", "-121.82%"],
        ["A fully dynamic two-tier risk budget", "270.99%", "+62.27%"],
        ["Dynamic 0-30%", "441.52%", "+170.52%"],
    ], font_size=8.8)
    add_text(doc, "Subordination is sensitive to the order of accession. Therefore, the report uses the Shapley break-up to average all additions to the kinetic filter, EWMA-ERC and the fully dynamic double budget. The results show that the budget contribution for dynamic screening and dynamic risk is positive, and that strict EWMA-ERC is primarily responsible for reducing risk and inhibiting extreme exposure and is not a cumulative source of income.")
    doc.add_page_break()

    add_section_heading(doc, "5.3 Contribution to and residual of seed proceeds Alpha", 2)
    add_text(doc, "Simplified four-factor model uses global equities, medium-term national debt, commodities and United States dollars and receives annualized remnants of Alpha 11.47 per cent. The model, however, omits gold, bitcoin, regional stock markets, multi-asset momentum and dynamic budget increases, thus exposing observable strategies as Alpha.")
    add_figure(doc, "cell_105_img_01_def_expanded_factor_performance_attribution_result__factor_frame__rf__start__end.png", "Figure 12: Contribution of the cumulative benefits of the extended factor")
    add_table(doc, ["Attribution", "Beta/factor", "Annual computational contributions", "Cumulative contribution", "HAC t", "p Value"], [
        ["Global equities", "0.30", "+3.15%", "+91.59%", "9.22", "0.00"],
        ["Gold", "0.09", "+0.92%", "+30.47%", "8.27", "0.00"],
        ["Bitcoin.", "0.01", "+0.69%", "+13.14%", "3.32", "0.00"],
        ["Multi-asset momentum", "0.18", "+2.16%", "+57.53%", "13.29", "0.00"],
        ["Dynamic risk budget increment", "0.88", "+4.72%", "+98.32%", "29.09", "0.00"],
        ["Non-linear stock selection", "0.34", "+0.08%", "+0.67%", "0.62", "0.54"],
        ["Residue Alpha", "0.00", "+2.85%", "+82.88%", "2.45", "0.01"],
    ], font_size=8.2)
    add_text(doc, "R2 for the extended model is 0.87, adjusted R2 is 0.86, the standardized condition is 4.84 and the maximum factor is 0.76. The residual Alpha dropped from 11.47 per cent to 2.85 per cent as a result of the factor expansion, indicating that this strategy is more accurately positioned as a “multi-asset Beta + system momentum + dynamic risk budget + limited residual Alpha” rather than a pure Alpha strategy.")
    add_section_heading(doc, "5.4 Trade losses from gross to net proceeds", 2)
    add_figure(doc, "cell_101_img_01_realistic_attribution_result___realistic_submission_result.png", "Figure 13: Gross proceeds under realistic differential costs to net gains waterfalls Figure")
    add_table(doc, ["Item", "Cumulative proceeds contribution"], [
        ["Gross proceeds from assets", "+408.19 percentage points"],
        ["Commissions and transaction costs", "- 7.78 percentage points"],
        ["FX Conversion Costs", "-19.91 percentage points."],
        ["Bid-Ask Spread and Slippage", "-10.87 percentage points"],
        ["Market Impact", "-1.20 percentage points"],
        ["Obsessed taxes", "- 4.31 percentage points."],
        ["Final net gain", "+364.12 percentage points"],
    ], font_size=9)
    add_text(doc, "The cumulative benefits of real costs were reduced from 44.52 per cent to 364.12 per cent. The largest visible wear and tear are exchange and price differentials/slides, which are consistent with the multi-market ZXQ0QZ combination of cross-currency transactions. Market shocks are smaller because the current model assumes that the size of the funds is small compared to the target ADV; if the management scale is expanded, the square root impact item will not increase linearly.")
    doc.add_page_break()

    # Conclusions
    add_section_heading(doc, "Conclusions, product positioning and risk tips", 1)
    add_section_heading(doc, "Product positioning", 2)
    add_text(doc, "Zing Ji-jin Spirit is positioned as a global multi-asset strategy with SGD as the base currency, no nominal leverage, moderate volatility, and dynamically adjusted offensive ratios. It is not committed to absolute positive returns, nor is it aimed at the complete substitution of equities, but rather seeks to reduce extreme retreats through trends, risk parity and crisis management while preserving equity and commodity elasticity.")
    add_section_heading(doc, "Research findings", 2)
    add_bullets(doc, [
        "Relative equivalents and single stock market indices, with final strategies increasing returns and risk adjustment gains during the study period, have resulted in significant reliance on currencies, gold and long-trend stock markets.",
        "Dynamic screening and dynamic risk budgeting are the main structural sources of gains; strict risk parities in EWMA provide more risk constraints and do not directly generate benefits.",
        "The simplified factor model overestimates Alpha; after adding bitcoin, gold, area, multi-asset momentum and dynamic budgets, Alpha, the annualized residual, dropped to 2.85 per cent.",
        "Real-life differential costs have reduced Sharp from 1.32 to 1.20, proving that exchange, price differentials and taxes for multi-market transactions cannot be ignored.",
        "The degree of stability of the forward validation support strategy, however, Deflad Sharpe's probability of 44.56 per cent and short locks on a regular basis means that evidence has not been sufficient to rule out multiple tests and sample accidents.",
    ])
    add_section_heading(doc, "Main risks and constraints", 2)
    add_table(doc, ["Risk", "Impact", "Recommendations"], [
        ["Sample and listing history", "Assets such as BTC, DBMF have a relatively short history and different asset start dates are inconsistent", "Continue reporting from the same starting point and going BTC results"],
        ["Negotiability of proxy object", "510300.SS, 511010.SS may be restricted to individuals in Singapore, and BTC is not ETF until 2024", "Reinvestability test with UCITS/local equivalent"],
        ["Cost model", "Sub-market rates, slide points and taxes are proxy values, non-historical. Hand it over.", "Replace with actual voucher quotations, account tax and AUM scenarios"],
        ["Mobility and Jumping", "Models may not be available in extreme crisis.", "Use of more high-frequency risk gates and maximum asset weight limits"],
        ["Compromise", "Multiple strategies and parameters are compared, DSR evidence is weak", "Freezing the current rules and continuing with new data in the future"],
        ["Baseline selection", "A low Beta multi-asset strategy is not just a return to stock indices.", "Equal rights, 60/40, full assets, ERC and multifactor assessment"],
    ], font_size=8.1)
    add_section_heading(doc, "What else do you need from course research to voucher research?", 2)
    add_text(doc, "In terms of the current code workload, the outside sample, cost, stress testing, parameter stability and four layers of attribution already constitute a complete chain of evidence for the course. The gap with the same type of research on Chinese-Chinese securities is first and foremost the structure of writing and evidence organization, rather than the lack of a large number of strategic codes. In order to further increase the intensity of portfolio development for external issuers, it is suggested that only the following code modules that can change conclusions be added.")
    add_table(doc, ["Priority", "Suggested new code jobs", "Current Status", "Why does it work?"], [
        ["P1", "Increase the rolling minimum variance, the contraction average difference and the full-assets volatility bottom benchmark, using the same reconciliation points and costs", "Not fully integrated into the official results", "Moving from a theoretical to a empirical comparison of the same calibre"],
        ["P1", "Create a AUM capacity curve to recalculate the ADV participation rate, impact cost and net Sharp fund sizes of SGD 1m, 10m, 50m, 100m", "A square root impact model exists, size gradient is missing", "\"Recoverable\" to \"size of product to carry\""],
        ["P1", "Complement White Reality Check or Hansen SPA and calculate PBO using CSCV", "DSR and Walk-forward already exist, not covering all data mining tests", "A more complete answer to whether the final strategy is just a survivor after multiple searches."],
        ["P2", "Establish an investmentable alternative asset pool, replace the restricted bid with a Singapore tradable or UCITS equivalent, re-evaluated after tax", "Limits and parts of UCITS, not completely replaced", "Turning the pool from an academic agent to a truly investable portfolio for Singaporean clients"],
        ["P2", "Join macro-state groups to report returns, fallbacks and asset contributions by growth, inflation, interest rate and liquidity", "Has history and hypothetical pressure, missing complete status panel", "It's closer to the issuer's reading of what's working in what environment."],
        ["P3", "Establishment of a paper transaction log to keep track of signals, target weights, actual trade differences, costs and net daily values", "A long-term physical observation library has not been established", "Replace the short lockset with real time stream data and ultimately validate the transaction's realizability"],
    ], font_size=7.7)
    add_text(doc, "60/40, 18 Rights to Assets, Full of Movementless Assets ERC, Random Top10, Go to BTC, Parameter Sensitivity, Walker-Forward, Real Costs, Pressure Tests and Performance Attributions have been completed in existing codes and need not be rewritten; these elements need to be re-organised mainly in the body of the text by study.")
    add_section_heading(doc, "Next study", 2)
    add_text(doc, "The next step is not to recommend the continuation of fine-particle excavations on Top10, poaching or offensive scales, but rather a shift to more economically meaningful structural improvements: multi-cycle consensus momentum, signal separation within and between categories, double caps on asset weights and factor exposure, and freezing certification on real voucher costs and longer unseen samples.")
    doc.add_page_break()

    add_section_heading(doc, "Appendix Code architecture and recovery path", 1)
    add_table(doc, ["Documentation", "Functions"], [
        ["config.py", "Asset pool, time horizon, policy name, transaction cost and global parameters"],
        ["data_pipeline.py", "Prices, foreign exchange, VIX and SORA data access, SGD total return build"],
        ["models.py", "Sample differences, EWMA, semi-column differences, GARCH and ERC solver"],
        ["strategies.py", "Motion Top10, dynamic double budget, attack kit and retroactivity"],
        ["reporting.py", "Performance, realistic costs, holding, retreat and chart"],
        ["stress_tests.py", "History/Assumption Pressure, Cost, Delay, Bootstream, VaR/ES"],
        ["benchmarks.py", "Equal rights, 60/40, full assets ERC, random Top10 and market benchmark"],
        ["robustness.py", "Parameter sensitivity, BTC digest, Walker-Forward, multifactor Alpha and DSR"],
        ["attribution.py", "Asset/category, module, factor, currency and wear and tear"],
        ["main.py", "Main entrance, call complete research process in stages"],
    ], font_size=8.5)
    add_text(doc, "Put SORA daily.csv on the root of the project, then execute the Main. py-through attachment. HOLDOUT_START and HOLDOUT_END are adjusted in config.py to change the outer space of future samples, while ensuring that Yahoo/MAS data are updated to the target end. All outputs should be re-executed and no old maps should be used to replace the new compartmental results.")
    add_section_heading(doc, "References", 2)
    refs = [
        "Markowitz, H. (1952). Portfolio Selection. Journal of Finance.",
        "Qian, E. (2005). Risk Parity Portfolios: Efficient Portfolios Through True Diversification.",
        "Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.",
        "CSI (2023). Twentyth of the Thematic Series on Asset Allocation: A Global Multi-Asset Fund Portfolio Strategy based on the Trends Configuration Model.",
        "Yahoo Finance Historic Line Data; Medical Information of Singapore SORA Daily Frequency Data.",
    ]
    for ref in refs:
        add_text(doc, ref, size=9, first_line=False, after=2)

    # Update fields on open.
    settings = doc.settings._element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings.append(update_fields)

    doc.core_properties.title = "Global multi-asset dynamic defence strategy study"
    doc.core_properties.subject = "Dynamic Top10, fully dynamic double-tier risk budget, EWMA risk parity and self-adaptation offensive"
    doc.core_properties.author = "MFE5209 Financial Engineering Project Group"
    doc.core_properties.keywords = "Global multi-asset, risk parity, momentum, EWMA, stress testing, performance attribution"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
