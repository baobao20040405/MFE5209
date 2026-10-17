"""Editorial reflow of the evidence-backed report; trading code is untouched."""
from copy import deepcopy
from pathlib import Path
import re
import shutil
from docx import Document
from docx.oxml.ns import qn
from report_revision_content import (ROOT, NAMES, CORE, read_table, table, figure,
    performance, add_text, add_section_heading, note)

TARGET = ROOT / 'Study on Global Multi-Assessment Dynamic Strategies   Chinese.docx'


def text_of(el):
    return ''.join(el.xpath('.//w:t/text()')).lstrip('▌▍▎▏■▪█ │')


def heading(el):
    styles = el.xpath('./w:pPr/w:pStyle/@w:val')
    return bool(styles and styles[0] in ('Heading1', 'Heading2', 'Heading3', '1', '21', '31'))


def rename(el, text):
    ts = el.xpath('.//w:t')
    if ts:
        ts[0].text = text
        for t in ts[1:]:
            t.text = ''


def no_break(el):
    for b in el.xpath('./w:pPr/w:pageBreakBefore'):
        b.getparent().remove(b)


def main():
    backup = ROOT / 'report_work' / 'Study   Backup before main re-engineering. Docx'
    if not backup.exists():
        shutil.copy2(TARGET, backup)
    doc = Document(TARGET)
    body = doc._element.body
    elems = list(body)
    start = next(i for i, e in enumerate(elems) if heading(e) and text_of(e).startswith('Part II'))
    end = next(i for i, e in enumerate(elems) if heading(e) and text_of(e).startswith('Part IV'))
    # Preserve all tabular evidence and equations; identify existing logical blocks.
    blocks = {}
    current = None
    for e in elems[start:end]:
        s = text_of(e)
        if heading(e) and (re.match(r'^[23]\.\d', s) or s.startswith(('Part II', 'Part III'))):
            current = s.split(' ')[0]
            blocks[current] = []
        if current:
            blocks[current].append(e)

    new = []
    def emit(key, title=None, strip=False):
        group = blocks[key]
        if title:
            rename(group[0], title)
        if strip:
            group = group[1:]
        for e in group:
            no_break(e)
        new.extend(group)

    emit('Part II')
    emit('2.1', '2.1 Asset pool and research design')
    emit('2.2', strip=True)
    emit('2.2.1', strip=True)
    emit('2.3', '2.2 From static configuration to drive risk configuration')
    emit('2.4', strip=True)
    emit('2.8', '2.3 Dynamic budgets and evolution of offensive mechanisms')
    emit('2.9', strip=True)
    emit('2.11', strip=True)
    emit('2.13', '2.4 Interdivisional testing and final selection of 12 strategies')
    emit('2.14', strip=True)
    emit('2.15', strip=True)
    emit('2.16', strip=True)
    emit('Part III')
    emit('3.1', '3.1 Product performance at two cost calibres')
    emit('3.5', strip=True)
    emit('3.2', '3.2 7-step construction from asset signal to real deal')
    emit('3.3', strip=True)
    emit('3.4', strip=True)
    emit('2.12.1', '3.3 How the actual configuration translates into benefits')
    emit('2.12.2', strip=True)

    extra = doc
    original_children = set(body)
    add_section_heading(extra, '3.4 Proceeds path Reversal experience and trading activities', 2)
    add_text(extra, 'The configuration mechanism ultimately needs to be validated by the net path of the investor \' s actual experience. The following is a look at the final strategy and the accumulated gains from global equities and Singapore equities, and then test their risk costs with three core product homogenous indicators. The two comparisons carry different tasks: the market index describes opportunity costs, and the core product compares the choice of dynamic offensive rather than the low stock Beta itself as Alpha.')
    figure(extra, 73, 1, 'Psychic assault versus cumulative gains from global equities and Singapore equities')
    add_text(extra, 'The ultimate strategy is not to lead the stock market in every line. It participates in trends through the deployment of core risk budget controls and through dynamic offensives; gains can accrue when strong trends persist, and signals may lag when rapidly reversed. Therefore, cumulative gains, annual results and fallbacks should be read at the same time and not be based solely on the top point evaluation model at the end of the period.')
    for period in ['Complete research', 'In Sample', 'Validation interval']:
        performance(extra, f'Three core products{period}A unified unilateral cost of 0.05 per cent', ['core', 'fixed', 'final'], period)
    add_text(extra, 'In the complete zone, the increase in the relative continuity of the attack will increase the cumulative gain by 78.97 percentage points, with Sharp increasing from 1.15 to 1.32 percentage points, with fluctuations from 11.19 per cent to 10.77 per cent and a change of hand from 446.31 per cent to 34.17 per cent. However, relative intelligence is robust, with the maximum retreat from -9.75 per cent to -14.98 per cent. This group, by contrast, supports it as a primary product with greater value for income participation, and does not support the most conservative of the three.')
    add_text(extra, 'The interdivisional results also limited the conclusion that the final strategic returns were 295.05 per cent in the sample, 1.27 per cent in Sharp, 37.08 per cent in the certification area and 1.61 per cent in Sharp. Part II shows that the proceeds of the EWMA base during the certification period were 40.30 per cent higher than those of Sharp. The choice of dynamic precautions is based on a combination of inter-temporal and stress evidence, rather than on the assertion that any phase of them is better than a simplified model.')
    figure(extra, 73, 3, 'The final strategy, the historical retreat of the core against the market.')
    add_text(extra, 'Reversal maps serve to explain the holding of experience, not to duplicate the ranking of benefits. In times of rapid market downturns, trend screening, fragmentation of core risks and offensive contractions can cushion losses together, but do not eliminate retreats. The final strategy of the outbreak window retreats - 5.17 per cent, and the double-kill window of 2022 shares - 5.18 per cent; the largest withdrawal of the whole history remains - 14.98 per cent, indicating that risks at different stress stages cannot be represented by a single crisis window. Part IV retains the entire pressure sheet to test the scope of application of these advantages.')
    ledger = read_table(69)
    ledger = ledger[ledger.iloc[:, 0].isin(CORE)]
    table(extra, 'Actual gains and losses and trading activities under the initial $1 million SGD', ledger.iloc[:, [0,1,2,3,5,7,8]],
          ['Policy', 'Initial funding SGD', 'SGD net end of period', 'Cumulative surplus/deficit SGD', 'Shortest single-day surplus/loss', 'Number of redeployments', 'Number of traded legs'], 7.6)
    add_text(extra, 'The cumulative gains of 44.15 per cent corresponded to an increase of about SSD 4.4152 million in initial funding and about SSD 5.4152 million in end-of-period funding; it received 16.19 per cent per year for the same period. The more close the single-day surplus or deficit is to the size of the funds at that time, the greater the currency loss corresponding to the same percentage fluctuations. The number of legs to be traded records the buying and selling of assets and cannot be mixed with the number of times or the odds of success.')
    add_text(extra, 'The lower hand exchange is a low hand change in the sense of a fixed attack rather than a passive product. The annualized unilateral change of hands of 342.1 per cent still requires stable implementation, cross-market accounts and cost controls. While the above-mentioned half-speed adjustment and the fine-tuning of the non-tradable buffer reduction target have resulted in transactions, the existing complete programme has not yet been able to separate the independent contribution of each implementing rule.')
    figure(extra, 59, 2, 'The change in Sharp as transaction costs rise, the final strategy versus the other.')
    add_text(extra, 'The sensitivity of the fixed rate is different from a copy of the actual cost. The former uniformly raise each transaction rate and test hand-to-hand wear and tear; the latter add exchange, price differential, shock and tax burden to test more specific performance assumptions. After doubling costs, Sharp\'s final strategy remains at 1.29 and the actual cost is 1.20. Both types of results show that performance is not entirely dependent on a single optimistic scenario of 0.05 per cent, but that they are not a record of transactions validated by the securities firm.')
    for e in list(body):
        if e not in original_children and e.tag != qn('w:sectPr'):
            body.remove(e)
            new.append(e)
    emit('3.6', '3.5 Target client-product boundary')
    # Reinsert the new middle without touching the theory or stress/attribution evidence.
    for e in elems[start:end]:
        body.remove(e)
    for offset, e in enumerate(new):
        body.insert(start + offset, e)

    # Keep a small coherent heading hierarchy; join related sections naturally.
    keep = {
      '1.1':'1.1 Average variance configuration and estimated error',
      '1.3':'1.2 Downfall from volatility to strict risk parity',
      '1.5':'1.3 Boundary and dynamic orientation of static configuration',
      '4.1':'4.1 Historical pressure windows and defensive performance',
      '4.6':'4.2 Combined impact Simulation path and tail loss',
      '4.10':'4.3 Cost Implementation and Parameter robustness',
      '4.13':'4.4 Prior year testing and research selection deviations',
      '5.1':'5.1 Source of assets and contribution to strategy modules',
      '5.5':'5.2 External factors explained and dynamic risk exposure mouth',
      '5.8':'5.3 Gains on real costs',
      '5.9':'5.4 Research findings and subsequent validation',
    }
    for p in list(doc.paragraphs):
        s = p.text
        m = re.match(r'^([145]\.\d+)\s', s)
        if m and p.style.name.startswith('Heading'):
            if m[1] in keep:
                rename(p._p, keep[m[1]])
                no_break(p._p)
            else:
                p._p.getparent().remove(p._p)
        elif p.style.name == 'Heading 3':
            p.style = doc.styles['Normal']
            for run in p.runs:
                run.bold = True
            no_break(p._p)
        # Remove obsolete section cross-references after consolidation.
        for run in p.runs:
            if 'Listed in section 3.3' in run.text or 'Independently presented in Section 5.8.' in run.text:
                run.text = run.text.replace('Listed in section 3.3', 'Listed later in this section').replace('Independently presented in Section 5.8.', 'Independently presented in Section 5.3')
    # Do not retain explicit empty page breaks within a continuous argument.
    for p in list(doc.paragraphs):
        if not p.text.strip() and p._p.xpath('.//w:br[@w:type="page"]'):
            nxt = p._p.getnext()
            if nxt is not None and not text_of(nxt).startswith(('Part I','Part II','Part III','Part IV','Part V')):
                p._p.getparent().remove(p._p)
    doc.save(TARGET)
    print(TARGET)


if __name__ == '__main__':
    main()
