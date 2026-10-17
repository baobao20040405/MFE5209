"""Structured, editable Office Math equations, with genuine fractions and scripts.

These are native Word OMML objects, not MathType OLE objects or equation images.
"""
from copy import deepcopy

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def node(tag, *children, **attributes):
    result = OxmlElement('m:' + tag)
    for key, value in attributes.items():
        result.set(qn('m:' + key), str(value))
    for child in children:
        if isinstance(child, str):
            child = run(child)
        if tag in {'e', 'num', 'den', 'sub', 'sup', 'deg', 'oMath'} and child.tag == qn('m:e'):
            for grandchild in list(child):
                result.append(grandchild)
        else:
            result.append(child)
    return result


def run(text, upright=False):
    result = node('r')
    props = node('rPr', node('sty', val='p' if upright else 'i'))
    result.append(props)
    font_props = OxmlElement('w:rPr')
    fonts = OxmlElement('w:rFonts')
    fonts.set(qn('w:ascii'), 'Cambria Math')
    fonts.set(qn('w:hAnsi'), 'Cambria Math')
    font_props.append(fonts)
    size = OxmlElement('w:sz')
    size.set(qn('w:val'), '23')
    font_props.append(size)
    result.append(font_props)
    token = node('t')
    token.set(qn('xml:space'), 'preserve')
    token.text = text
    result.append(token)
    return result


def expr(*items):
    return node('e', *items)


def sub(base, index):
    return node('sSub', expr(base), node('sub', index))


def sup(base, power):
    return node('sSup', expr(base), node('sup', power))


def script(base, index, power):
    return node('sSubSup', expr(base), node('sub', index), node('sup', power))


def frac(top, bottom):
    return node('f', node('fPr', node('type', val='bar')), node('num', top), node('den', bottom))


def sqrt(body):
    return node('rad', node('radPr', node('degHide', val='1')), node('deg'), expr(body))


def group(*items, left='(', right=')'):
    return node('d', node('dPr', node('begChr', val=left), node('endChr', val=right)), expr(*items))


def series(symbol, lower, upper, *body):
    return node('nary', node('naryPr', node('chr', val=symbol), node('limLoc', val='undOvr'),
                             node('grow', val='1'), node('supHide', val='1' if upper == '' else '0')),
                node('sub', lower), node('sup', upper), expr(*body))


def summation(lower, *body, upper=''):
    return series('∑', lower, upper, *body)


def portfolio_variance():
    return expr(sup('w', 'T'), 'Σw')


def budget_equation():
    return frac(expr(sub('w', 'i'), sub(group('Σw'), 'i')), portfolio_variance())


def formulas():
    return {
        '(1)': [script(run('max', True), 'w', ''), '  ', sup('w', 'T'), 'μ − ', frac('γ', '2'),
                portfolio_variance(), '，  ', sup('1', 'T'), 'w = 1，  w ≥ 0'],
        '(2)': [sub(run('min', True), 'w'), '  ', portfolio_variance(), '，  ', sup('w', 'T'),
                'μ ≥ ', sub('μ', '0'), '，  ', sup('1', 'T'), 'w = 1，  w ≥ 0'],
        '(3)': [sub('w', 'i'), ' = ', frac(frac('1', sub('σ', 'i')),
                 summation('j=1', frac('1', sub('σ', 'j')), upper='n'))],
        '(4)': [sub('σ', 'p'), ' = ', sqrt(portfolio_variance())],
        '(5)': [sub(run('MRC', True), 'i'), ' = ', frac(expr('∂', sub('σ', 'p')), expr('∂', sub('w', 'i'))),
                ' = ', frac(sub(group('Σw'), 'i'), sub('σ', 'p'))],
        '(6)': [sub(run('RC', True), 'i'), ' = ', frac(expr(sub('w', 'i'), sub(group('Σw'), 'i')), sub('σ', 'p')),
                '，  ', summation('i=1', sub(run('RC', True), 'i'), upper='n'), ' = ', sub('σ', 'p')],
        '(7)': [frac(sub(run('RC', True), 'i'), sub('σ', 'p')), ' = ', sub('b', 'i'), '，  ',
                summation('i=1', sub('b', 'i'), upper='n'), ' = 1'],
        '(8)': [sub('M', 'i,t'), ' = ', series('∏', 's=t−251', 't−21', group('1 + ', sub('r', 'i,s'))), ' − 1'],
        '(9)': [sub('Σ', 't'), ' = λ', sub('Σ', 't−1'), ' + ', group('1−λ'), sub('r', 't'),
                script('r', 't', 'T'), '，  λ = 0.94'],
        '(10)': [sub('b', 'i'), ' = ', frac(expr(sub('q', 'i'), sub('u', 'i')),
                    summation('j', sub('q', 'j'), sub('u', 'j'))), '，  ', sub('q', 'i'),
                 ' = 0.75 + 0.50 ', run('RankPct', True), group(sub('M', 'i'))],
        '(11)': [sub('B', 'c'), ' = ', frac(expr(sub('Q', 'c'), sub('U', 'c')),
                    summation('k', sub('Q', 'k'), sub('U', 'k')))],
        '(12)': [sub('b', 'i'), ' = ', sub('B', 'c(i)'), sub('β', 'i|c'), '，  ',
                 summation('i', sub('b', 'i')), ' = 1，  ', budget_equation(), ' = ', sub('b', 'i')],
        '(13)': [sup('w', run('target', True)), ' = 0.70 ', sup('w', run('core', True)),
                 ' + 0.30 ', sup('w', run('attack', True))],
        '(14)': [script('w', 't', run('target', True)), ' = ', group('1−', sub('a', 't')),
                 script('w', 't', run('core', True)), ' + ', sub('a', 't'), script('w', 't', run('attack', True)),
                 '，  0 ≤ ', sub('a', 't'), ' ≤ 0.30'],
        '(14a)': [sub('S', 't'), ' = ', run('Top10', True), group(sub('M', 'i,t'))],
        '(14b)': [sub('B', 'c,t'), ' = ', frac(expr(sub('Q', 'c,t'), sub('U', 'c,t')),
                          summation('k', sub('Q', 'k,t'), sub('U', 'k,t')))],
        '(14c)': [sub('b', 'i,t'), ' = ', sub('B', 'c(i),t'), ' ', frac(expr(sub('q', 'i,t'), sub('u', 'i,t')),
                          summation('j∈c(i)', sub('q', 'j,t'), sub('u', 'j,t')))],
        '(14d)': [frac(expr(script('w', 'i,t', run('core', True)),
                           sub(group(sub('Σ', 't'), script('w', 't', run('core', True))), 'i')),
                     expr(sup(group(script('w', 't', run('core', True))), 'T'), sub('Σ', 't'),
                          script('w', 't', run('core', True)))), ' = ', sub('b', 'i,t')],
        '(14e)': [sub('H', 't'), ' = ', frac(summation('i∈Sₜ', sub('I', 'i,t')), '10'),
                  '，  ', sub('a', 't'), ' ∈ {0，0.10，0.20，0.30}'],
        '(14f)': [script('w', 't', run('raw', True)), ' = ', group('1−', sub('a', 't')),
                  script('w', 't', run('core', True)), ' + ', sub('a', 't'), script('w', 't', run('attack', True))],
        '(14g)': [sub('c', 't+1'), ' = 0.0005 ', summation('i', '|', sub('Δw', 'i,t+1'), '|'),
                  '，  ', script('r', 't+1', run('net', True)), ' = ', group('1+', sub('g', 't+1')),
                  group('1−', sub('c', 't+1')), '−1'],
        '(15)': [sub('S', 'i'), ' = ', frac(expr('0.50 ', sub('M', 'i,6M'), ' + 0.50 ', sub('M', 'i,12−1')),
                                             sub('σ', 'i,63D'))],
        '(16)': [script('w', 't', run('partial', True)), ' = ', script('w', 't', run('curr', True)),
                 ' + 0.50 ', group(script('w', 't', run('target', True)), ' − ', script('w', 't', run('curr', True)))],
        '(17)': [sub('g', 't'), ' = ', summation('i', script('w', 'i,t', run('pre', True)), sub('r', 'i,t')),
                 ' + ', script('w', 't', run('cash', True)), script('r', 't', 'f')],
        '(18)': [sub('c', 't'), ' = 0.0005 ', summation('i', '|', sub('Δw', 'i,t'), '|'),
                 '，  ', script('r', 't', run('net', True)), ' = ', group('1+', sub('g', 't')),
                 group('1−', sub('c', 't')), '−1'],
        '(19)': [sub(run('Impact', True), 'i,t'), ' = ', run('min', True),
                 group('0.01， 0.10 ', sub('σ', 'i,t−1'),
                       sqrt(frac(sub('Q', 'i,t'), sub(run('ADV', True), 'i,t−1'))), left='[', right=']')],
        '(20)': [sub('c', 'i,t'), ' = ', script('w', 'i,t', run('pre', True)), sub('r', 'i,t'),
                 '，  ', summation('i', sub('c', 'i,t')), ' + ', script('c', 't', run('cash', True)),
                 ' + ', script('c', 't', run('cost', True)), ' = ', script('r', 't', run('net', True))],
        '(21)': [sub('r', 'p,t'), ' − ', script('r', 't', 'f'), ' = α + ',
                 summation('k=1', sub('β', 'k'), sub('f', 'k,t'), upper='K'), ' + ', sub('ε', 't')],
    }


def add_professional_equation(doc, number):
    registry = formulas()
    if number not in registry:
        raise KeyError(f'No structured equation for {number}')
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.keep_together = True
    math_paragraph = node('oMathPara', node('oMathParaPr', node('jc', val='center')),
                          node('oMath', *[deepcopy(item) if not isinstance(item, str) else item
                                         for item in registry[number]]))
    paragraph._p.append(math_paragraph)
    label = paragraph.add_run('    ' + number)
    label.font.name = 'Times New Roman'
    label.font.size = Pt(10)
    return paragraph
