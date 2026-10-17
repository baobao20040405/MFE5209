"""Source-backed research narrative; does not import or run trading strategies."""
from __future__ import annotations

from pathlib import Path
import ast
import json
import re

import pandas as pd
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

from build_research_report import (
    ROOT, ASSETS, add_text, add_bullets, add_section_heading,
    add_table, add_equation, add_figure, add_caption, set_run_font,
)

TABLES = ROOT / "report_work" / "tables"
NAMES = {
    "equal": "Global Multi-Asset Equal Weight (GMA-EW)", "monthly": "Global Multi-Asset Classic Risk Parity (GMA-CRP)",
    "inv": "Global Momentum Inverse Volatility (GMI-IV)", "classic": "Global Momentum Classic Risk Parity (GMI-CRP)",
    "ewma": "Global Momentum EWMA Risk Parity (GMI-ERP)", "down": "Global Momentum Downside Risk Parity (GMI-DRP)",
    "garch": "Global Momentum GARCH Risk Parity (GMI-GRP)", "single": "Global Momentum Dynamic Risk Budget (GMI-DRB)",
    "static": "Global Momentum Static Hierarchical Risk Budget (GMI-SHRB)", "core": "Global Momentum Adaptive Hierarchical Risk Budget (GMI-AHRB)",
    "fixed": "Global Momentum Fixed Offensive Allocation (GMI-FOA)", "final": "Global Momentum Adaptive Defense and Growth (GMI-ADG)",
}
SHORT = {v: v.split("·")[-1] for v in NAMES.values()}
SHORT.update({
    NAMES['equal']: '18 Rights, etc.',
    NAMES['monthly']: 'Full Assets ERC',
    NAMES['classic']: 'Top10 classic ERC',
    NAMES['ewma']: 'EWMA agile balance',
    NAMES['core']: 'GMI-AHRB',
})
ORDER = ["equal", "monthly", "inv", "classic", "ewma", "down", "garch",
         "single", "static", "core", "fixed", "final"]
CORE = [NAMES[k] for k in ("core", "fixed", "final")]
METRICS = ["Cumulative Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Maximum Drawdown", "Annual One-Way Turnover"]
WEIGHT_KEYS = {1:'inv', 2:'classic', 3:'ewma', 4:'down', 5:'equal', 6:'single',
               7:'static', 8:'garch', 9:'core', 10:'fixed', 11:'final', 12:'monthly'}
WEIGHT_READINGS = {
    1: 'Configuration observation: The red stock area in the figure expanded significantly from 2017 to 2018 and 2021, giving way to blue bonds and green foreign exchange from late 2018 to 2019. The long-term small size of the purple encrypted assets suggests that the 63-day counter-variant will depress the share of funds at high volatility; many ETFs of the same shares may still be co-located to form larger red areas.',
    2: 'Configure Watch: The classic ERC is in the same direction as the reverse fluctuations, increasing bonds and foreign exchange equally around 2019, but not across the categories. The relative negative fluctuations in the size of the red equity stock from 2017 to 2018 have shrunk; the complete matrix does not allow highly relevant assets to obtain a higher weight at the same time as their respective low volatility.',
    3: 'Configuration observation: EWMA retained the common wheel of stock expansion in 2017, bond and foreign exchange increases in 2019, stock rebounds in 2021, but the red and blue borders from 2018 to 2020 were even more extreme, reflecting the combined effect of recent risk input with the regulation of warehousing. Rather than always increasing the weight of equities, it redistributed the same Top10 funds with recent fluctuations and relevance.',
    4: 'Configuration observation: the ratio of capital to commodities and encrypted assets has not experienced a long-term structural leap in terms of stock, bonds and foreign exchange transitions in the lower-line model, similar to the classic ERC. This is consistent with the use of the same Top10 portal; modifying the impact of the risk matrix does not mean that the “downline” model holds fewer shares at each point of decline.',
    5: 'Allocation observation: The initial target for equal rights is determined by the number of assets, with nine equities accounting for approximately 50.00 per cent, two bonds for approximately 11.11 per cent, goods for 16.67 per cent each and BTC for 5.56 per cent. The subsequent area in the figure changes with the proceeds and the actual rebalancing, and the initial equivalent cannot be misread as a category.',
    6: 'Configure Watch: Single-layer dynamics are roughly the same as the EWMA-based class transition time period, but the purple regions around 2017 and 2024-2025 are wider, with several commodity configurations changing with trend multipliers. This is because each asset is no longer given the same share of risk; having several selected assets in the class still affects its total risk budget.',
    7: 'Configure Watch: In the static double layer, the blue bond area in 2015-2016, 2019 and around 2020 is more dynamic than the single layer, and the purple encrypted asset area as a whole is thinner. This tendency arises from a fixed 35 per cent risk budget for bond classes and a 5 per cent budget for encryption, but the weight of funds is also affected by fluctuations and cannot be read directly at 35 per cent and 5 per cent.',
    8: 'Configuration observation: GARCH equities, bonds and foreign exchange rotations are largely consistent with the other Top10 models, with significant bond and foreign exchange configurations retained in 2019 and 2022-2023, with a higher share of equities in 2021 and 2025. It changes the projected fluctuations without changing the order of magnitude, so the graphics do not have an asset rotation logic independent of Top10.',
    9: 'Configure observation: Full dynamic double layer relative static double layer reduces the binding of fixed class ratios. In 2017-2018 and 2024-2025, commodities and encrypted assets were better combined, while bonds and foreign exchange continued to take on significant funding allocations in 2022-2023. Red equities are no longer determined solely by the number of shares selected, but the rankings and trend breadth of the categories themselves are also part of the budget.',
    10: 'Configure Observation: The fixed offensive map retains the core of large-scale rotations and has a grey cash phase from its own core exit and defence rules. The grey peaks around 2025, which are more than 100 per cent, are part of the cash mix of the source maps, and cannot be used to calculate the cash ratio or leverage; the official performance in this table does not come from the size of the map.',
    11: 'Configure Observation: Dynamic offensives are close to a robust core \' s rotational direction, but the areas of purple encrypted assets in 2016-2018 and 2024-2025 are more visible, as are parts of the orange configuration. It did not clean up the attack as a whole and the funds returned to the core. The graphics indicate which categories of participation have been increased and the specific sources of proceeds are further checked by the category contribution tables.',
    12: 'Configuration observation: The total asset ERC class boundaries are clearly flat, blue bonds and green foreign exchange have long occupied large capital areas and purple BTC remains thin. Compared to equal rights, it bears the balance of risk with more low-volatile categories of funds, while all categories remain present and lack the entry points to remove vulnerable assets from the ranking, as did Top10.',
}


def read_table(cell, number=1):
    file = next(TABLES.glob(f"cell_{cell:03d}_table_{number:02d}_*.csv"))
    sample = pd.read_csv(file, dtype=str).fillna("")
    # Notebook MultiIndex exports contain a second header record.
    if any(str(v).startswith("Unnamed:") for v in sample.iloc[0]):
        frame = pd.read_csv(file, header=[0, 1], dtype=str).fillna("")
        frame.columns = [b if not b.startswith("Unnamed:") else a for a, b in frame.columns]
        return frame
    return sample


def value(v):
    text = str(v)
    if text in ("nan", "NaN", "", "—"):
        return "—"
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        return f"{float(text):.2f}"
    return text


def pretty(frame):
    return [[SHORT.get(str(v), value(v)) for v in row] for row in frame.itertuples(index=False, name=None)]


def note(doc, text):
    add_text(doc, text, size=8.3, color="666666", first_line=False)


def table(doc, title, frame, headers=None, size=8.5):
    add_caption(doc, title)
    labels = headers or list(frame.columns)
    rows = pretty(frame)
    for row in rows:
        for column, label in enumerate(labels):
            if 'Percentage Point' in label or ('Unit as percentage points' in title and column > 0):
                row[column] = row[column].removesuffix('%')
    add_table(doc, labels, rows, font_size=size)


def page(doc, title=None):
    if title:
        paragraph = add_section_heading(doc, title, 2)
        paragraph.paragraph_format.page_break_before = True
    else:
        doc.add_page_break()


def figure(doc, cell, number, caption, width=6.65):
    path = next(ASSETS.glob(f"cell_{cell:03d}_img_{number:02d}_*.png"))
    add_figure(doc, path.name, caption, width=width)


def performance(doc, title, keys, period="Complete research"):
    source = read_table(75)
    subset = source[source["Period"].str.startswith(period)].set_index("Policy")
    chosen = subset.loc[[NAMES[k] for k in keys], METRICS].reset_index()
    table(doc, title, chosen,
          ["Policy", "Cumulative proceeds", "Annualized proceeds", "Yearic fluctuations", "Sharp.", "Maximum Drawdown", "Change hands."], 8.2)
    return chosen


def weight_figures(doc, entries, commentary):
    for index, label in entries:
        figure(doc, 35, index, f"Configure Contrast{label}Capital weights for large asset classes at end of month", width=5.6)
        add_text(doc, WEIGHT_READINGS[index], size=9.5, after=4)
    performance(doc, 'Proceeding risk results corresponding to the top weight map', [WEIGHT_KEYS[index] for index, _ in entries])
    add_text(doc, commentary)


def introduce_asset_pool(doc):
    tree = ast.parse((ROOT / 'data_pipeline.py').read_text(encoding='utf-8'))
    rows = next(ast.literal_eval(statement.value) for statement in tree.body
                if isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'asset_rows'
                        for target in statement.targets))
    assert len(rows) == 18
    add_text(doc, 'The asset pool determines what risks the strategy can assume and the starting point for comparison of rights with risk budgets. The current official pool contains 18 items: 9 stocks, 2 national bonds, 3 commodities, 3 foreign exchange instruments and 1 bitcoin. Except for BTC-USD, which is a bitcoin price series, the rest is an ETF or a fund trading tool; they provide an index or associated asset exposure and do not all amount to direct holding of spot.')
    frame = pd.DataFrame([[row[0], row[3], row[2], row[1], row[5]] for row in rows],
                         columns=['Asset Class', 'Code', 'Track Object', 'Listed or marketed', 'Trading currency'])
    table(doc, '18 official asset pools itemized by current code', frame, size=8.5)
    add_text(doc, 'Equities cover the United States, Germany, the United Kingdom, Japan, Mainland China, Hong Kong, China, and Singapore, with SPY, QQQ, and IWM providing different exposures to large United States dishes, technological growth and compacts. Geographical dispersion does not represent independence: there are still significant common market risks between the three United States and other equities, and therefore a complete matrix is used instead of treating each ETF as an independent source of risk.')
    note(doc, 'The currency column is the quoted currency of the instrument and not its full economic risk. FXY and FXE deal in USD, but each provides a Japanese yen-related exposure to the euro; all return returns are eventually converted to SSD. The German stock EXS1.DE is still in the pool and was deleted from the German-British Treasury debt.')
    page(doc, '2.2 Configure role of asset pools and unified research calibre')
    add_table(doc, ['Asset Class', 'Role in the group', 'Risk awareness to be retained'], [
        ['Equities 9', 'Participation in global growth, profitability and risk preferences', 'Stock market relevance may rise simultaneously, with multiple areas not equal to independent hedges'],
        ['National debt 2', 'Provide United States-China interest rate risk and participate in part of the avoidance phase', 'Interest rate hikes and inflation shocks can also fall and cannot be considered risk-free cash'],
        ['Commodities 3', 'Gold, crude oil, industrial metals cover different objects and supply and demand-driven', 'GLD is not the same as USO, DBB; futures category funds are the difference between extension and tracking'],
        ['Foreign exchange 3 items', 'Provision of United States dollars, Japanese yen and euro-related investment tools', 'Currency of exchange quotations, instrument returns and SDD conversion risk cannot be confused'],
        ['1 bitcoin', 'Participation in independent high volatility trends and risk preference cycles', 'High volatility and high risk of leapfrogging, which could not be considered a spot ETF performance until 2024'],
    ], widths=[2.5, 6.5, 8], font_size=8.7)
    add_text(doc, 'The study did not include BIL, A35.SI or DBMF in the current 18 official pools and could not be written as held assets when explaining the combination of defence sources. SORA is a daily risk-free rate for external imports for real cash interest and excess earnings evaluation, which is not part of block 19 ranking securities and does not participate in Top10.')
    add_text(doc, 'The number of assets is uneven: with 18 simple rights, equities receive about half of the initial funding on a natural basis; and with a single-asset-level spread of risk, equities also have more aggregate target risk because of the higher target. The category budget is designed precisely for this structure, but it does not eliminate the rise in the relevance of common factors or cross-categories within the same category in a crisis.')


def write_revised_body(doc):
    part_two(doc)
    part_three(doc)
    part_four(doc)
    part_five(doc)


def part_two(doc):
    add_section_heading(doc, "Part II. Policy construction from configuration benchmarks to dynamic defense", 1)
    add_section_heading(doc, "2.1 Marked asset pool and configuration boundary", 2)
    introduce_asset_pool(doc)
    page(doc, '2.2.1 Sample area and 12 formal strategic routes')
    add_text(doc, "We have divided strategic research into four phases of configuration starting point, comparison of risk models, adaptation of budget mechanisms and modification of offensive modules. The 12 formal strategies include both parallel comparisons and incremental modifications: downside risk and GARCH to test alternative risk models; single, static and fully dynamic double layers to compare risk budget structures; and fixed and dynamic offensive to compare top-line engagement. They do not constitute 12 linear upgrades, one by one.")
    table(doc, "Standardized data and evaluation calibration", pd.DataFrame([
        ["Asset pool", "18 assets converted to SSD without retaining German and British Treasury debt"],
        ["In Sample", "2015-09-07 to 2024-12-31"],
        ["Validation interval", "2025-01-01 to 2026-07-31"],
        ["Study the complete range.", "2015-09-07 to 2026-07-31, excluding locked observations"],
        ["Lock observation", "2026-08-01 to 2026-10-01, for 44 common transactions Day"],
        ["Main measurement cost", "Unilaterally 0.05 per cent based on the value of the sale, with all official strategies of the same calibre"],
        ["Cash versus excess income", "Cash is at SORA interest rate and Sharp uses the corresponding daily no-risk gain"],
        ["Implementation", "The signal day closes into target and the next trading day closes. Hand it over."],
    ], columns=["Item", "Set"]), size=9)
    add_text(doc, "The formal evaluation takes into account both earnings, Sharp, retreats, fluctuations and hand changes, as well as validation zones, stress tests and realistic cost reviews. Validation zones have been used for comparison purposes and cannot continue to be described as independent samples that are completely unexposed; short-locked areas provide only observation. The mean difference is the theoretical starting point and has not yet resulted in a rolling empirical result of the same calibre as the 12 strategies.")
    add_section_heading(doc, "Four-phase study route", 3)
    add_table(doc, ["Phase", "Formal policy", "Research issues"], [
        ["Configure Start", "Equal rights benchmark, all assets classic balance", "Difference between fragmentation of funds and static risk"],
        ["Risk model", "Counter-volatility, classic, EWMA, row down, GARCH", "The same Top10 signal. How the risk model changes the outcome."],
        ["Risk Budget", "Single layer dynamic, static double layer, fully dynamic double layer", "How the budget balances the fragmentation of categories and trends"],
        ["Attack module", "Fixed 30% offensive, dynamic 0% to 30% offensive", "How to increase the cost of top-level participation and control continued attacks"],
    ], widths=[2, 7, 8], font_size=8.8)
    note(doc, "The main line is equal power to the full asset ERC → Top10 risk allocation dynamic double-tier budget fixed offensive versus self-adapted offensive. The advantages and disadvantages of the strategy are determined by the evidence at each stage, and not all new modules will improve each indicator.")
    note(doc, 'Each set of weights below is followed by the inter-temporal performance table of the corresponding strategy. Category colours and fonts are coded; area is a financial weight, not a risk contribution. The precise breakdown of the proceeds of assets is also shown in the category contribution statement, which cannot be calculated on the basis of area alone.')

    page(doc, "2.3 Allocation starting point Static risk spreads to improve recovery at the expense of gains")
    performance(doc, "Full inter-temporal performance of the configuration starting point", ["equal", "monthly"])
    add_text(doc, "The cumulative return on 18 rights to equity was 211.15 per cent, Sharp 0.99, maximum retreat - 17.87 per cent; the maximum reversal of the full-asset monthly classic risk parity fell to -8.30 per cent, with fluctuations from 9.26 per cent to 4.97 per cent, but only 79.75 per cent and Sharp 0.81. Risk parity achieved lower volatility and did not automatically yield higher-risk adjustment gains.")
    add_text(doc, "The weaknesses of the static programme are thus clear: first, weak trend assets still occupy funds and risk budgets; and second, low-volatile assets have easy access to higher funding weights, with insufficient participation in the upswing phase. This led to the introduction of a relative drive filter, while retaining the entire asset, ERC, as a low-risk reference. The difference between the two is not attributable entirely to the risk parity formula.")
    # Formal strategy registration order in the notebook determines these image IDs.
    page(doc, '2.3.1 Allocation of evidence Equivalence equal to full asset risk')
    weight_figures(doc, [(5, "18 Rights benchmarks"), (12, "Classic asset monthly risk parities")],
        "The linkage between configuration and results: the more stable bond, foreign exchange configuration of the full asset, corresponding to the lower volatility of 4.97 per cent of the table and to the −8.30 per cent reversal; and the right-to-share that allows equity and BTC assets to play a greater role through the weight of funds and the drift of proceeds, corresponding to a higher cumulative gain of 211.15 per cent. This is a comparison of the overall performance of the portfolio, and it is not possible to attribute a total of 131.40 percentage points of the margin of return to a particular category.")

    page(doc, "2.4 Parallel comparison of kinetic screening with five risk models")
    add_text(doc, "The first step is to determine which assets are held. Using the 252-day historical window, the last 21 trading days were eliminated to calculate the 12-1 month momentum and to select the top 10 points. Top10 is in relative order and does not guarantee a positive momentum for all selected assets; absolute trend conditions play a role mainly in the subsequent budget and offensive rules.")
    add_equation(doc, "Mᵢ(t) = ∏[s=t−251,…,t−21](1+rᵢ,s) − 1", "(8)")
    add_text(doc, "Under the same screening conditions, the reverse fluctuations are configured according to the most recent 63-day standard deviation; the classic ERC uses sample bands; the EWMA increases the weight of recent information; the next-line model focuses on the second-tier rectangular of negative gains; and the GARCH projectes the fluctuations of individual assets and integrates historical and related matrices. The latter four were all conditioned by risk contributions, with the distinction derived mainly from risk matrix estimates.")
    add_equation(doc, "Σₜ = λΣₜ₋₁ + (1−λ)rₜrₜᵀ,   λ = 0.94", "(9)")
    note(doc, "Format (9) describes the EWMA progressive idea; the actual calculation of initialization, annualization and numerical amendment is based on the risk model function. The bottom-line model is a second-order agent in the bottom row and cannot be directly equated with the distribution of loss or the constraint of the VaR.")
    performance(doc, "Comparison of risk models under the same Top10 conditions", ["inv", "classic", "ewma", "down", "garch"])
    add_text(doc, "The cumulative earnings of EWMA were 208.73 per cent and Sharp 1.23, higher than those of classic ERC 178.42 per cent, 1.05 and 182.84 per cent and 1.10. The next-line model does not reflect higher returns, with a maximum retreat −10.52 per cent greater than EWMA −9.01 per cent. As a result, the more complex risk model did not lead to better integrated results. EWMA is the basis for the subsequent budgetary reform.")
    add_text(doc, "There is still a price to be paid for this option: EWMA has an annual hand change of 341.40 per cent, which is higher than 254.66 per cent of the classic ERC, and has a total cumulative yield of slightly less than the equivalent of 211.15 per cent. EWMA-ERC improved Sharp and retreats, but did not adequately address the lack of top-level participation, and the subsequent optimization shifted to budget and offensive mechanisms.")
    page(doc, "2.5 Comparison of configuration results of risk models")
    weight_figures(doc, [(1, "Top10 Volatility Last"), (3, "Top10 EWMA risk parities")],
        "The linkage between configuration and results: the two have a common stock and defence category rotation, but the EWMA boundary changes more frequently. The above table shows that the cumulative gains were 33.66 percentage points higher than the negative fluctuations, which decreased from 8.01 per cent to 7.30 per cent and Sharpe from 1.00 to 1.23 per cent, at the cost of an increase of 106.68 percentage points. As a result, support has been expressed for improving overall efficiency following the distribution of relevant participation, and it is not possible to claim that a more distorted weight line alone generates gains at every turn.")
    page(doc, "2.6 Why the alternative risk model is not the ultimate main line")
    weight_figures(doc, [(2, "Top10 classic risk parities"), (4, "Top10 down, risk par.")],
        "Configuration-relationship to results: The bottom line model does not change the category configuration to a more conservative and continuous set. Accordingly, the cumulative proceeds of 174.95 per cent from the above table are lower than the classic ERC 178.42 per cent, and the maximum retreat-10.52 per cent is not better than the equivalent of −10.35 per cent, but the change of hands is lower. The change of the risk matrix only to the second-tier rectangular of the next row has not yet produced the consolidated improvements required for this product.")
    page(doc, "2.7 GARCH and EWMA risk model configuration and performance comparison")
    weight_figures(doc, [(8, "Top10 Garch risk parities"), (3, "Top10 EWMA risk parities")],
        "The linkage between configuration and outcome: both retain a clear stock-bond-exchange rotation, distinguishing the distribution range at each stage. EWMA ' s cumulative gains of 208.73 per cent, Sharp 1.23 outperformed GARCH 182.84 per cent, and 1.10 were slightly smaller, while GARCH ' s turnover of 301.3 per cent was lower than EWMA 341.40 per cent. On this basis, we choose EWMA to continue the study, based on the overall balance in the table, not the complexity of the model or the smoothness of the graphics.")

    page(doc, "2.8 From the range of risks to the single layer of dynamics and static double budget")
    add_text(doc, "The share of the risk is equal for each selected asset and does not represent a strong and weak trend. The single-tier dynamic budget ranking and the six-month trend, MA200, tilted asset risk budgets. The ranking multiplier is 0.75 plus 0.50 per cent; when trend conditions are met simultaneously, only one or none, the multipliers are 1.50, 1.00 and 0.50 respectively. The consolidated budget was given to the EWMA risk model for resolution.")
    add_equation(doc, "bᵢ = qᵢuᵢ / Σⱼqⱼuⱼ,   qᵢ = 0.75+0.50·RankPct(Mᵢ)", "(10)")
    add_text(doc, "Single-tier budgets may still be affected by the number of asset classes selected. The static two-tiered programme initially included 40 per cent of equities, 35 per cent of bonds, 15 per cent of commodities, 5 per cent of foreign exchange and 5 per cent of encrypted assets as risk classes; only the actual Top10 appearances were re-unified and distributed by ranking and trend. Forty per cent or 35 per cent of this is the target risk share, not the funding position.")
    performance(doc, "Complete inter-district results of risk budget adaptations", ["ewma", "single", "static", "core"])
    add_text(doc, "The single layer of dynamics increased the cumulative gains from 208.73 per cent to 223.95 per cent compared to EWMA-ERC and from 1.23 to 1.25 per cent for Sharp, but the maximum withdrawal increased to -9.48 per cent. The maximum withdrawal of the static double layer was −8.77 per cent, but with 207.36 per cent of the cumulative earnings and 1.22 Sharp lag behind the single layer and increase the hand-over. It has improved some of the risk indicators, but a fixed class view may still limit strong trend participation.")
    add_text(doc, "Thus, the value of the two-tiered structure is not to guarantee an increase in returns, but to provide an explanatory type of budgetary framework. The next step is to retain the budget at two levels, within the categories, and to liberalize the fixed group ratio, so that the category budget also changes with the trend.")
    page(doc, '2.8.1 Configure evidence Single layer dynamics and static category budget')
    weight_figures(doc, [(6, 'EWMA Single Layer Dynamic Budget'), (7, 'Two-tier budget for static category')],
        'The distribution is linked to the results: more static double layers of bonds and more thinly encrypted configurations, accompanied by fluctuations ranging from 7.52 per cent to 7.32 per cent on the single floor and withdrawals from minus 9.48 per cent to minus 8.77 per cent; however, the cumulative gains fell from 223.95 per cent to 207.36 per cent, and the change of hands increased. The fixed category budget provides another risk trade-off and does not simultaneously address under-receiving and adjustment losses.')

    page(doc, "2.9 How a fully dynamic two-tiered budget is structured")
    add_text(doc, "The outer layer ranks the average 12-1 month momentum for each selected category, while counting the proportion of assets in that category that meet the six-month momentum positive and above MA200. When it reached 50 per cent, 25 per cent and less than 25 per cent, the category trend multipliers were 1.50, 1.00 and 0.50, respectively. The budget for the outer layer is derived by multiplying the category ranking multiplier with the trend multiplier. The rule no longer incorporates the static 40/35/15/5/5 benchmark.")
    add_equation(doc, "B꜀ = Q꜀U꜀ / ΣₖQₖUₖ", "(11)")
    add_text(doc, "Within each category, the inner layer again produces a generic risk share of beta by asset movement ranking and trend status. In the end, the risk budget for each asset is the multiplier of the budget for the category and the share of the class, which is combined with a complete EWMA matrix to solve the weights.")
    add_equation(doc, "bᵢ = B꜀(ᵢ)βᵢ|꜀,   Σᵢbᵢ=1,   wᵢ(Σw)ᵢ/(wᵀΣw)=bᵢ", "(12)")
    add_text(doc, "Full dynamic double layer formation: cumulative gain of 270.99 per cent, Sharp 1.26 per cent, higher than the static double layer of 207.36 per cent, and 1.22 per cent, but maximum withdrawals increased from -8.77 per cent to -9.75 per cent, with a shift from 377.83 per cent to 397.89 per cent. Dynamic budgets improve income participation without simultaneously reducing all risks. It became the core module of the final strategy because of a better balance between benefits, risk structures and interpretability.")
    note(doc, "The above comparison is a comparison of the full strategic path and includes the combined effect of the change of target and the trigger of the transaction. The benefit contribution of a separate module is further prioritized at Part V and tested at Shapley.")
    page(doc, "2.10 Staple comparison between static and fully dynamic category budgets")
    weight_figures(doc, [(7, "Two-tier budget for static category"), (9, "Full dynamic double budget.")],
        "The distribution and results link: The dynamic budget allowed for greater financial participation of goods and encrypted assets at a partial trend stage, with the cumulative gains in the above table rising from 207.36 per cent to 270.99 per cent, but fluctuations also increased from 7.32 per cent to 8.49 per cent, with a withdrawal from 8.77 per cent to 9.75 per cent. This means that breaking the fixed class ratio increases the elasticity of returns and also assumes more trend asset risk, which cannot be described as cost-neutral.")

    page(doc, "2.11 Increased top-level participation in fixed attacks also magnifies risks and transactions")
    add_text(doc, "The relative basis of soundness and soundness EWMA has increased returns, but risk parity will still reduce the financial weight of high-variant trends. The fixed-attack programme combines a core configuration with a dispersed type of offensive configuration, in principle at 70 per cent core, 30 per cent attack configuration, and incorporates its own trends and crisis rules. The maximum number of targets for the attack is three, giving priority to different categories and limiting two or two absolute relevance.")
    add_equation(doc, "wᵗᵃʳᵍᵉᵗ = 0.70wᶜᵒʳᵉ + 0.30wᵃᵗᵗᵃᶜᵏ", "(13)")
    performance(doc, "Benefits risk versus transactions for the three core products", ["core", "fixed", "final"])
    add_text(doc, "Fixed offensives increased the cumulative earnings from 270.99 per cent to 362.55 per cent, but fluctuations rose from 8.49 per cent to 11.19 per cent, with the largest withdrawals increasing from -9.75 per cent to -15.51 per cent and Sharpe from 1.26 to 1.15 per cent. The annual change of hands rose to 446.31 per cent. Thus, fixed 30 per cent is not fully upgraded, and the main problem is that offensive capital ratios lack mechanisms to scale down with market breadth and portfolio status.")
    add_text(doc, "Dynamic programmes employ 0 per cent, 10 per cent, 20 per cent and 30 per cent slots and incorporate evacuation caps, VIX status and warehouse holding limits, reshaping the combination of offensive intensity and enforcement rules. The cumulative proceeds of 44.1.2 per cent, Sharp 1.32, 10.77 per cent fluctuations, maximum retreats - 14.98 per cent and hand exchanges - 34.1.7 per cent are better than fixed attacks in this sample. However, it remains more volatile and retreating than a robust core.")
    add_text(doc, "Dynamic attacks and fixed attacks do not only change the individual 30 per cent coefficient, and there are differences in their selection and implementation. As a result, the cumulative margin of return of about 79 percentage points cannot be attributed entirely to the dynamic ratio per se; the current results support the choice of a full dynamic option and are not sufficient to identify separately the causal contribution of each new rule.")

    page(doc, "2.12 Comparison of category weights for fixed and dynamic offensives")
    weight_figures(doc, [(10, "Resilient advance, steady 30%."), (11, "Psychic attack.")],
        "When a dynamic attack is zero, the funds remain in the core portfolio and cannot be understood as a whole liquidation. The fixed scheme itself permits the holding of cash, but its original weight map contains a cash superseding anomaly, which is described below; this part of the anomaly is not used to infer a difference in configuration or leverage. Performance comparisons of the three core strategies are based on the official income book.")
    note(doc, "Original map data validation: Fixed offensive maps showed over 100 per cent grey cash peaks around 2025. The drawing function mixs the cash record between the proceeds and the security weights after the close-off, at different points of time. This peak is not a real 170 per cent leverage; it has to be rebuilt from the cash column in the end of day waters on the same day before it can be used for a rigorous whole group weight comparison. Here, the source chart is retained and clearly stated as abnormal, without modifying the proceeds or assuming alternative weights.")
    page(doc, '2.12.1 A robust core and eventual dynamic defence with evidence')
    weight_figures(doc, [(9, 'It\'s a solid mind, a fully dynamic double core.'), (11, 'Psychic attack.')],
        'The distribution and results link: the final portfolio increased the participation of funds in categories such as encrypted assets, commodities, etc. at a strong trend stage, resulting in an increase of 170.53 percentage points in total inter-territory returns, an increase of 0.06 percentage points in Sharp, and a sharp increase of 5.23 percentage points in maximum withdrawals. An independent daily contribution reconciliation is used below to test the specific categories in which the proceeds fall and not to estimate them directly from the area of weight.')
    page(doc, '2.12.2 Reconcile changes in configuration with contributions from warehouse receipts')
    contrib = read_table(116)
    table(doc, 'Accumulated contribution of benefits corresponding to the change in configuration in percentage points', contrib,
          ['Policy', 'Trading Friction', 'Bonds', 'Crypto', 'Commodities', 'FX', 'Cash', 'Equities'], 7.9)
    add_text(doc, 'The relatively robust core saw an increase of 48.03 percentage points in the contribution of politically motivated equities, 79.84 percentage points in the number of encrypted assets and 45.67 percentage points in the number of commodities; the increase was partially offset by bond and wear. This is in line with the enhanced orientation of the strong trend category in the weighting chart. The final strategy \' s equity contribution increased by 56.67 percentage points relative to the fixed offensive, and the formal shift fell from 446.31 per cent to 34.17 per cent. These are ex post facto compartmentalizations of warehouse books, and the respective independent roles of offensive proportions and enforcement of the rules still need to be absorbed separately.')
    add_text(doc, 'The final solution is not just an increase in BTC revenues compared to fixed attacks. The cumulative contribution of equities rose from 107.91 to 16.58 percentage points, an increase of 56.67; commodities from 114.66 to 129.50, an increase of 14.84; and encrypted assets from 139.25 to 149.62, an increase of 10.37. As equities are the largest source of increases in this group of comparisons, the final strategy should be understood as a change in offensive choices with core configurations rather than a single BTC silo strategy.')
    core_ledger = read_table(116, 2)
    table(doc, 'Reconciliation of gains and losses under the same fixed-cost calibre', core_ledger.iloc[:, :6],
          ['Policy', 'Percentage of gross contribution of assets', 'SORA contribution percentage points', 'Trade loss percentage point', 'Percentage of net proceeds', 'Change hands.'], 8.2)
    add_text(doc, 'Under the official flat cost, the gross contribution of the spiritually protected assets is 451.27 percentage points, which is higher than the fixed attack of 373.23; the trade wears to 9.76 percentage points, which is smaller than the fixed attack to 12.63. The final net accrued gains were 44.1.2 per cent and 362.55 per cent, respectively. The increase in revenues and the decrease in wear and tear explain why the final choice is not simply to increase the fixed offensive ratio.')
    add_text(doc, 'This group of evidence links the preceding figure to performance: the weighting chart illustrates the direction of financial participation, the typologies contribution statement quantifys the actual yield point, and the net gain/loss statement reconciles the final investment results. It retains a combination of interactive and enforcement effects, on the basis of which it cannot be claimed that each transaction rule has been shown to be a positive contribution.')

    for period, title in [("In Sample", "2.13 Sample performance of all 12 strategies"),
                          ("Validation interval", "2.14 Interval performance of all twelve strategies"),
                          ("Complete research", "2.15 Full inter-regional performance of all 12 strategies")]:
        page(doc, title)
        performance(doc, "Formally measure key targets after the harmonization of the unilateral 0.05 per cent cost", ORDER, period)
        if period == "In Sample":
            add_text(doc, "The cumulative proceeds of the inner spirit offensive in the sample were 295.05 per cent, and Sharp 1.27, which was higher than the fixed attack 265.54 per cent, and 1.16; a robust core retreat - 9.75 per cent, which was significantly lower than the psychic attack - 14.98 per cent. The objective of the study is to determine the final choice not only by retreat or by ranking only by proceeds.")
        elif period == "Validation interval":
            add_text(doc, "The lead in the certification period has changed: static double-sharp 2.17, EWMA-ERC 2.16, single-layer dynamic 2.10, all higher than Psychiatry 1.61; Psychiatry defence revenues 37.08 per cent are also lower than EWMA-ERC 40.30 per cent. This suggests that the advantages of the final programme are a combination of return risks and stress performance across the study areas, rather than the first of all indicators during the validation period.")
        else:
            add_text(doc, "Full inter-discretional action was the highest of the 12 formal strategies, with a 104.14 percentage point reduction in the exchange of hands compared to fixed attacks. However, 342.1 per cent of the annual change of hands is much higher than the equivalent of 13.11 per cent, which is still a proactive arrangement requiring discipline. The robust core and the full asset ERC, respectively, provide an alternative location for lower retreat and lower turnover.")
        source = read_table(75)
        sub = source[source['Period'].str.startswith(period)]
        # Retain the remaining formal risk/operation metrics in the main text.
        extra = sub.set_index('Policy').loc[[NAMES[k] for k in ORDER],
              ['Sortino Ratio', 'Calmar Ratio', 'Positive-Day Ratio', 'Number of Rebalances', 'Number of Trade Legs']].reset_index()
        table(doc, "Formally retrospect downside risk and trading activity", extra,
              ["Policy", "Sortino Ratio", "Calmar Ratio", "Positive-Day Ratio", "Number of Rebalances", "Number of Trade Legs"], 8.1)
        note(doc, "The annual change of hands is based on a unilateral caliber; the number of legs traded in assets is a record of the targeted trading activity, which varies from one full warehouse to another and cannot be used as evidence in Alpha. The indicators are calculated independently for each period and the inter-territory gains cannot be added directly.")

    page(doc, "2.16 Lock-up is a supplementary observation rather than a basis for selection")
    holdout = read_table(55)
    table(doc, "The 44th of the 12 official strategies targeted for observation.", holdout[['Policy', *METRICS]],
          ["Policy", "Cumulative proceeds", "Annualized proceeds", "Yearic fluctuations", "Sharp.", "Maximum Drawdown", "Change hands."], 8.1)
    add_text(doc, "The targeted observation was 44 common trading days from 1 August to 1 October 2026. The annualized benefits of the short sample and the high sensitivity of Sharp cannot justify the long-term excess gains, nor can the final product be selected in reverse. Inter-research comparisons, validation periods and observations on this page remain independent throughout the text.")
    figure(doc, 75, 1, "Comparison of total inter-area benefits of the study of formal strategies")
    add_text(doc, "The final product was selected as an elitist defence based on the full range of higher yields versus Sharp, lower-handedness relative to fixed offensives, protection of the main downward pressure window, and the performance of earnings retained after actual cost. The robust core remains suitable for a more stringent retreat preference; fixed offensives remain a contrast to testing dynamic values.")


def part_three(doc):
    page(doc)
    add_section_heading(doc, "Part three: The final structure of the attack and the real deal.", 1)
    add_section_heading(doc, "3.1 Fixed cost versus real cost performance first", 2)
    cost = read_table(41).rename(columns={read_table(41).columns[0]: 'Compare objects'})
    table(doc, "Performance of the final strategy against the market at two cost scales", cost,
          ["Compare objects", "Cumulative proceeds", "Annualized proceeds", "Yearic fluctuations", "Sharp.", "Maximum Drawdown"], 8.6)
    figure(doc, 41, 1, "Cumulative benefits of fixed cost versus realistic differential cost")
    add_text(doc, "At the official flat cost, the cumulative proceeds of the attack were 441.52 per cent and Sharp 1.32 per cent. After maintaining the original signal and the trade path, joining the sub-market commissions, exchange of currency, price differentials, market shocks and withholding taxes, the cumulative gain fell to 364.12 per cent and Sharp 1.20, with the largest withdrawal from -14.98 per cent to -15.44 per cent. This indicates that the yield advantage is still retained under this cost assumption, while the cumulative gain loss of 77.40 percentage points is not negligible.")
    add_text(doc, "The parity benchmark on this page still uses a fixed 0.05 per cent cost and VT and STI follow the market reference sequence in the code. The real-cost strategy is thus a more rigorous demonstration comparison of the fixed-cost benchmark, and not all objects have been competing for net gains at the same realistic rates. Realistic copies do not recalculate the exit threshold arising from cost changes and order decisions, and are friction scenarios for freezing the course of transactions.")

    page(doc, "3.2 Seven-step construction and actual implementation of the final strategy")
    write_final_construction(doc)

    page(doc, "3.3 Proportion of eligible assets to dynamic offensive")
    add_text(doc, "A candidate for attack must simultaneously meet six months ' positive momentum, above MA 200, and 20 days ' fluctuations do not exceed 1.50 times the fluctuations of 126 days. The eligible breadth H is the proportion of assets in Top10 that meet all the conditions. The design allowed the attack to be based on a state of widespread trends and short-term fluctuations that had not been significantly amplified.")
    add_table(doc, ["Qualified breadth H", "Basic attack rate"], [
        ["H below 30 per cent", "0%"], ["30% to less than 50%", "10%"],
        ["50 to less than 70 per cent", "20%"], ["70% and above", "30%"],
    ], font_size=9.3)
    add_text(doc, "The broad slotting is also subject to a combination retreat: when the retreat reaches 5 per cent, 8 per cent and 12 per cent, the upper limit of the offensive is tightened to 20 per cent, 10 per cent and 0 per cent. VIX is in a state of crisis of more than 30, less than 25, and the state of crisis is zero. If there is only one qualified offensive hold, the ceiling is reduced to 15 per cent; when there is no mark, it is zero. Rescission of the co-restriction of targeted offensive funds with VIX was affected by the frequency of the 15-day inspection and was not continuous real-time loss.")
    add_equation(doc, "Sᵢ = [0.50Mᵢ,6M+0.50Mᵢ,12−1]/σᵢ,63D", "(15)")
    add_text(doc, "The offensive assets are ranked in terms of the risk-adjusted momentum of the type (15), with priority given to different categories before other eligible assets are considered; a maximum of three items are selected, and if any new asset has an absolute correlation with the selected asset of 0.75 days, it will skip. Select the power configuration of the object of the attack. This has helped to avoid concentrating the 30 per cent offensive entirely on a few highly related ETFs, but the limited category Top10 may not be fully dispersed.")
    add_text(doc, "The mechanism improves the possibility that fixed offensives may still take a higher risk at a vulnerable stage, and there is a backsliding: when retreats occur, they contract, and when quick rebounds, they may fall short of repositioning. The results of the rebound window in 2020 reflect this trade-off and the stress chapter is presented in its entirety.")

    page(doc, "3.4 Reduction in the correspondence between hand change and proceeds of implementation")
    add_text(doc, "The strategy is checked every 15 trading days. The Top10 list changes or is fully implemented at initial build-up; the list remains unchanged towards only half of the new target and retains the current weight for changes in proposed transactions of less than 2 percentage points, which are then consolidated. It is used to reduce small amounts and rebalancing repeatedly, and de facto integration may still fine-tune individual small weights and cannot be understood as an absolute lower limit for each final deal.")
    add_equation(doc, "wᵖᵃʳᵗᵢᵃˡₜ = wᶜᵘʳʳₜ + 0.50(wᵗᵃʳᵍᵉᵗₜ−wᶜᵘʳʳₜ)", "(16)")
    add_text(doc, "Signals, holdovers and transaction costs must be placed in the right time order. The proceeds from the t-to-t-+1 closing of the asset are the old holding stock before the transaction is made; t-+1 closing is calculated as the amount and cost of the transaction using the weight of the old hold-out drifting to the new target. New targets cannot receive pre-offer returns.")
    add_equation(doc, "gₜ = Σᵢwᵖʳᵉᵢ,ₜrᵢ,ₜ + wᶜᵃˢʰₜrᶠₜ", "(17)")
    add_equation(doc, "cₜ = 0.0005·Σᵢ|Δwᵢ,ₜ|,   rⁿᵉᵗₜ = (1+gₜ)(1−cₜ)−1", "(18)")
    add_text(doc, "The study of the complete inter-temporal dynamical unilateral replacement of 342.1 per cent was lower than the fixed offensive of 446.31 per cent and the robust core of 397.89 per cent. This supports a more restraining approach to transactions in the final complete programme, but does not specifically remove the separation between the semi-implementation and non-trading areas and does not allow for the precise allocation of all the changes to one of the rules.")
    note(doc, "Unilateral exchange of hands is half the sum of the transactional weights; costs are aggregated on the basis of actual purchases and sales transactions, and the unilateral swap of 0.05 per cent cannot be considered a full bilateral cost. Boundary processing such as initial build-up is based on statistical functions.")

    page(doc, "3.5 Parameters of real transaction costs and market shocks")
    add_table(doc, ["Listed market", "Unilateral commissions and transaction costs", "Unilateral price differentials and slide points"], [
        ["United States", "0.02%", "0.03%"], ["Germany", "0.08%", "0.08%"],
        ["United Kingdom", "0.05%", "0.08%"], ["Japan", "0.05%", "0.06%"],
        ["Mainland China", "0.03%", "0.10%"], ["Hong Kong SAR", "0.08%", "0.08%"],
        ["Singapore", "0.06%", "0.08%"], ["Bitcoin is a separate hypothesis.", "0.15%", "0.15%"],
    ], font_size=9)
    add_text(doc, "Exchanges are calculated as extras in exchange: SSD is zero, USD is 0.1%, EUR, GBP, JPY, HKD is 0.15% and CNY is 0.20%. These are artificial fixed scenario parameters, without the use of historical coupon rates; each foreign currency sale is represented by a currency exchange, which may be more conservative than the implementation of cash accounts and netting of funds in foreign currency.")
    add_equation(doc, "Impactᵢ,ₜ = min[0.01, 0.10·σᵢ,ₜ₋₁·√(Qᵢ,ₜ/ADVᵢ,ₜ₋₁)]", "(19)")
    add_text(doc, "Q is the amount of the transaction, and ADV is the median of the 20-day cut-off of SDD, considering the 20-day run-off of earnings, with a combined initial size of SSD 1 million. The impact rate is multiplied by the trade weight and becomes a combination cost. The withholding tax is deducted on a day-to-day basis by the annualized agent of the holding asset class; this is not the actual red-day tax. The market shock reflects a volume assumption that the true order book is not calibrated; the missing transaction value will result in the corresponding impact item being zero, so that the true capacity of the entire target can not be claimed.")
    add_text(doc, "A copy of the real cost is used to test the degree to which frictions are tolerated and cannot be a substitute for real market investmentability checks. The availability of inland ETFs, the BTC historical trading vehicle and the time of closing across the market remain the product boundary; the same date labels are not equal to the time of global synchronization.")

    page(doc, "3.6 Evidence of product positioning and final selection")
    add_text(doc, "Psychic protection is positioned as a global, non-nominal leverage-based, multi-asset proactive configuration that measures performance with SSD. Target clients are willing to take a double-digit retreat, want to participate on a decentralized basis, and can accept rule-based accommodation and cross-market transactions. The proceeds are not fixed interest rates and are not suitable for clients who place short-term principal stability first.")
    add_table(doc, ["Compare objects", "The Advantage of the Final Strategy", "The price to accept."], [
        ["18 Rights, etc.", "More complete interfacing with Sharp, better protected by the main crisis window.", "The hand exchange and the model are more complex, and a quick rebound may fall behind."],
        ["EWMA-ERC", "Accumulated inter-territory gains and top-level participation", "Higher volatility, deeper maximum retreat, lower validation period."],
        ["GMI-AHRB", "Better earnings elasticity, lower total turnover.", "We can't keep a steady core smaller retreat."],
        ["GMI-FOA", "Full inter-sector gains, Sharpes and hand change improved", "Differences include multiple rules and cannot be attributed solely to dynamic ratios"],
    ], widths=[3, 7, 7], font_size=8.6)
    add_text(doc, "Pressure evidence is an important component of the final choice. The benefits of the outbreak window +0.79 per cent, the fallback - 5.17 per cent, the 2022 double-mortgage gains - 0.43 per cent, and the fallback - 5.18 per cent are clearly better than equal rights. In crisis-weighted Bootstream, the final strategy has a higher median return, lower probability of loss and lighter tail-backing relative fixed offensive. Part IV further lists all the windows to test where this advantage is established and where it disappears.")


def write_final_construction(doc):
    add_text(doc, 'The final strategy is to generate a real setup in the following seven steps. The first four steps form the core combination, the fifth stage constructs the attack configuration, the sixth moves the two into implementable targets, and the seventh steps are traded and gains and losses are calculated. The risk budget is established before the joint weight resolution; the capital weight is not commingled with the offensive portfolio until it is resolved.')
    add_table(doc, ['Steps', 'Completed calculations', 'Send to Next Output'], [
        ['Step one.', 'Top10 of 18 assets at 12-1 month movements', 'Candidate pool and asset score'],
        ['Step two.', 'Development of category budgets based on categories \' drive and trend breadth', 'Share of risks in selected categories'],
        ['Step three.', 'Formulating and multiplying budget lines', 'Target risk share for each asset'],
        ['Step four.', 'EWMA co-ordinated with target budget resolution', 'Core portfolio funding weights'],
        ['Step five.', 'Screening of targets and determination of 0-30 per cent', 'Weight of attack funds versus mix ratio'],
        ['Step six.', 'Mixed two combinations and apply implementation buffers', 'Actual target weight to be implemented'],
        ['Step seven.', 'The next transaction will be closed and charged.', 'Net gain and real hold on the deal.'],
    ], widths=[2.2, 8.0, 6.8], font_size=8.7)
    add_section_heading(doc, 'Step one: Top10 from 18 assets.', 3)
    add_text(doc, 'After closing up at each inspection day, the amount of SSD proceeds up to the date of the 252 transactions was taken out of the last 21 days and the movement of each asset was calculated in formula (8). Sorts the fractions from high to low, keeps the first 10 items and is assembled as S. At that time, only the asset selection had been completed and the funds had not yet been allocated; Top10 did not represent 10 per cent of each of the 10 assets held.')
    add_equation(doc, '', '(14a)')
    add_section_heading(doc, 'The second step is to determine how much risk each of the categories would take.', 3)
    add_text(doc, 'Comparisons were made only in the categories in which Top10 actually appeared. Q indicates the ranking multiplier for the average momentum of this category, and U indicates the trend multiplier corresponding to the proportion of assets meeting a positive six-month drive and above MA200. The combination of the two received category B budgets. If a class does not have assets into Top10, it is not budgeted at the core for the current period.')
    add_equation(doc, '', '(14b)')
    add_text(doc, 'The output of this step is the share of risk rather than the share of funds. For example, with a higher budget for a certain category, the ultimate level of investment still depends on the volatility of the category and its relevance to other assets. The fully dynamic rules also no longer pre-fix static ratios of 40 per cent of equities and 35 per cent of bonds.')
    add_section_heading(doc, 'Step 3 Decision on risk budgets within categories', 3)
    add_text(doc, 'Within each category, the movement of assets is re-ranked, with q representing the ranking multiplier within the class and u representing the six-month trend and the status multiplier corresponding to MA200. When consolidated, the risk ratio within category is obtained, multiplied by category B, which forms the final budget for each assetb. Total of all selected assetsb is 1.')
    add_equation(doc, '', '(14c)')
    add_section_heading(doc, 'Step four. Converting risk budgets into core funding rights with EWMA Heavy', 3)
    add_text(doc, 'For the same Top10, the λ = 0.94 estimate of the immediate association matrix, combined with a non-negative, combined core weight of 1, bring each asset \' s risk contribution close to its budgetb. The two tiers of budget formulation are within categories and classes, and the weight of assets is still solved simultaneously in a complete alignment matrix, rather than by the aggregation of the two ERCs.')
    add_equation(doc, '', '(14d)')
    add_text(doc, 'If budgets for high-volatile assets were to increase, the funding would not necessarily have a higher weight than those for low-volatile assets. Risk budgeting, risk contribution and funding weights must be understood separately here. The resulting core combination is the fully dynamic two-tiered EWMA configuration.')
    add_section_heading(doc, 'Step five, build a combination of attacks and determine the rate of current attacks.', 3)
    add_text(doc, 'Top10 sifts assets that simultaneously satisfy six months \' positive momentum and are above MA200, without abnormal short-term fluctuations. For the indicator variable I meets these conditions, H is the eligible width. The base offensive ratio is selected between 0%, 10%, 20% and 30%, and is then limited downwards by combination retreat, VIX crisis state and eligible holding stock.')
    add_equation(doc, '', '(14e)')
    add_text(doc, 'Qualified assets are ranked in terms of complex risk-adjusted kinetics in the form (15), with priority given to different categories, excluding candidates with absolute relevance to the selected target of 0.75, with a maximum of three concurrent powers to form an offensive combination. No attacks are carried out without a qualified target, with a maximum of 15 per cent at one point. The full threshold is shown in section 3.3.')
    add_section_heading(doc, 'Step six: Combining the core with the attack and forming the actual target.', 3)
    add_text(doc, 'The current offensive ratioa combines two groups of funds with linear weights. a Full core for 0:00; a 70 per cent core and 30 per cent attack for 30 per cent. The 30 per cent is a capital ratio, and the final composition of the mix does not have to continue to meet the core strict risk budget.')
    add_equation(doc, '', '(14f)')
    add_text(doc, 'In the event of a change in Top10 membership or initial build-up, complete new targets are implemented. If the list remains unchanged, only 50 per cent will be moved to the new target and changes in proposed transactions of less than 2 percentage points will be retained and consolidated. Theoretically mixed weights may therefore differ from the actual transaction objectives, and subsequent proceeds must be attributed to the use of physical holdouts.')
    add_section_heading(doc, 'Step seven, close the deal and record the net gain.', 3)
    add_text(doc, 'The target generated by the t-day is closed only at t+1. Revenues from t to t+1 are still received from the old holder, which is then calculated as the difference between the old weights after drifting, and a single unilateral cost of 0.05 per cent based on the sale. The new weight is effective from the next yield area after the transaction. g Represents gross proceeds on the date of the old hold and c indicates transaction costs for this portfolio.')
    add_equation(doc, '', '(14g)')
    add_text(doc, 'This process is checked every 15 transaction days and continuously records the date of the signal, the date of execution, the state of the offensive, the balance of the transaction and the weight of the transaction. It explains how the final product went from data to hold and why the signal day gain could not be directly calculated using the signal day target weight.')


def historical_data():
    a = read_table(43)
    a = a.rename(columns={'Start date': 'Start date of transaction', 'End Date': 'End date of transaction'})
    b = read_table(57)
    return pd.concat([a, b], ignore_index=True)


def part_four(doc):
    page(doc)
    add_section_heading(doc, "Part IV. Pressure testing and robustness", 1)
    add_section_heading(doc, "4.1 Object of pressure test and method of determination", 2)
    add_text(doc, "The stress test answers three different types of questions: how the historical path actually works, how much the current hold could be lost by a hypothetical shock, and how the historical gains might change in the tail after the re-grouping. The rules of historical resonance allow for encumbrance along the path; the end-of-cycle combined shock freeze hold; and the benefits of the Bootstream re-sampling strategy. The three cannot be confused with the same prediction. The following stress result follows a uniform master return cost of 0.05 per cent, with realistic cost scenarios presented separately in Parts III and V.")
    hist = historical_data()
    windows = hist[['Pressure scenario', 'Start date of transaction', 'End date of transaction']].drop_duplicates()
    table(doc, "All nine historical pressures and reverse windows", windows,
          ['History Window', 'Start date', 'End Date'], 8.8)
    add_text(doc, "The nine windows included a systemic decline, an appreciation of the United States dollar, a collapse of single assets, commodity performance and a rapid rebound. The windows overlap and cannot be treated as nine independent experiments. Compare the seven strategies covered by the original test: equal rights, full-assets classic balance, basic EWMA, unilayer dynamics and three core products.")
    add_section_heading(doc, "4.2 Defence strengths in key crisis windows", 2)
    add_text(doc, "The final strategic gains from the epidemic are +0.79 per cent, the maximum recall - 5.17 per cent, the equivalent rights - 16.84 per cent, -17.87 per cent, and the final strategic gains from the double-killing of 2022 shares - 0.43 per cent, the retreat - 5.18 per cent, the equivalent rights - 12.78 per cent, -15.66 per cent. These two sets of evidence suggest that the model reduces losses at the stage of systemic pressure in the sample, but does not induce future crises to be profitable.")
    page(doc, "4.3 Comparison of benefits for all historical windows")
    names = [NAMES[k] for k in ['equal', 'monthly', 'ewma', 'single', 'core', 'fixed', 'final']]
    returns = hist.pivot(index='Pressure scenario', columns='Policy', values='Cumulative Return')
    returns = returns.reindex(index=windows['Pressure scenario'], columns=names).reset_index()
    table(doc, "Cumulative benefits of historical pressure windows in percentages", returns,
          ['Window', 'Equal rights', 'Full Assets ERC', 'EWMA', 'GMI-DRB', 'GMI-AHRB', 'Hold your fire!', 'GMI-ADG'], 7.7)
    figure(doc, 43, 1, "Cumulative gains from the core strategy and historical stress phase")
    add_text(doc, "The final strategy was +3.45 per cent in the United States dollar appreciation window, the highest of seven; the crude oil abnormality window was +3.14 per cent, which was also higher than the comparison. However, the rebound after the low point in 2020 was only +3.20 per cent, which is clearly less than equal rights +19.39 per cent. Trends and risk control decrease exposure and may also reduce V-shaped rebound participation.")

    page(doc, "4.4 Return comparison of all historical windows")
    drawdowns = hist.pivot(index='Pressure scenario', columns='Policy', values='Maximum Drawdown')
    drawdowns = drawdowns.reindex(index=windows['Pressure scenario'], columns=names).reset_index()
    table(doc, "Maximum echo of historical pressure window in percentage", drawdowns,
          ['Window', 'Equal rights', 'Full Assets ERC', 'EWMA', 'GMI-DRB', 'GMI-AHRB', 'Hold your fire!', 'GMI-ADG'], 7.7)
    figure(doc, 57, 1, "All extended historical window returns hot pursuit")
    add_text(doc, "Psychic offensives have a maximum retreat of less than equal rights in the nine historical windows, but not always superior to other active programmes. Bitcoin crash window retreats - 7.91 per cent, greater than base EWMA - 4.10 per cent; commodity boom window retreats - 8.23 per cent, greater than total asset ERC - 4.06 per cent. As a result, its defensive advantage is largely equivalent to a partial fixed attack scenario, with a robust core still offering lower risk options.")

    page(doc, "4.5 Volatility and single-day end of the three core strategies")
    risk = hist[hist['Policy'].isin(CORE)][['Pressure scenario', 'Policy', 'Annualized Volatility', 'Worst day of earnings']]
    table(doc, "All three core product windows", risk,
          ['Window', 'Policy', 'Yearic fluctuations', 'Worst single day'], 8.0)
    add_text(doc, "The maximum range retreat is different from the worst single-day gain: the former measure the loss of a continuous net value, while the latter measure the risk of a single-day jump. The final strategy is smaller in some windows and may still bear a larger single-day decline; it cannot therefore be used to improve the management of liquidity and flight risks.")

    page(doc, "4.6 Assuming that the combined impact test is the end-of-life holdout")
    shock = read_table(63, 3)
    pivot = shock.pivot(index='Simulation', columns='Policy', values='Combination of one month of impact gains')
    pivot = pivot.reindex(columns=CORE).reset_index()
    table(doc, "End-of-cycle hold losses under five scenarios of combined shock", pivot,
          ['Scenario', 'GMI-AHRB', 'GMI-FOA', 'GMI-ADG'], 8.8)
    figure(doc, 63, 1, "At the end of the period, the real hold was under a supposed combined shock.")
    add_text(doc, "This set of tests will foresee a monthly asset shock that directly affects the real weight at the end of 31 July 2026. The final strategy was to lose -6.22 per cent of the inflationary stock debt, less than the fixed offensive - 6.86 per cent and the robust core - 8.22 per cent; and to lose - 26.86 per cent in the liquidity crisis. Dynamic rules do not re-reciprocate within the months of the shock, so the result should be interpreted as the current silo vulnerability, rather than the projected return on the full strategy for the coming month.")
    add_text(doc, "In fast-risk rebound scenarios, the final strategy + 16.86 per cent is slightly less than two comparisons. The scenarios do not cover the full range of risks and, in particular, small losses in several scenarios cannot be considered the maximum loss cap.")

    page(doc, "4.7 Proceeds from block Bootstream and evacuation distribution")
    boot = read_table(65)
    for method in boot['Sample method'].unique():
        selected = boot[boot['Sample method'].eq(method)].drop(columns='Sample method')
        table(doc, f"{method}1000 252-day paths.", selected,
              ['Policy', '1% share of proceeds', '5% share of proceeds', 'Medium proceeds', 'Probability of loss', 'Revert Middle', 'Return 5% tail.', 'Over 20% chance of retreat.'], 7.4)
    figure(doc, 65, 1, "Distribution of income at the end of Bootstream as at 252")
    add_text(doc, "The median final strategic gains of 15.55 per cent were higher than those of fixed offensives by 14.53 per cent in the average sample; the probability of loss was 7.90 per cent, or below 12.3 per cent. The median return on the final post-crisis weighted strategy was 12.33 per cent and the probability of loss was 13.50 per cent, still better than fixed offensives of 10.19 per cent and 17.30 per cent. This is one of the main stress advantages of dynamic programmes relative to fixed attacks.")
    page(doc, "4.8 Backward improvements and the trade-off between a robust core")
    figure(doc, 65, 2, "Bootstream maximum retreat distribution on 252")
    add_text(doc, "Under the weight of the crisis, the maximum exit strategy was 5 per cent -14.19 per cent at the end, which was better than the fixed offensive -15.78 per cent; and the rate of evacuations exceeding 20 per cent was 0.20 per cent, or less than 1.30 per cent. However, the back end of the robust core is smaller, at minus 11.32 per cent, and the probability of a loss of 11.80 per cent is lower than the final strategy. The ultimate choice is to retain the proceeds of the attack in exchange for greater end risk in exchange for elasticity, which cannot be called the lowest risk product.")
    add_text(doc, "Sampling paths are derived from historical strategic gains and are partially relevant within the reserved blocks, but do not re-create asset prices, signals and warehousing. Such tests measure the uncertainty behind the re-routing of historical gains, do not exhaust an unprecedented crisis and do not automatically reflect strategic feedback in a new crisis.")
    table(doc, "One-day history, VAR and ES, initial funding, SSD 1 million.", read_table(67),
          ['Policy', 'Trust Horizontal', 'Observation', 'End Number', '1 VaR', '1 January ES', 'VaR Amount SGD', 'ES Amount SGD'], 7.6)
    note(doc, "VAR and ES are shown as losses. Ninety-five per cent of VaR is the loss of the corresponding sub-point, and ES is the average loss at the end of the line above that threshold, neither of which is the upper limit of the loss.")

    page(doc, "4.9 Full comparison of VAR and ES at 252")
    yearly = read_table(67, 2)
    for method in yearly['Sample method'].unique():
        sub = yearly[yearly['Sample method'].eq(method)].drop(columns='Sample method')
        table(doc, f"{method}End-of-year path loss", sub,
              ['Policy', 'Trust Horizontal', 'Paths', 'End Number', '252 VaR', '252 ES', 'VaR Amount SGD', 'ES Amount SGD'], 7.7)
    add_text(doc, "At 99 per cent confidence, the final strategy, VaR 11.64 per cent, ZXQ0QZ per cent, was lower than the fixed offensive of 14.76 per cent, 17.49 per cent and higher than the robust core of 9.45 per cent and 10.98 per cent. This ranking is consistent with the distribution of revenues and the maximum withdrawal: a dynamic offensive improves risk relative to a fixed offensive, but does not replace a more conservative core combination.")

    page(doc, "4.10 Cost and delayed stress tests")
    costs = read_table(59)
    costs = costs[costs['Policy'].isin(CORE)]
    table(doc, "All fixed cost multiple tests for the three core strategies", costs,
          ['Policy', 'Cost multiplier', 'Cumulative proceeds', 'Annualized proceeds', 'Sharp.', 'Maximum Drawdown', 'Change hands.'], 8.1)
    delayed = read_table(61)
    delayed = delayed[delayed['Policy'].isin(CORE)]
    table(doc, "Delay one more day from the closing date of the original transaction", delayed[
        ['Policy', 'Original cumulative rate of return', 'Cumulative rate of return after delay', 'Former Sharp ratio', 'Sharp ratio after delay', 'Max. Back.', 'Maximum retreat after delay']],
        ['Policy', 'Original proceeds', 'Delayed gains', 'Old Sharp.', 'Delay Sharp.', 'Back', 'Delayed retreat'], 8.0)
    add_text(doc, "When the cost doubled, Sharp ' s final strategy remained at 1.29; the additional delay of one day resulted in a cumulative gain of 425.01 per cent and Sharp 1.31. The results suggest that advantages do not exist only in the most optimistic fixed rates or single-day implementation assumptions, but that these tests do not address the different closing times in the real market, partial engagements and extreme spreads. The actual variable cost is presented in a complete copy instead of using the fixed rate results of this table as a realistic record.")

    page(doc, "4.11 Parameter sensitivity requires a distinction between test coverage")
    table(doc, "Results for all parameters of the base EWMA path", read_table(83),
          ['Dimensions', 'Parameter Value', 'Cumulative proceeds', 'Annualized proceeds', 'Sharp.', 'Maximum Drawdown', 'Change hands.'], 8.1)
    add_text(doc, "Top N, from 8 to 10 p.m., was about 1.22 to 1.21 per cent in Sharp; while N was increasing gains, it was withdrawn to -23.57 per cent. The dynamic window 189 is close to the 231-day result, and the 63-day Shap is only 0.75. From 0.94 to 0.98, Sharp dropped from 1.21 to 1.12 and dropped simultaneously; the 15-day interval between inspections is better than 20 or 21 days, but does not represent the best in the future.")
    add_text(doc, "These tables are based on the EWMA path, the cumulative proceeds from the baseline of 205.27 per cent are not the same as those from the official EWMA of 208.73 per cent, and each should be retained with its own caliber and cannot be combined with the same performance. They support the local soundness of the basic design and have not yet covered the threshold of the final strategy for offensive breadth, fallback slots, correlation limits and implementation buffers.")
    page(doc, "4.12 Base parameter maps and BTC dependency check")
    figure(doc, 83, 1, "Top N momentum window EWMA decay and frequency sensitivity check")
    nobtc = read_table(81)
    table(doc, "Include BTC and remove the BTC base EWMA study path", nobtc.iloc[:, [0, 1, 2, 4, 7, 8, 9]],
          ['Experiment', 'Start date', 'End Date', 'Cumulative proceeds', 'Sharp.', 'Maximum Drawdown', 'Change hands.'], 8.0)
    add_text(doc, "After the elimination of the BTC between the same zones, the cumulative proceeds from the base path decreased from 205.27 per cent to 125.82 per cent and from 1.21 to 0.92. The results show that the BTC in the asset pool is an important source of revenue; it is not the last dynamic attack to rerun the BTC and cannot be directly linked to the final strategy. The BTC replacement and investable versions of the final product still require specialized testing.")

    page(doc, "4.13 Each year Walker forward and multiple strategy choices")
    wf = read_table(87)
    for metric, label in [('Cumulative Return', 'Full annual returns'), ('Sharpe Ratio', 'Every full year, Sharp.')]:
        pivot = wf.pivot(index='Forward year', columns='Policy', values=metric)
        pivot = pivot.reindex(columns=[NAMES[k] for k in ['equal', 'monthly', 'ewma', 'core', 'fixed', 'final']]).reset_index()
        table(doc, label, pivot, ['Year', 'Equal rights', 'Full Assets ERC', 'EWMA', 'GMI-AHRB', 'GMI-FOA', 'GMI-ADG'], 8.0)
    add_text(doc, "The final strategy was in 2022 - 0.43 per cent, significantly better than the equivalent rights - 12.78 per cent; in 2024 - +31.52 per cent, higher than the equivalent rights - +20.39 per cent. However, in 2025, 19.94 per cent of the proceeds were lower than the basic EWMA 22.63 per cent. An annual roll-out allows for a change in strengths and avoids being masked by an entire curve.")
    page(doc, "4.14 Annual Consistency and Deflated Sharpe")
    figure(doc, 87, 1, "The annual Walk forward Sharp comparison of existing rules")
    table(doc, "All six Walker versus consistency statistics", read_table(87, 2),
          ['Policy', 'Sharp Middle', 'Sharp minimum', 'Year of positive return', 'Worst year back.', 'The year Sharp wins.'], 8.2)
    dsr = read_table(89, 2)
    table(doc, "Multiple choice correction for three core strategies", dsr[dsr.iloc[:, 0].isin(CORE)].iloc[:, [0, 1, 4, 5, 6, 7]],
          ['Policy', 'General annual Sharp.', 'DSR probability', 'Argument', 'Peak', 'Comparison of strategies'], 8.5)
    add_text(doc, "The final strategy is yielding 80 per cent of the annual rate of return and 60 per cent of the annual rate of overweight in Sharp; the robust core is 60 per cent. The current annual assessment is a year-by-year split after the current rule of freezing, and does not mean that all research decisions taken today have been frozen in advance since 2016. Deflad Sharpe had only 44.56 per cent of the 12 formal candidates, while history had been explored more than 12 times, and the actual selection bias was likely to be greater. The report therefore supports product research choices and does not interpret historical excellence as a proven long-term Alpha.")

    page(doc, "4.15 How stress outcomes support the selection of final strategies")
    add_text(doc, "The advantage of dynamic programmes in relation to fixed offensives comes not only from complete inter-zonal gains. The epidemic has improved from 2.93 per cent of fixed attacks to +0.79 per cent, from −2.99 per cent to −0.43 per cent of the 2022 stock of debt; the probability of losses from crisis-weighted Bootstream has dropped from 17.30 per cent to 13.50 per cent, and from 17.49 per cent to 13.86 per cent of the year for ES. These different tests together support dynamic programmes as the current primary product.")
    add_text(doc, "Relatively robust cores, the final strategy does not gain full defensive advantage. The latter remained more robust in most years and at the end of Bootstream. The rational positioning of the final product is to preserve stronger trend returns and improve the risk efficiency of fixed offensives, while the soundness of intelligence is the product option for lower retreats. There is still a need to accept the corresponding risk of loss and flight for a rapid rebound, a sudden collapse of encrypted assets and a global liquidity crisis.")
    add_text(doc, "The available evidence also contains clear boundaries: the current official results do not yet provide a uniform caliber 60/40 against the random Top10 set, the underlying parameters and the go-BTC experiment do not cover the final offensive rules, and the White Reality Check, SPA and PBO do not produce quoted outputs. These are studies that can be added and are not written or adopted in the present report.")


def part_five(doc):
    page(doc)
    add_section_heading(doc, "Part V. Attribution of performance and findings", 1)
    add_section_heading(doc, "5.1 Contribution of assets explains where each proceeds comes from", 2)
    add_text(doc, "Performance is attributed to the fact that it begins with real hold, rather than to the fact that returns are explained. Daily gains are earned by multiplying the value of the funds before handing over, and losses and losses of cash and transactions are recorded separately, with geometric links accumulated to the end of the period. Catalogue contributions and target contributions are expressed as percentage points of cumulative proceeds, and totals should be reconciled with net aggregate combined proceeds. It reflects both the revenue from holding and the timing of the investment and the recovery path.")
    add_equation(doc, "cᵢ,ₜ = wᵖʳᵉᵢ,ₜrᵢ,ₜ,   Σᵢcᵢ,ₜ+cᶜᵃˢʰₜ+cᶜᵒˢᵗₜ=rⁿᵉᵗₜ", "(20)")
    figure(doc, 95, 1, "Psychic precipitating the contribution of cumulative gains by large asset classes")
    table(doc, "Accumulated contribution of large asset classes to the three core strategies in percentage points", read_table(116),
          ['Policy', 'Trading Friction', 'Bonds', 'Crypto', 'Commodities', 'FX', 'Cash', 'Equities'], 7.9)
    add_text(doc, "Anticipatory equity contribution of 16.58 percentage points, encrypted assets of 149.62, commodities of 129.50, foreign exchange of 10.50, bonds - 2.93, trade losses - 9.76. Compared with the robust core, the gains from the final product are mainly reflected in equity, encryption and commodity trend participation. The increase in equity contributions was particularly marked when compared to fixed attacks. The breakdown is recorded in a post facto warehouse and cannot be interpreted to mean that each asset class will still provide the same benefit in the future.")

    page(doc, "5.2 Single target and centralized source of proceeds")
    figure(doc, 95, 2, "Psychic precipitating the contribution of cumulative gains by disaggregating the target assets", width=5.65)
    asset = read_table(95)
    asset = asset[asset['Asset Class'].ne('Trading Friction')]
    table(doc, "Detailed contributions by target", asset.iloc[:, [0, 3, 4, 1]],
          ['Subject', 'Asset Name', 'Asset Class', 'Percentage of cumulative contribution'], 7.7)
    add_text(doc, "Bitcoin contributed 149.62 percentage points, gold 85.93, ZXQ0QZSI Channel Times index 52.53, industrial metal 28.82, QQQ 24.76. The higher aggregate returns depend on a small number of long-term strong trend sources, and the number of assets alone cannot be used to show that the actual gains have been fully dispersed. There is also a difference between a concentration of proceeds and a concentration of financial weights: highly volatile assets are held over a long period of time with smaller weights and may still contribute significantly.")
    note(doc, "All 18 securities or subject-matter assets and SORA cash are shown in the table; the items of wear and tear of the transaction are shown independently in section 5.8.")

    page(doc, "5.3 Cost of layer-by-layer contribution and risk efficiency")
    table(doc, "Fixed module increment in order", read_table(97),
          ['Steps', 'Phase', 'Cumulative proceeds at this stage', 'Relatively forward step increment'], 8.8)
    figure(doc, 97, 1, "Incremental contribution from equal rights to final strategy")
    add_text(doc, "The cumulative gain of 330.55 per cent on the right to the exclusive drive of Top10 is higher than the right of 211.15 per cent on the right to 18 assets; the EWMA risk parity is reduced to 208.73 per cent. This means that risk parity is not a benefit enhancement module in this sample, but its main value is risk reduction and risk diversification contributions. The fully dynamic budget increased returns to 270.99 per cent, and eventually the complete dynamic offensive and implementation programme reached 44.52 per cent.")
    add_text(doc, "This table uses the cumulative yield differential of a fixed sequence of adjacent strategies, in percentage points, and is not an overall independent Alpha per day. Each stage interacts with the subsequent module, and changing the order of addition will change the increment. The last step, which includes both offensive selection, dynamic funding ratios and implementation design, cannot be identified as a single coefficient of return. Top 10, etc., is not a formal product of item 13.")

    page(doc, "5.4 There is a significant interaction between the Shapley Note modules")
    table(doc, "The results of all eight alliances in the three modules", read_table(113),
          ['Enable module', 'Number of modules', 'Cumulative proceeds'], 8.7)
    table(doc, "Three modules Shapley average marginal contribution", read_table(113, 2),
          ['Modules or Check Items', 'Percentage increase in cumulative gains'], 8.7)
    add_text(doc, "Shapley takes averages of all additions to the kinetic filters, EWMA risk parities, and a fully dynamic double budget. Risk parities are negative, reflecting their relative high-risk configurations limiting returns; it cannot be inferred that the deletion of risk parities increases risk adjustment performance. In some unions, the cumulative gains were over 800 per cent, and there was a greater need to look at their fluctuations and reversals than to select them on the basis of the cumulative gains alone.")
    add_text(doc, "The cumulative benefit of the three complete modules of the experiment is 269.97 per cent, which does not amount to a formal robust core of 270.99 per cent, nor does it include final dynamic offensive and enforcement rules. It uses an independent research path and can therefore only be used to explain the interactive structure of the three modules and cannot be forcibly reconciled with the official sequence incremental scale. A separate response to the dynamic attack on its own Shapley contribution will require a further expansion of the coalition design.")

    page(doc, "5.5 Attribution should evolve from external interpretation to internal rules")
    add_equation(doc, "rₚ,ₜ−rᶠₜ = α + Σₖβₖfₖ,ₜ + εₜ", "(21)")
    add_text(doc, "The four-factor model uses global stocks, medium-term national debt, commodity baskets and United States dollars to calculate excess earnings at daily SORA and uses HAC standards. It estimates the annualized remnants of Alpha 11.47 per cent for the final strategy. The model, however, does not contain separate gold, bitcoin, area and momentum, and amplitudes may absorb leakage exposure and cannot be considered as directly tradable Alpha.")
    table(doc, "Four factors for three core products back.", read_table(116, 3),
          ['Policy', 'Stock beta', '[ Chuckles ]', 'Commodity beta', 'United States dollar beta', 'Alpha', 'HAC t', 'HAC p', 'R Square'], 7.7)
    table(doc, "Results of the insertion of external factors and internal rules", read_table(109),
          ['Model', 'Factor', 'R Square', 'Adjust R Square', 'Yearic Alpha', 'HAC t', 'HAC p'], 8.1)
    add_text(doc, "Using external market and style factors only, Alpha was 3.45 per cent, p = 0.06; after adding non-linear status, 3.21 per cent, p = 0.03; after adding to the internal budget rules, 2.85 per cent, p = 0.01. The last step is a high level of interpretation, but the internal rules are inherently strategic, institutional, and cannot be used as an independent external factor. 3.45 per cent and their insufficient visibility should be given greater weight when discussing the benefits of external risk divestment.")
    add_text(doc, "SORA here is a risk-free reference. Cash gains are generated only when cash is actually held; the addition of SORA in the return of excess proceeds is an expression of the constant equation and does not mean that the strategy receives an additional “basic gain” without hold.")

    page(doc, "5.6 Detailed and non-linear contributions of extended factors")
    expanded = read_table(105)
    table(doc, "Full attribution of extended models", expanded,
          ['Attribution', 'Beta or coefficient', 'Annual computational contributions', 'Percentage of cumulative contribution', 'HAC t', 'HAC p'], 8.0)
    add_text(doc, "Expansion factors include global equities, regional relative global, national debt, gold, energy, industrial metals, United States dollars, BTC, multi-asset momentum, non-linear stock timing and dynamic budget increases. The annualization of multiple asset movements contributed 2.16 per cent and internal dynamic budget increases of 4.72 per cent, indicating a clear link between returns and trends and the state of rules. The non-linear timing of stocks p = 0.54, and the current model does not provide strong evidence that Alpha could not be claimed to have proven independent choice.")
    add_text(doc, "Factors may be relevant, contributing to the statistical distribution under the given model rather than the real cash flow of assets. BTC has contributed 149.62 percentage points, which is very different from the 13.14 percentage points contributed by the extended return BTC factor, because the latter interprets the gains together with momentum, internal rules, etc.; the two attributions answer different questions. Because the child Beta also changes over time, the static return can only give an average area exposure.")

    page(doc, "5.7 Risk exposures change rather than always fixed Beta")
    figure(doc, 111, 1, "Factor exposure change in rolling regression")
    figure(doc, 111, 2, "Scroll Residue Alpha and Interpretation")
    add_text(doc, "The rolling chart is used to observe whether market exposure and surplus gains change over time and cannot be used as an ex ante predictable signal for a higher section of the curve. The strategic risks arise from proactive changes in the allocation of funds, the possible failure of trend signals, and the combined impact of equities, commodities, BTCs and exchange rates; it should not be classified as a fixed market, Beta, or as a proven pure Alpha product.")
    add_text(doc, "The funds at the end of 31 July 2026 were mainly in equities of about 67.22 per cent, commodities of about 32.78 per cent and no holding stock of BTC, bonds or foreign exchange securities. This cross-section does not change the historical BTC dependence on earnings, nor does it mean that there is no exchange rate risk: overseas equities and commodities are still denominated in foreign currency. The “foreign exchange” category of securities should be understood separately from the currency exposures contained in all overseas assets.")

    page(doc, "5.8 Gross proceeds to real costs to net proceeds Because")
    figure(doc, 101, 1, "A copy of the cost of real-life differentials. Figure")
    table(doc, "Impact of the cost breakdown of realistic copies on cumulative gains", read_table(101),
          ['Item', 'Percentage of cumulative contribution', 'End-of-cycle wealth contribution SGD'], 9.0)
    add_text(doc, "The gross proceeds of the assets, which are copies of the actual cost, contribute 408.19 percentage points, commissions and transaction costs - 7.78, swaps - 19.91, price differential points - 10.87, market shocks - 1.20, withholding taxes - 4.31 and, ultimately, net gains - 364.12 per cent. Exchanges are the largest wear and tear projects under this group scenario, demonstrating the real value of real implementation of the design of the Chinese currency account and net fund management.")
    add_text(doc, "The gross asset contribution of the formal fixed-cost path of 451.27 percentage points cannot be reduced directly to the actual copy of 408.19 as an additional cost. The net cost changes and the size of the follow-up funds, and the cumulative contribution also includes a compound link, so that the gross contribution of the different copies will change per se. The difference between the net gain of 441.52 per cent and 364.12 per cent should be used to compare the total cost impact; the sub-interpretation is done within the actual copy.")

    page(doc, "5.9 Final research judgement and product selection")
    add_text(doc, "We're finally going to have to go for it. It identifies candidate assets with a kinetic Top10, expresses trends in categories and categories in a fully dynamic two-tiered budget, then constructs cores with a joint EWMA risk budget; eligible breadth, fluctuations, portfolio withdrawals and VIX decisions on 0-30 per cent of offensive capital; and implements buffer reductions in fine-tangles. Compared to the initial static configuration, the final configuration is accompanied by asset selection, state response and top-line participation.")
    add_text(doc, "The selection was based on three sets of evidence. First, 441.52 per cent of the cumulative revenues and 1.32 Sharps of the complete zone of research are ahead of the 12 official strategies, with the annual change of hands lower than the fixed offensive. Secondly, the benefits of the epidemic and the double-indebted debt of 2022 were significantly superior to the withdrawal; the crisis-weighted Bootstream and the annual ES relative fixed offensive improved. Third, the cumulative benefits of 364.12 per cent after actual costs and 1.20 per cent for Sharp indicate that the advantages under this scenario remain valid.")
    add_text(doc, "At the same time, we maintain a robust core position. The full range of the vibrator's defence is maximum, at 14.98 per cent, above the robust core, at 9.75 per cent; during the validation period, Sharp was less than a number of simplified options, and the rebound window was also likely to lag behind. If clients prioritize smaller retreats, they should consider a robust core; if they value revenue participation and accept deeper fluctuations, dynamic resistance is more consistent with current objectives. By contrast, the increase in the proportion of attacks does not in itself guarantee a better Sharp.")
    add_text(doc, "The strategy performance is derived mainly from multi-asset trends, dynamic risk taking and a small number of long-term strong trend assets, and statistical residues are affected by factor definitions and research selection. The most valuable code work to follow is the final strategy of reruning the BTC, the digestion of offensive parameters and enforcement rules, the replacement of externally tradable assets, and the random selection and multiple testing of uniform calibres; these studies should not be described as evidence available at present.")
    add_section_heading(doc, "Recurrence and data sources", 2)
    add_text(doc, "The report forms and graphs were saved from the project Notebook and the study ended on 31 July 2026; the lock-in observation line ended on 1 October. Prices are from Yahoo Finance, SORA from external import data. Codes run sequentially by config, data conduit, risk model, strategy, reporting statistics, robustness and attribution module; repeat using Main.py, with specific parameters and SORA import instructions available in Project README. Report that no changes have been made to the policy parameters or transaction codes for the reconstruction.")
    add_section_heading(doc, "References", 2)
    for line in [
        "Markowitz H. Portfolio Selection. Journal of Finance, 1952.",
        "Qian E. Risk Parity Portfolios. PanAgora Asset Management, 2005.",
        "Bailey D H, López de Prado M. The Deflated Sharpe Ratio, 2014.",
        "Global multi-asset portfolio strategy based on the trend configuration model 15 September 2023. This paper refers to its research organization and does not cite its performance as evidence of this strategy.",
    ]:
        add_text(doc, line, size=9, first_line=False)
