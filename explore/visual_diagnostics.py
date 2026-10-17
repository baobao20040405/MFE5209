"""Supplementary report figures built from the formal backtest ledgers."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def build_visual_diagnostics(context, output_dir=None, show=True):
    """Build English figures and export their source tables without changing strategy logic."""
    c = context
    output = Path(output_dir or Path(c['PROJECT_ROOT']) / 'report_work' / 'visual_diagnostics')
    output.mkdir(parents=True, exist_ok=True)
    results, names = c['results'], c['STRATEGY_NAMES']
    final = names['dynamic_attack']
    core = [names[k] for k in ['fully_dynamic', 'fixed_attack', 'dynamic_attack']]
    colors, labels = c['STRATEGY_COLORS'], c['STRATEGY_LABELS']
    assets, end = c['assets'], c['RESEARCH_END']
    returns = c['returns'].loc[:end]
    palette = c['ASSET_CLASS_COLORS']
    classes = ['Equities', 'Bonds', 'Commodities', 'FX', 'Crypto']
    common = pd.concat({n: results[n]['returns'].loc[:end] for n in results}, axis=1).dropna()
    if common.index.max() >= c['HOLDOUT_START']:
        raise AssertionError('Supplementary research figures must not include the holdout period.')
    files, tables, interpretations = {}, {}, {}

    def save_table(key, frame):
        tables[key] = frame
        frame.to_csv(output / f'{key}.csv', encoding='utf-8-sig', index=True)

    def finish(key, fig, explanation):
        for ax in fig.axes:
            if ax.get_label() != '<colorbar>':
                c['style_axis'](ax)
        fig.savefig(output / f'{key}.png', dpi=180, bbox_inches='tight', facecolor='white')
        files[key] = str(output / f'{key}.png')
        interpretations[key] = explanation
        if show:
            plt.show()
        plt.close(fig)

    def legend(ax, **kwargs):
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, ncol=min(len(l), 4), loc='upper center',
                  bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=9, **kwargs)

    def flow(key, title, items, footer):
        fig, ax = plt.subplots(figsize=(12, 2.7 if len(items)<=4 else 4.4))
        ax.set_axis_off()
        for i, (name, text) in enumerate(items):
            row, col = divmod(i, 4)
            x, y = .02 + col * .247, (.50 if len(items)<=4 else .62 - row * .46)
            ax.text(x, y, f'{i+1:02d}  {name}\n\n{text}', transform=ax.transAxes,
                    va='center', fontsize=10, linespacing=1.5,
                    color=colors[final] if i == len(items)-1 else '#333333',
                    bbox=dict(boxstyle='round,pad=0.65', fc='#FAFAFA', ec='#D9D9D9'))
            if col < 3 and i < len(items)-1:
                ax.annotate('', xy=(x+.225,y), xytext=(x+.193,y), xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color=colors[final]))
        ax.set_title(title, pad=20)
        fig.text(.03,.015,footer,fontsize=9,color='#666666')
        finish(key, fig, footer)

    flow('research_path', 'Research Path Across the 12 Retained Strategies', [
        ('Allocation Baselines', '18-asset equal weight\nMonthly all-asset ERC'),
        ('Risk Models', 'Same Top 10 signal\nFive covariance specifications'),
        ('Budget Design', 'Single-level and static hierarchy\nAdaptive two-level hierarchy'),
        ('Offensive Sleeve', 'Fixed 30% versus dynamic 0%-30%\nFinal selection: GMI-ADG'),
    ], 'Arrows show the progression of research questions; they do not imply that every added module improves every metric. Downside and GARCH models are parallel tests.')
    flow('construction_steps', 'GMI-ADG: Seven Steps from Signal to Execution', [
        ('Momentum Screen', '12-1 month momentum\nSelect the Top 10 of 18 assets'),
        ('Class Budgets', 'Class rank and trend breadth\nSet class-level risk shares'),
        ('Security Budgets', 'Within-class rank and trend\nSet security-level risk shares'),
        ('Core Optimizer', 'EWMA covariance, lambda=0.94\nJoint risk-budget solution'),
        ('Dynamic Offense', 'Eligibility and diversification\nBreadth, drawdown, and VIX caps'),
        ('Trading Buffer', 'Core plus 0%-30% offense\nPartial adjustment and no-trade band'),
        ('Next-Day Execution', 'Trade at the next close\nDeduct costs from executed notional'),
    ], 'Risk budgets are not capital weights. A 0% offensive target returns the strategy to its core portfolio; it does not liquidate the entire portfolio.')

    metrics = pd.DataFrame({n: c['compute_metrics'](results[n], end=end) for n in common}).T
    save_table('strategy_tradeoff', metrics)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    # Discrete size bins avoid making a tiny turnover difference look material.
    bins = pd.cut(metrics['Annual One-Way Turnover'], [-np.inf,1,3,np.inf], labels=[90,170,270]).astype(int)
    offsets = {names['dynamic_attack']:(12,12), names['fixed_attack']:(10,-19),
               names['fully_dynamic']:(10,10), names['equal_weight']:(10,-15),
               names['monthly_erc']:(10,10), names['ewma_erc']:(-100,25),
               names['static_hierarchical']:(-120,-12), names['single_dynamic']:(-100,48),
               names['garch_erc']:(-85,-43), names['classic_erc']:(44,6),
               names['downside_erc']:(40,-32)}
    strategy_handles = []
    for i, (n, row) in enumerate(metrics.iterrows()):
        x, y = abs(row['Maximum Drawdown']), row['Annualized Return']
        ax.scatter(x,y,s=bins[n],c=colors[n],edgecolors='white',linewidth=1,zorder=3)
        strategy_handles.append(Line2D([], [], linestyle='None', marker='o', markersize=7,
                                       markerfacecolor=colors[n], markeredgecolor='white',
                                       label=labels[n]))
        dx, dy = offsets.get(n, ((-85 if i%2 else 12), (i%3-1)*16))
        ax.annotate(labels[n],(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=9,
                    arrowprops=dict(arrowstyle='-',color=colors[n],lw=.6))
    ax.set(xlabel='Maximum Drawdown Magnitude (lower is better)', ylabel='Annualized Net Return',
           title='Return-Drawdown Trade-Off Across 12 Strategies')
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=2))
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
    size_handles = [plt.scatter([], [], s=size, color='#A7ADB4', label=label)
                    for size, label in [(90,'Turnover <= 100%'),(170,'100% < Turnover <= 300%'),(270,'Turnover > 300%')]]
    strategy_legend = ax.legend(handles=strategy_handles, ncol=4, loc='upper center',
                                bbox_to_anchor=(.5, -.17), frameon=False, fontsize=8.5,
                                columnspacing=1.2, handletextpad=.4)
    ax.add_artist(strategy_legend)
    ax.legend(handles=size_handles, ncol=3, loc='upper center',
              bbox_to_anchor=(.5, -.34), frameon=False, fontsize=8.5)
    ax.margins(x=.24,y=.2); fig.subplots_adjust(bottom=.31)
    finish('strategy_tradeoff',fig,'Same research period and fixed 0.05% one-way cost. Bubble size represents three annual one-way turnover bands, not portfolio size or statistical confidence.')

    yearly = common[core].groupby(common.index.year).apply(lambda x:(1+x).prod()-1).T
    yearly.columns = [f'{y}' + ('*' if y in [common.index[0].year,common.index[-1].year] else '') for y in yearly.columns]
    save_table('annual_core_returns',yearly)
    fig,ax=plt.subplots(figsize=(12,3.8))
    cmap=LinearSegmentedColormap.from_list('approved',['#2b549c','#f0f0f0','#b82626'])
    limit=max(.01,np.nanmax(np.abs(yearly.values)))
    im=ax.imshow(yearly.values,cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
    ax.set_xticks(range(len(yearly.columns)),yearly.columns,fontsize=10)
    ax.set_yticks(range(3),[labels[n] for n in core],fontsize=10)
    for i in range(3):
        for j in range(len(yearly.columns)):
            v=yearly.iloc[i,j]
            ax.text(j,i,f'{v:.2%}',ha='center',va='center',fontsize=10,color='white' if abs(v)>.65*limit else '#333333')
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=2)); cb.ax.tick_params(labelsize=10)
    ax.set_title('Annual Net Returns of the Three Core Strategies (* Partial Year)')
    fig.tight_layout()
    finish('annual_core_returns',fig,'Calendar-year net returns, not annualized returns. 2015 and 2026 are partial years.')

    r=results[final]
    triggers=r['triggers'].copy().sort_values('Execution Date')
    triggers.index=pd.DatetimeIndex(triggers['Execution Date'])
    daily=triggers[['Offensive Allocation','Qualified Trend Breadth','Portfolio Drawdown','VIX Crisis']].reindex(common.index).ffill()
    # Before the first execution the strategy is cash, not an observed 0% decision.
    save_table('attack_state_daily',daily)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[2,1.3,1.3]})
    for n in core:
        nav=(1+common[n]).cumprod()
        axes[0].plot(nav.index,nav,color=colors[n],lw=1.5,label=labels[n])
    axes[0].set_ylabel('NAV (Start = 1)'); axes[0].set_title('Dynamic Offensive Target and Market-State Alignment')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1].step(daily.index,daily['Offensive Allocation'],where='post',color=colors[final],label='Executed offensive target')
    axes[1].plot(daily.index,daily['Qualified Trend Breadth'],color=palette['FX'],lw=1,label='Qualified trend breadth at execution')
    axes[1].set_ylabel('Share'); axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
    state=r['state_history'].reindex(common.index)
    axes[2].fill_between(state.index,0,state['Portfolio Drawdown'],color=colors[final],alpha=.20,linewidth=0,label='Portfolio Drawdown')
    axes[2].fill_between(state.index,0,1,where=state['VIX Crisis'].fillna(False).astype(bool),
                         transform=axes[2].get_xaxis_transform(),color='#A7ADB4',alpha=.3,linewidth=0,label='VIX crisis state')
    axes[2].set_ylabel('Drawdown'); axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
    handles, legend_labels = [], []
    for axis in axes:
        h, l = axis.get_legend_handles_labels()
        handles.extend(h); legend_labels.extend(l)
    fig.legend(handles, legend_labels, ncol=4, loc='lower center',
               bbox_to_anchor=(.5, .005), frameon=False, fontsize=9)
    fig.subplots_adjust(hspace=.28,bottom=.15)
    finish('attack_state',fig,'The offensive allocation is the model target carried by each executed order and forward-filled thereafter. It is not a separately drifted offensive sub-account. Qualified breadth is also recorded at execution rather than recalculated daily.')

    def aggregate(series):
        return series.groupby(assets['Asset Class']).sum().reindex(classes,fill_value=0)

    # Same date and covariance: capital equality does not imply risk equality.
    baseline_date=common.index[-1]
    baseline_cov=c['ewma_covariance'](returns.loc[:baseline_date].tail(c['LOOKBACK']))
    for key in ['equal_weight','monthly_erc']:
        name=names[key]
        w=results[name]['end_of_day_weights'].loc[baseline_date].reindex(returns.columns,fill_value=0)
        risk=w*(baseline_cov@w)/float(w@baseline_cov@w)
        frame=pd.DataFrame({'Capital Weight':aggregate(w),'Risk Contribution':aggregate(risk)})
        save_table(f'capital_risk_{key}',frame)
        fig,ax=plt.subplots(figsize=(10,5.0))
        y=np.arange(len(classes)); height=.32
        capital=frame['Capital Weight'].to_numpy(); risk=frame['Risk Contribution'].to_numpy()
        for i,category in enumerate(classes):
            ax.barh(y[i]-height/2,capital[i],height=height,color=palette[category],alpha=.95)
            ax.barh(y[i]+height/2,risk[i],height=height,color=palette[category],alpha=.34,
                    edgecolor=palette[category],linewidth=1.1,hatch='///')
            ax.text(capital[i] + .008, y[i]-height/2, f'{capital[i]:.2%}', va='center', fontsize=9)
            ax.text(risk[i] + (.008 if risk[i]>=0 else -.008), y[i]+height/2, f'{risk[i]:.2%}',
                    va='center', ha='left' if risk[i]>=0 else 'right', fontsize=9)
        ax.set_yticks(y,classes); ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=2))
        ax.axvline(0,color='#A7ADB4',lw=.7); ax.margins(x=.18,y=.12)
        ax.set(title=f'{labels[name]}: Capital and Risk Distribution ({baseline_date:%Y-%m-%d})',xlabel='Portfolio Share')
        ax.legend(handles=[Patch(facecolor='#6B7280',label='Capital Weight'),
                           Patch(facecolor='#D7DADE',edgecolor='#6B7280',hatch='///',label='Risk Contribution')],
                  ncol=2,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False,fontsize=9)
        fig.subplots_adjust(bottom=.20)
        finish(f'capital_risk_{key}',fig,'End-of-day holdings are compared under one common EWMA covariance matrix. GMA-CRP was originally optimized with a sample covariance matrix, so it need not show equal risk under this EWMA remeasurement. Neither benchmark targets equal asset-class risk. Cash is retained in the source holdings but contributes no risky-asset variance.')

    snapshots=[]
    for requested in ['2017-12-01','2020-03-16','2022-10-03']:
        after=triggers[triggers.index>=pd.Timestamp(requested)]
        if after.empty:
            continue
        trade=after.iloc[0]; execution=after.index[0]; signal=pd.Timestamp(trade['Signal Date'])
        h=returns.loc[:signal].tail(c['LOOKBACK'])
        if len(h)!=c['LOOKBACK']: raise AssertionError('Insufficient warm-up history for the risk snapshot.')
        selected,score=c['select_top10'](h)
        budget=c['fully_dynamic_budget'](h,selected,score)
        cov=c['ewma_covariance'](h)
        wcore=c['strict_risk_budget'](cov.loc[selected,selected],budget).reindex(h.columns,fill_value=0)
        wactual=r['end_of_day_weights'].loc[execution].reindex(h.columns,fill_value=0)
        def rc(w):
            variance=float(w@cov@w)
            if variance<=0: raise AssertionError('Risk-snapshot variance must be positive.')
            out=w*(cov@w)/variance
            np.testing.assert_allclose(out.sum(),1,atol=1e-8)
            return aggregate(out)
        frame=pd.DataFrame({'Core Target Risk Budget':aggregate(budget), 'Core Risk Contribution':rc(wcore),
                            'Final Risk Contribution':rc(wactual),'Final Capital Weight':aggregate(wactual)})
        frame.index.name='Asset Class'; snapshots.append((execution,frame))
        save_table(f'risk_snapshot_{execution:%Y%m%d}',frame)
        fig,ax=plt.subplots(figsize=(11,5.1))
        display_order=['Core Target Risk Budget','Core Risk Contribution','Final Capital Weight','Final Risk Contribution']
        display_labels=['Core Target Budget','Core Realized Risk','Final Capital Weight','Final Realized Risk']
        values=frame[display_order].T
        for row_i,(column,row) in enumerate(values.iterrows()):
            positive_left=0.0; negative_left=0.0
            for category,value in row.items():
                left=positive_left if value>=0 else negative_left
                ax.barh(row_i,value,left=left,height=.60,color=palette[category],
                        edgecolor='white',linewidth=.8,label=category if row_i==0 else None)
                if abs(value)>=.055:
                    ax.text(left+value/2,row_i,f'{value:.2%}',ha='center',va='center',
                            fontsize=8.5,color='white' if category in ['Equities','Crypto'] else '#333333')
                if value>=0: positive_left += value
                else: negative_left += value
        ax.set_yticks(range(4),display_labels); ax.invert_yaxis()
        ax.axvline(0,color='#A7ADB4',lw=.7)
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=2))
        lower=min(-.03,float(values.where(values<0,0).sum(axis=1).min())-.03)
        upper=max(1.03,float(values.where(values>0,0).sum(axis=1).max())+.03)
        ax.set_xlim(lower,upper)
        ax.set(xlabel='Asset-Class Composition',title=f'Risk Budget to Actual Holdings | Signal {signal:%Y-%m-%d} | Trade {execution:%Y-%m-%d}')
        ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,fontsize=9)
        fig.subplots_adjust(left=.20,bottom=.22)
        finish(f'risk_snapshot_{execution:%Y%m%d}',fig,'Each snapshot uses the first trade after a preselected date. Risk contributions use the signal-date EWMA covariance matrix. The core solution should closely match its target budget; the final portfolio may depart from that budget after the offensive overlay and trading buffer. Negative contributions may arise from hedging effects.')

    # Ex-post attribution by backward-looking benchmark state, not a new timing test.
    base=common[names['equal_weight']]
    trailing=(1+base).rolling(63,min_periods=63).apply(np.prod,raw=True)-1
    regime=pd.Series('Sideways',index=base.index)
    regime[trailing>=.05]='Rising';regime[trailing<=-.05]='Falling';regime[trailing.isna()]=pd.NA
    rows=[]
    for state_name in ['Rising','Sideways','Falling']:
        mask=regime.eq(state_name).fillna(False)
        for n in core:
            x=common.loc[mask,n]
            rows.append({'Status':state_name,'Policy':n,'Observations':len(x),
                         'Conditional Mean Daily Return':x.mean(),'Conditional Daily Volatility':x.std(ddof=1),
                         'Positive-Day Ratio':(x>0).mean()})
    conditional=pd.DataFrame(rows);save_table('market_state_returns',conditional)
    save_table('market_state_labels',pd.DataFrame({'GMA-EW Trailing 63-Day Return':trailing,'Descriptive Regime':regime}))
    fig,ax=plt.subplots(figsize=(10,4.8))
    for n in core:
        sub=conditional[conditional['Policy'].eq(n)]
        ax.plot(range(3),sub['Conditional Mean Daily Return']*10000,marker='o',color=colors[n],label=labels[n])
    ax.axhline(0,color='#A7ADB4',lw=.8)
    ax.set_xticks(range(3),[f'{s}\nN={int(conditional[conditional["Status"].eq(s)].iloc[0]["Observations"])}' for s in ['Rising','Sideways','Falling']])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set(ylabel='Conditional Mean Daily Net Return (bp)',title='Mean Daily Return by Descriptive Market Regime')
    legend(ax);fig.subplots_adjust(bottom=.25)
    finish('market_state_returns',fig,'Regimes use the trailing 63-day GMA-EW return: at least +5% is Rising, at most -5% is Falling, and the remainder is Sideways. This same-day classification is descriptive and ex post; it is not a tradable signal or independent evidence of timing alpha.')

    # Benchmark episodes select common evaluation windows independently of final-strategy performance.
    nav=(1+base).cumprod();values=np.r_[1.,nav.to_numpy()]
    peak=0;episodes=[];active=None
    for pos in range(1,len(values)):
        if values[pos]>=values[peak]-1e-12:
            if active is not None:
                episodes.append((active[0],active[1],pos,active[2],True));active=None
            peak=pos
        else:
            depth=values[pos]/values[peak]-1
            if active is None: active=[peak,pos,depth]
            elif depth<active[2]:active[1:]=[pos,depth]
    if active is not None:episodes.append((active[0],active[1],len(values)-1,active[2],False))
    recovery_rows=[]
    for rank,(peak_pos,trough_pos,end_pos,depth,recovered) in enumerate(sorted(episodes,key=lambda e:e[3])[:3],1):
        window=common.iloc[peak_pos:end_pos][core]
        if window.empty:continue
        paths=np.vstack([np.ones(3),(1+window).cumprod().values])
        save_table(f'recovery_path_{rank}',pd.DataFrame(paths,columns=core).rename_axis('From common starting date'))
        fig,ax=plt.subplots(figsize=(11,4.8))
        for j,n in enumerate(core):
            path=paths[:,j];trough=int(path.argmin());found=np.flatnonzero(path[trough+1:]>=1-1e-12)
            recovered_at=int(trough+1+found[0]) if len(found) else None
            recovery_rows.append({'Window':rank,'Policy':n,'Window Start Day':window.index[0],
                'Window End Day':window.index[-1],'Can not open message':float(path.min()-1),
                'Until the date of the transaction':recovered_at,'Fix by end of window':recovered_at is not None,
                'Benchmark complete repair':recovered})
            label=labels[n]+(f' | Recovered in {recovered_at} trading days' if recovered_at is not None else ' | Not recovered in window')
            ax.plot(range(len(path)),path,color=colors[n],label=label,lw=1.6)
        ax.axhline(1,color='#A7ADB4',lw=.8)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set(xlabel='Trading Days from Common Starting Point',ylabel='NAV (Start = 1)',
            title=f'Drawdown and Recovery Path {rank}: {window.index[0]:%Y-%m-%d} to {window.index[-1]:%Y-%m-%d}')
        legend(ax);fig.subplots_adjust(bottom=.23)
        finish(f'recovery_path_{rank}',fig,'The three non-overlapping windows are selected from the deepest GMA-EW drawdown episodes. Each strategy is rebased to 1 at the common window start. Recovery is the first return to 1 after the strategy\'s trough within that window; unrecovered observations remain missing rather than being recorded as zero days.')
    save_table('recovery_summary',pd.DataFrame(recovery_rows))
    save_table('figure_manifest',pd.DataFrame([{'Figure Name':k,'File':v,'Scope and Interpretation':interpretations[k]} for k,v in files.items()]))
    return {'figures':files,'tables':tables,'interpretations':interpretations,'output_dir':str(output)}


# The Notebook and modular pipeline call the same function, ensuring identical figures.
