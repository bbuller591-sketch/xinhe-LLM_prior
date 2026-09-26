#!/usr/bin/env python3

# --- package path bootstrap -------------------------------------------------
import os as _os
from pathlib import Path as _Path


def _repro_root(start=None):
    """Package root, located by the '.repro_root' marker (or REPRO_ROOT env)."""
    here = _Path(start or __file__).resolve()
    for cand in [here, *here.parents]:
        if (cand / ".repro_root").exists():
            return cand
    return _Path(_os.environ.get("REPRO_ROOT", _Path.cwd())).resolve()


REPRO_ROOT = _repro_root()
# ---------------------------------------------------------------------------
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(str(REPRO_ROOT / 'shared_engine/multi_dataset_expansion'))
TASK='BREAST_GSE25055_GSE25065'
IN=ROOT/'23_selective_SHUFFLE_CONTROL/PRIMARY_1000'
OUT=ROOT/'23_selective_SHUFFLE_CONTROL/RESULTS'
OUT.mkdir(parents=True,exist_ok=True)

cells=pd.read_parquet(IN/'selective_SHUFFLE_REPLICATE_CELL_RESULTS.parquet')
reps=pd.read_csv(IN/'selective_SHUFFLE_REPLICATE_SUMMARY.csv')
curves=pd.read_parquet(IN/'selective_SHUFFLE_REPLICATE_ETA_CURVES.parquet')
if cells.replicate.nunique()!=1000 or reps.replicate.nunique()!=1000:
    raise RuntimeError(f'NEED_1000_REPS got cells={cells.replicate.nunique()} summaries={reps.replicate.nunique()}')

ref=pd.read_csv(ROOT/'21_SEALED_VALIDATION/SEALED_VALIDATION_PRIMARY_RESULTS.csv')
ref=ref[(ref['task']==TASK)&(ref['analysis']=='PRIMARY')].copy()
selective=ref[ref['method']=='selective'].copy()
assert len(selective)==9
selective['actual_delta']=selective['delta_auroc_vs_reference'].astype(float)
actual_lam=pd.read_csv(ROOT/'13_FINAL_DEVELOPMENT_TUNING'/TASK/'FINAL_SELECTED_ETA.csv')

rows=[]
for r in selective.itertuples():
    n=cells[(cells.selector==r.selector)&(cells.k==r.k)].delta_auroc_vs_reference.to_numpy(float)
    actual=float(r.actual_delta)
    pdiag=(1+int(np.sum(n>=actual)))/(len(n)+1)
    rows.append({
      'selector':r.selector,'k':int(r.k),'actual_selective_auroc':float(r.auroc),'reference_auroc':float(r.reference_auroc),
      'actual_delta_auroc':actual,'null_mean_delta_auroc':float(n.mean()),'null_sd_delta_auroc':float(n.std(ddof=1)),
      'null_q025':float(np.quantile(n,.025)),'null_q25':float(np.quantile(n,.25)),
      'null_median':float(np.quantile(n,.5)),'null_q75':float(np.quantile(n,.75)),
      'null_q975':float(np.quantile(n,.975)),
      'actual_minus_null_mean':float(actual-n.mean()),
      'actual_percentile_strict':float(np.mean(n<actual)),
      'empirical_p_ge_actual_posthoc':float(pdiag),
      'actual_lam':float(actual_lam[(actual_lam.selector==r.selector)&(actual_lam.k==r.k)].iloc[0].chosen_lam)
    })
cell_summary=pd.DataFrame(rows)
cell_summary.to_csv(OUT/'selective_LLM_VS_SHUFFLE_CELL_SUMMARY.csv',index=False)

actual_mean=float(selective.actual_delta.mean())
nmean=reps.mean_delta_auroc.to_numpy(float)
aggregate={
 'n_shuffle_replicates':1000,
 'shuffle_type':'PAIRED_SOURCE_BUNDLE_PERMUTATION',
 'actual_selective_mean_delta_auroc':actual_mean,
 'null_mean_of_mean_delta_auroc':float(nmean.mean()),
 'null_sd_of_mean_delta_auroc':float(nmean.std(ddof=1)),
 'null_q025':float(np.quantile(nmean,.025)),
 'null_median':float(np.quantile(nmean,.5)),
 'null_q975':float(np.quantile(nmean,.975)),
 'actual_minus_null_mean':float(actual_mean-nmean.mean()),
 'actual_percentile_strict':float(np.mean(nmean<actual_mean)),
 'empirical_p_ge_actual_posthoc':float((1+np.sum(nmean>=actual_mean))/(len(nmean)+1)),
 'actual_positive_cells':int((selective.actual_delta>0).sum()),
 'actual_zero_cells':int(np.isclose(selective.actual_delta,0,atol=1e-12).sum()),
 'actual_negative_cells':int((selective.actual_delta<0).sum()),
 'null_mean_positive_cells':float(reps.positive_cells.mean()),
 'null_mean_zero_cells':float(reps.zero_cells.mean()),
 'null_mean_negative_cells':float(reps.negative_cells.mean()),
 'interpretation_guardrail':'Post-hoc randomization diagnostic; original sealed results were observed before this shuffle control was designed.'
}
(OUT/'selective_LLM_VS_SHUFFLE_AGGREGATE_SUMMARY.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')

# selector-level aggregates
sel_rows=[]
for sel,gactual in selective.groupby('selector'):
    aval=float(gactual.actual_delta.mean())
    q=cells[cells.selector==sel].groupby('replicate').delta_auroc_vs_reference.mean().to_numpy(float)
    sel_rows.append({'selector':sel,'actual_mean_delta':aval,'null_mean_delta':float(q.mean()),
                     'null_sd':float(q.std(ddof=1)),'null_q025':float(np.quantile(q,.025)),
                     'null_q975':float(np.quantile(q,.975)),'actual_minus_null_mean':float(aval-q.mean()),
                     'actual_percentile_strict':float(np.mean(q<aval)),
                     'empirical_p_ge_actual_posthoc':float((1+np.sum(q>=aval))/(len(q)+1))})
sel=pd.DataFrame(sel_rows); sel.to_csv(OUT/'selective_LLM_VS_SHUFFLE_SELECTOR_SUMMARY.csv',index=False)

# lam frequencies
ef=(cells.groupby(['selector','k','chosen_lam'],as_index=False).size()
    .rename(columns={'size':'count'}))
ef['frequency']=ef['count']/1000
ef=ef.merge(actual_lam[['selector','k','chosen_lam']].rename(columns={'chosen_lam':'actual_llm_lam'}),
            on=['selector','k'],how='left')
ef.to_csv(OUT/'selective_SHUFFLE_ETA_FREQUENCY.csv',index=False)

# development best CV vs actual LLM development best CV
dev_rows=[]
for r in actual_lam.itertuples():
    q=cells[(cells.selector==r.selector)&(cells.k==r.k)].dev_cv_best_mean_auroc.to_numpy(float)
    act=float(r.cv_mean_auroc)
    dev_rows.append({'selector':r.selector,'k':int(r.k),'actual_llm_cv_best_mean_auroc':act,
                     'null_mean_cv_best':float(q.mean()),'null_sd_cv_best':float(q.std(ddof=1)),
                     'actual_minus_null_mean':float(act-q.mean()),
                     'actual_percentile_strict':float(np.mean(q<act)),
                     'empirical_p_ge_actual_posthoc':float((1+np.sum(q>=act))/(len(q)+1))})
pd.DataFrame(dev_rows).to_csv(OUT/'selective_LLM_VS_SHUFFLE_DEVELOPMENT_CV_SUMMARY.csv',index=False)

# Plots
FIG=OUT/'FIGURES'; FIG.mkdir(exist_ok=True)

fig,ax=plt.subplots(figsize=(8,5))
ax.hist(nmean,bins=35,alpha=.8)
ax.axvline(actual_mean,linewidth=2,label=f'Actual LLM selective = {actual_mean:+.4f}')
ax.axvline(nmean.mean(),linestyle='--',label=f'Shuffle mean = {nmean.mean():+.4f}')
ax.set_xlabel('Mean sealed Delta AUROC over 9 selector x k cells')
ax.set_ylabel('Shuffle replicates')
ax.set_title('selective semantic-shuffle null: aggregate external effect')
ax.legend(); fig.tight_layout()
fig.savefig(FIG/'aggregate_null_histogram.png',dpi=180); plt.close(fig)

# cell actual vs null 95%
cs=cell_summary.copy()
labels=[f"{r.selector}\nk={r.k}" for r in cs.itertuples()]
x=np.arange(len(cs))
fig,ax=plt.subplots(figsize=(13,6))
mid=cs.null_mean_delta_auroc.to_numpy()
lo=mid-cs.null_q025.to_numpy(); hi=cs.null_q975.to_numpy()-mid
ax.errorbar(x,mid,yerr=np.vstack([lo,hi]),fmt='o',capsize=4,label='Shuffle null mean and 95% interval')
ax.scatter(x,cs.actual_delta_auroc,s=55,label='Actual LLM selective')
ax.axhline(0,linewidth=.8)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=45,ha='right')
ax.set_ylabel('Sealed Delta AUROC vs reference')
ax.set_title('Actual selective versus 1000 semantic shuffles by selector and k')
ax.legend(); fig.tight_layout()
fig.savefig(FIG/'cell_actual_vs_shuffle_95ci.png',dpi=180); plt.close(fig)

# percentile / p
fig,ax=plt.subplots(figsize=(13,5))
ax.bar(x,cs.actual_percentile_strict)
ax.axhline(.95,linestyle='--')
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=45,ha='right')
ax.set_ylim(0,1)
ax.set_ylabel('Fraction of shuffled runs below actual LLM selective')
ax.set_title('Actual selective percentile within semantic-shuffle null')
fig.tight_layout(); fig.savefig(FIG/'cell_actual_percentile.png',dpi=180); plt.close(fig)

status={'status':'PASS_selective_SEMANTIC_SHUFFLE_1000','n_replicates':1000,
        'aggregate':aggregate,'cell_summary_sha256':hashlib.sha256((OUT/'selective_LLM_VS_SHUFFLE_CELL_SUMMARY.csv').read_bytes()).hexdigest(),
        'replicate_cells_sha256':hashlib.sha256((IN/'selective_SHUFFLE_REPLICATE_CELL_RESULTS.parquet').read_bytes()).hexdigest()}
(OUT/'selective_SHUFFLE_CONTROL_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(aggregate,indent=2))
print('\nCELL SUMMARY\n',cell_summary.to_string(index=False))
print('\nSELECTOR SUMMARY\n',sel.to_string(index=False))
