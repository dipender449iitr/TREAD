#!/usr/bin/env python3
"""Recalculate numeric MSR manuscript Tables II--VI from archived output CSVs.

Separate archived experiment outputs from this independent, non-model-fitting
reanalysis. This does not run GMMs or downstream learners or check input CSVs.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/archived_experiment'
OUT=ROOT/'verification'
OUT.mkdir(exist_ok=True)
FAMS=('aeeem','jira','promise')
records=[]

def ensure(label, observed, expected, tol=.00051):
    passed=np.isclose(float(observed),float(expected),rtol=0,atol=tol)
    records.append(dict(check=label,observed=float(observed),expected=float(expected),tolerance=tol,pass_=bool(passed)))
    if not passed: raise AssertionError(f'{label}: observed {observed}, expected {expected}')

def csv(name): return pd.read_csv(RES/name)

original_code=ROOT/'code/TREAD_MSR_FINAL_ONE_CLICK.py'
m=json.loads((RES/'run_manifest.json').read_text())
observed_sha=hashlib.sha256(original_code.read_bytes()).hexdigest()
assert observed_sha==m['script_sha256'], 'Source code hash differs from manifest'
assert m['mode']=='full'
assert m['settings']['downstream_seeds']==10
integrity=json.loads((RES/'integrity_report.json').read_text())
assert integrity['completed_ok']==19 and integrity['integrity_errors']==0

inventory=csv('dataset_inventory.csv')
profile=csv('representation_profiles.csv')
diag=csv('target_diagnoses.csv')
rq2=csv('diagnostic_associations.csv')
abl=csv('ablation_associations.csv')
ext=csv('external_diagnostic_associations.csv')
full=csv('downstream_full_metrics.csv')
sens=csv('sensitivity_results.csv')
analysis=csv('project_rq2_analysis.csv')
assert len(inventory)==len(diag)==len(analysis)==19
assert {'R0','R1','U','K0','K1'}.issubset(diag.columns)
assert not diag[['R0','R1','U']].isna().any().any()

# Table II: class-representation structure, component medians, project medians.
table2=[]
expected_t2={'aeeem':(5,5,.546,.441,.037),'jira':(1,0,.169,.365,.076),'promise':(1,5,.754,.435,.054)}
for family in FAMS:
    d=diag[diag.family==family]
    pp=profile[profile.family==family]
    vals=(int((d.K0>1).sum()),int((d.K1>1).sum()),pp.D.median(),pp.A.median(),d.U.median())
    n0,n1,med_d,med_a,med_u=vals
    for label,val,ref,tol in zip(('K0_multi','K1_multi','D_median','A_median','U_median'),vals,expected_t2[family],(0,0,.00051,.00051,.00051)):
        ensure('Table II '+family+' '+label,val,ref,tol)
    table2.append(dict(Family=family.upper(),K0_multi=f'{n0}/{len(d)}',K1_multi=f'{n1}/{len(d)}',D_median=med_d,A_median=med_a,U_median=med_u))
pd.DataFrame(table2).to_csv(OUT/'table_II_rebuilt.csv',index=False)

# Table III correlations - no module rows or seeds treated as independent.
expected_t3={'aeeem':(1.0,-.9,.9),'jira':(-.05,-1/6,11/15),'promise':(.7,.9,1.0)}
table3=[]
for fam in FAMS:
    def rho(diagnostic,outcome):
        d=rq2[(rq2.family==fam)&(rq2.type==('RQ2_secondary' if outcome=='full_logloss' else 'RQ2'))&(rq2.diagnostic==diagnostic)&(rq2.outcome==outcome)]
        assert len(d)==1, (fam,diagnostic,outcome)
        return float(d.iloc[0].spearman_rho)
    r0=rho('R0','proximate_logloss_y0');r1=rho('R1','proximate_logloss_y1');u=rho('U','full_logloss')
    for label,v,ex in zip(('R0','R1','U'),(r0,r1,u),expected_t3[fam]):ensure('Table III '+fam+' '+label,v,ex,.0051)
    table3.append(dict(Family=fam.upper(),n=len(diag[diag.family==fam]),rho_R0=r0,rho_R1=r1,rho_U=u))
pd.DataFrame(table3).to_csv(OUT/'table_III_rebuilt.csv',index=False)

# Table IV: average learner seed results per target, then median across targets.
exp_t4={'aeeem':(.315,.629,.415,.335,.009),'jira':(.345,.517,.350,.284,.019),'promise':(.806,.258,.061,.367,.372)}
table4=[]
metrics=('full_logloss','full_auprc','full_mcc','full_recall','full_fpr')
for fam in FAMS:
    d=full[(full.family==fam)&(full.learner=='GBT')]
    assert d.groupby('target').seed.nunique().eq(10).all()
    summary=d.groupby('target')[list(metrics)].mean().median()
    for name,exp in zip(metrics,exp_t4[fam]):ensure('Table IV '+fam+' '+name,summary[name],exp,.00051)
    table4.append({'Family':fam.upper(),**summary.to_dict()})
pd.DataFrame(table4).to_csv(OUT/'table_IV_rebuilt.csv',index=False)

# Table V: class-agnostic controls versus full-target loss.
exp_t5={'aeeem':(.9,-.1,-.3,-.3),'jira':(11/15,.65,19/60,.1),'promise':(1.,-.9,.3,-.9)}
controls=('U','mean_predictive_entropy','mmd','domain_separability')
table5=[]
for fam in FAMS:
    vals=[]
    for metric in controls:
        d=ext[(ext.family==fam)&(ext.outcome=='full_logloss')&(ext.diagnostic==metric)]
        assert len(d)==1
        vals.append(float(d.iloc[0].spearman_rho))
    for name,x,y in zip(controls,vals,exp_t5[fam]):ensure('Table V '+fam+' '+name,x,y,.0051)
    table5.append({'Family':fam.upper(),**dict(zip(controls,vals))})
pd.DataFrame(table5).to_csv(OUT/'table_V_rebuilt.csv',index=False)

# Table VI: q=5, alpha grid, median U across 5/9/5 target projects.
exp_t6={'aeeem':(.004,.037,.101),'jira':(.010,.076,.158),'promise':(.009,.054,.109)}
table6=[]
for fam in FAMS:
    row={'Family':fam.upper()}
    for alpha,reference in zip((.01,.05,.1),exp_t6[fam]):
        d=sens[(sens.family==fam)&(sens.dimension=='q_alpha')&(sens.q==5)&np.isclose(sens.alpha,alpha)]
        assert len(d)==len(diag[diag.family==fam]),(fam,alpha,len(d))
        med=float(d.U.median())
        ensure('Table VI '+fam+' alpha '+str(alpha),med,reference,.00051)
        row[f'alpha_{alpha:.2f}']=med
    table6.append(row)
pd.DataFrame(table6).to_csv(OUT/'table_VI_rebuilt.csv',index=False)

# Figure 5: complete ablation panels (all five variants, all three families).
expected_abl={
 0:{'TREAD':(1,-.05,.7),'source_frequency':(-.4,-.05,.7),'single_component':(-.4,-.05,.7),'transfer_only':(.5,.05,.7),'ambiguity_only':(.9,-.1,.7)},
 1:{'TREAD':(-.9,-1/6,.9),'source_frequency':(-.9,-1/6,.9),'single_component':(-1,-1/6,.9),'transfer_only':(-.6,-1/6,.9),'ambiguity_only':(-1,-.2,.7)},
}
for cls, rows in expected_abl.items():
    for variant,vals in rows.items():
        for fam,exp in zip(FAMS,vals):
            d=abl[(abl.type==f'RQ3_y{cls}')&(abl.diagnostic==variant)&(abl.family==fam)]
            assert len(d)==1
            ensure(f'Figure 5 class{cls} {fam} {variant}',float(d.iloc[0].spearman_rho),exp,.0051)

# Explicitly retain the known AEEEM feature-count discrepancy as an audit flag.
for family,n in [('aeeem',19),('jira',65),('promise',20)]:
    arr=inventory[inventory.family==family].shared_family_metrics.unique().tolist()
    assert arr==[n], (family,arr)

# Some supplementary checks have target-level outputs but are not rerun here.
report={'paper':'Beyond Source Similarity: Diagnosing Evidence Adequacy for Defect Prediction',
        'full_recorded_targets':19,
        'recorded_script_hash_match':True,
        'verified_checks':len(records),
        'all_checks_pass':all(r['pass_'] for r in records),
        'manuscript_input_discrepancy':{'AEEEM_manuscript_table_I_metrics':61,'AEEEM_archived_run_features':19},
        'source_data_present':False,
        'scope':'Manuscript numerics rechecked from archived CSVs, not rerun from original benchmark data.'}
(OUT/'numerical_verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
pd.DataFrame(records).to_csv(OUT/'numerical_verification_checks.csv',index=False)
print('PASS:',len(records),'numeric checks of manuscript Tables II-VI and RQ3 ablations')
print('Saved rebuilt tables and report under verification/')
print('OPEN PROVENANCE CHECK: AEEEM features in run = 19; manuscript Table I = 61')
