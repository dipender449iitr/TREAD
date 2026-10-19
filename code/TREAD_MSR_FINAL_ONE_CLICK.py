#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TREAD MSR: ONE-FILE, ONE-CLICK research pipeline.

USAGE FOR NORMAL RUN:
  1. Keep this .py file next to "Full Dataset-MSR", containing project CSV
     files within available AEEEM, JIRA, and PROMISE subfolders. The program
     discovers the projects and usable metric columns from the files, rather
     than enforcing workbook project counts or published feature dimensions.
  2. Double-click this file (if Python is associated with .py files), OR run
     `python TREAD_MSR_ONE_CLICK.py` from any working directory.
  3. All RQ1/RQ2/RQ3 tables, diagnostics, prediction traces, and figures are
     created automatically in TREAD_MSR_RESULTS/ next to this script.

No additional PROJECT Python modules, configuration files, plotting scripts,
unit-test scripts, CLI arguments, or requirements.txt are needed. Python 3.9+
and the project CSV files remain necessary. Missing scientific Python packages
are installed automatically with pip if internet/package access is available.

The default is the FULL manuscript protocol. This will be slow on real data.
The optional CLI `--mode smoke` is FOR FUNCTIONAL TESTS ONLY.

TREAD DIAGNOSES, it does not modify/weight the classifier's training data.
Unlabeled target metrics are used only in build_diagnosis(), with label data
supplied only after diagnosis for independent downstream evaluation.
"""
from __future__ import annotations

# ---------------------------- EDITABLE OPTIONS ------------------------------
# For ordinary use leave these unchanged: just double-click this one .py file.
FULL_DATASET_FOLDER = "Full Dataset-MSR"  # may be changed to absolute path
RESULTS_FOLDER = "TREAD_MSR_FINAL_RESULTS"       # auto-created next to this file
FULL_DOWNSTREAM_SEEDS = 10
GENERATE_PUBLICATION_FIGURES = True
RUN_BUILT_IN_TESTS = True
INSTALL_MISSING_PACKAGES = True
AUTO_RESUME_SAME_RUN = True
# -----------------------------------------------------------------------------

import importlib.util
import os
import subprocess
import sys
import time


def _ensure_packages():
    """Only third-party scientific runtime packages; no companion .py scripts."""
    packages = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
                "sklearn": "scikit-learn", "matplotlib": "matplotlib"}
    missing = [dist for module, dist in packages.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return
    if not INSTALL_MISSING_PACKAGES:
        raise RuntimeError("Missing Python libraries: " + ", ".join(missing))
    print("Installing missing scientific Python libraries:", ", ".join(missing), flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing]
    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("Automatic package installation failed. Python and internet/pip "
                           "access are needed for the first run. Install manually with: "
                           + " ".join(cmd)) from exc


_ensure_packages()

import argparse
import hashlib
import json
import logging
import math
import platform
import re
import sys
import traceback
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logsumexp
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold

LOG = logging.getLogger('tread')

# Only the defect-label definitions are fixed. ALL PROJECTS, COUNTS, AND
# FEATURE DIMENSIONS are discovered from the real CSV files at runtime.
BENCHMARKS = {'aeeem': {'label':'bugs'},
              'jira': {'label':'RealBug'},
              'promise': {'label':'bug'}}

def canon(s): return re.sub('[^a-z0-9]','',str(s).lower())

class Inadmissible(RuntimeError):
    """An explicitly reported, unavailable diagnosis (never converted to zero)."""

@dataclass(frozen=True)
class Settings:
    seed:int=2027
    k_max:int=5
    component_min_floor:int=20
    bic_tolerance:float=10.0
    gmm_n_init:int=10
    gmm_reg:float=1e-6
    lr_c:float=1.0
    rf_trees:int=300
    q:int=5
    alpha:float=.05
    lam:float=.5
    epsilon:float=1e-8
    probability_clip:float=1e-6
    downstream_seeds:int=10
    # Analysis settings are fixed before observing target labels.
    lambda_grid:tuple=(0.,.25,.5,.75,1.)
    q_grid:tuple=(3,5,10)
    alpha_grid:tuple=(.01,.05,.10)
    gmm_sensitivity_seeds:tuple=(2028,2029,2030)
    bootstrap_replicates:int=2000
    mmd_sample_max:int=600
    gmm_selection:str="stability"  # comparison mode bic_first is supplementary only
    min_primary_class_n:int=10

@dataclass
class Project:
    family:str
    name:str
    X:np.ndarray
    y:np.ndarray
    features:list
    path:str
    digest:str

@dataclass
class SourceTarget:
    family:str
    target:str
    Xs_raw:np.ndarray
    ys:np.ndarray
    env:np.ndarray
    source_names:list
    Xt_raw:np.ndarray
    feature_names:list

@dataclass
class Transformation:
    median:np.ndarray
    keep:np.ndarray
    center:np.ndarray
    scale:np.ndarray
    def transform(self, X):
        data=np.asarray(X,dtype=float).copy()
        invalid=~np.isfinite(data)
        if invalid.any(): data[invalid]=np.broadcast_to(self.median,data.shape)[invalid]
        return (data[:,self.keep]-self.center)/self.scale


def fit_transformer(X):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or not len(X): raise Inadmissible('no source observations')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(np.where(np.isfinite(X),X,np.nan),axis=0)
    # Features containing only missing source values have no estimable median.
    # Exclude them from the source-fitted feature space; never consult target.
    nonempty=np.isfinite(median)
    if not np.all(nonempty):
        LOG.warning('Exclude %d all-missing source features before imputation',
                    int(np.sum(~nonempty)))
    median=np.where(nonempty,median,0.0)
    imputed=X.copy()
    bad=~np.isfinite(imputed)
    if bad.any(): imputed[bad]=np.broadcast_to(median,imputed.shape)[bad]
    keep=nonempty & (np.ptp(imputed,axis=0)>0)
    if not keep.any(): raise Inadmissible('all source metrics are constant')
    retained=imputed[:,keep]
    center=np.median(retained,axis=0)
    iqr=np.quantile(retained,.75,axis=0)-np.quantile(retained,.25,axis=0)
    scale=np.where(iqr>0,iqr,1.0)
    return Transformation(median,keep,center,scale)


def read_csv_flexible(path, label):
    """Read genuine CSVs with delimiter/header validation, without row loss.

    A parse is accepted only when it contains exactly one recognized label,
    so an apparently successful but incorrectly delimited read cannot pass.
    """
    attempts=[]
    for sep in (',',';',r'\s+', '\t'):
        for encoding in ('utf-8-sig','latin-1'):
            try:
                df=pd.read_csv(path,sep=sep,engine='python',encoding=encoding,
                               na_values=['?','NA','N/A','NaN','nan','null','NULL',''],
                               keep_default_na=True,on_bad_lines='error')
                df.columns=[str(c).strip().strip('\ufeff') for c in df.columns]
                found=[c for c in df.columns if canon(c)==canon(label)]
                if len(found)==1 and len(df)>0 and len(df.columns)>=2:
                    LOG.info('Parsed %s using %s and %s: %d rows, %d raw columns',
                             path.name,repr(sep),encoding,len(df),len(df.columns))
                    return df,found[0],sep
                attempts.append(f'{repr(sep)}:{encoding} label matches={len(found)}')
            except Exception as exc:
                attempts.append(f'{repr(sep)}:{encoding} {type(exc).__name__}: {exc}')
    raise ValueError(f'Could not read {path.name} with a unique {label!r} label. '
                     + ' | '.join(attempts[:12]))


def encode_labels(values):
    """Binary encoding for count/binary defect labels; >0 is defective."""
    numeric=pd.to_numeric(values,errors='coerce')
    text=values.astype('string').str.strip().str.lower()
    replacement={'true':1,'false':0,'yes':1,'no':0,'defective':1,
                 'clean':0,'buggy':1,'nondefective':0,'non-defective':0,
                 'positive':1,'negative':0,'y':1,'n':0,'t':1,'f':0}
    for name,val in replacement.items():
        numeric=numeric.where(~text.eq(name).fillna(False),val)
    if numeric.isna().any():
        examples=text[numeric.isna()].head(5).tolist()
        raise ValueError(f'Unrecognized/missing defect labels: {examples}. '
                         'No rows have been silently dropped.')
    return (numeric.to_numpy(float)>0).astype(np.int8)


# Names of known metadata/alternative outcomes, not predictive numeric metrics.
# We never infer that an identifier or a secondary defect-label column is a
# valid software metric merely because it contains numbers.
META={'id','moduleid','filename','filepath','path','name','classname','file','class',
      'project','release','version','package','heubug','heubugcount','realbugcount',
      'bugcount','bugs','bug','realbug','label','defect','defects','isdefective',
      'buggy','faulty','fault','faults','target','y','index','unnamed0'}


def _metric_column(col, values):
    """Return (numeric_series or None, reason, invalid_cell_count).

    Preserve numeric columns even when some values are missing; do not label
    encode nominal identifiers. Nonempty, nonnumeric tokens are treated as a
    data-quality issue and excluded rather than guessing an arbitrary code.
    """
    key=canon(col)
    if not key or key.startswith('unnamed'):
        return None,'empty/unnamed header',0
    if key in META:
        return None,'known ID/metadata/alternate outcome',0
    raw=values.replace([np.inf,-np.inf],np.nan)
    numbers=pd.to_numeric(raw,errors='coerce')
    original_nonmissing=raw.notna()
    malformed=int((original_nonmissing & numbers.isna()).sum())
    if malformed:
        return None,f'non-numeric values ({malformed})',malformed
    if not numbers.notna().any():
        return None,'no observed numeric values',0
    return numbers.astype(float),None,0


def load_benchmarks(root, strict=None):
    """Discover projects and usable metric schemas from the supplied CSVs.

    The workbook's metric counts are descriptive, not data-loading constraints.
    Reconcile exact canonical column names WITHIN each family so every
    source/target combination uses a single consistent feature space.
    A comprehensive feature alignment report is exposed on the function.
    """
    root=Path(root).expanduser()
    if not root.is_dir():raise FileNotFoundError(f'Dataset directory not found: {root}')
    projects=defaultdict(list)
    records=[]
    excluded=[]
    aligned=[]
    for family,spec in BENCHMARKS.items():
        matching_folders=[p for p in root.iterdir() if p.is_dir() and canon(p.name)==family]
        if not matching_folders:
            LOG.warning('No %s dataset folder found under %s; skipping family',family,root)
            continue
        if len(matching_folders)!=1:
            raise ValueError(f'Ambiguous family folders for {family}: {matching_folders}')
        folder=matching_folders[0]
        csv_paths=sorted(p for p in folder.rglob('*') if p.is_file() and
                         p.suffix.lower()=='.csv' and not p.name.startswith('.'))
        if not csv_paths:
            LOG.warning('No project CSVs in %s',folder)
            continue
        for path in csv_paths:
            df,label,delimiter=read_csv_flexible(path,spec['label'])
            y=encode_labels(df[label])
            included={}
            for col in df.columns:
                if col==label:continue
                series,reason,invalid=_metric_column(col,df[col])
                if reason:
                    excluded.append({'family':family,'project':path.stem,
                         'column':col,'reason':reason,'invalid_nonempty_cells':invalid})
                    LOG.info('Skip metric candidate %r in %s: %s',col,path.name,reason)
                    continue
                key=canon(col)
                if key in included:
                    raise ValueError(f'Conflicting metric headers in {path.name}: '
                                     f'{included[key][0]!r} and {col!r}')
                included[key]=(col,series)
            if not included:
                raise ValueError(f'No numeric metrics remain in {path.name}; inspect header')
            keys=list(included)
            X=np.column_stack([included[key][1].to_numpy(dtype=float) for key in keys])
            if len(df)!=len(y) or not np.isfinite(y).all():
                raise ValueError(f'Invalid observations or labels in {path.name}')
            name=path.stem
            if any(canon(p.name)==canon(name) for p in projects[family]):
                raise ValueError(f'Duplicate project stem within {family}: {name}')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            projects[family].append(Project(family,name,X,y,[included[key][0] for key in keys],
                                            str(path),digest))
            records.append({'family':family,'project':name,'rows':len(df),
                 'defect_count':int(y.sum()),'defect_percent':round(100*y.mean(),4),
                 'numeric_metrics_in_file':len(keys),'raw_columns':len(df.columns),
                 'excluded_columns':len(df.columns)-len(keys)-1,
                 'missing_numeric_cells':int(np.isnan(X).sum()),
                 'delimiter':repr(delimiter),'sha256':digest,'file':str(path)})
            LOG.info('Accepted %s/%s: %d rows; %d numeric metric columns; %.2f%% defective',
                     family,name,len(df),len(keys),100*y.mean())
        ps=projects[family]
        if len(ps)<2:
            LOG.warning('Skipping %s: within-family CPDP needs at least two CSV projects',family)
            del projects[family]
            continue
        # Canonical metric names (not columns or positions) define the shared
        # family schema. Report nonshared metrics; do not manufacture missing
        # columns or silently mix incompatible input dimensions.
        by_project=[{canon(x):x for x in p.features} for p in ps]
        common=set(by_project[0])
        for fmap in by_project[1:]:common.intersection_update(fmap)
        order=[canon(c) for c in ps[0].features if canon(c) in common]
        if len(order)<2:
            detail={p.name:len(p.features) for p in ps}
            raise ValueError(f'{family}: insufficient shared numeric metric columns '
                f'({len(order)}). Individual counts: {detail}. '
                'Verify that CSVs use compatible feature definitions; do not '
                'rename unrelated metrics to force a match.')
        for p,fmap in zip(ps,by_project):
            old=p.features.copy();n_original=len(old)
            idx={canon(x):i for i,x in enumerate(old)}
            removed=[x for x in old if canon(x) not in common]
            for col in old:
                aligned.append({'family':family,'project':p.name,'metric':col,
                                'shared_across_family':int(canon(col) in common),
                                'reason':'' if canon(col) in common else 'not in every project'})
            p.X=p.X[:,[idx[k] for k in order]]
            p.features=[fmap[k] for k in order]
            rec=next(r for r in records if r['family']==family and r['project']==p.name)
            rec['shared_family_metrics']=len(order)
            rec['nonshared_numeric_metrics']=len(removed)
            if removed:
                LOG.warning('%s/%s: %d of %d numeric metrics are not shared and are '
                            'excluded from within-family comparison: %s',
                            family,p.name,len(removed),n_original,removed[:25])
        LOG.info('FINAL %s shared feature schema: %d metrics, %d projects',family,len(order),len(ps))
    if not projects:
        raise ValueError('No dataset family has at least two valid project CSVs')
    load_benchmarks.excluded_columns=pd.DataFrame(excluded,
        columns=['family','project','column','reason','invalid_nonempty_cells'])
    load_benchmarks.feature_alignment=pd.DataFrame(aligned,
        columns=['family','project','metric','shared_across_family','reason'])
    load_benchmarks.family_schemas={family:ps[0].features.copy() for family,ps in projects.items()}
    return dict(projects),pd.DataFrame(records)

def make_view(family,projects,target_index):
    target=projects[target_index]; sources=[p for i,p in enumerate(projects) if i!=target_index]
    names=list(target.features)
    def align(p):
        positions={canon(c):i for i,c in enumerate(p.features)}
        return p.X[:,[positions[canon(c)] for c in names]]
    Xs=np.vstack([align(p) for p in sources]); ys=np.concatenate([p.y for p in sources]);
    env=np.concatenate([np.full(len(p.y),i,dtype=int) for i,p in enumerate(sources)])
    return SourceTarget(family,target.name,Xs,ys,env,[p.name for p in sources],align(target),names)


def gaussian_component_logpdf(model, X):
    """Each GMM component's NORMAL log density, without fitted priors."""
    X=np.asarray(X,dtype=float)
    diff=X[:,None,:]-model.means_[None,:,:]
    diag=model.covariances_
    d=X.shape[1]
    return -.5*(d*np.log(2*np.pi)+np.sum(np.log(diag),axis=1)[None,:]
                 +np.sum(diff*diff/diag[None,:,:],axis=2))


def component_assign(model,X,prior='equal'):
    values=gaussian_component_logpdf(model,X)
    if prior=='fitted': values=values+np.log(model.weights_)[None,:]
    return np.argmax(values,axis=1)


def component_responsibilities(model,X,prior='equal'):
    values=gaussian_component_logpdf(model,X)
    if prior=='fitted': values=values+np.log(model.weights_)[None,:]
    return np.exp(values-logsumexp(values,axis=1,keepdims=True))


def fit_gmm_class(X,env,settings,seed):
    n,d=X.shape
    min_n=max(settings.component_min_floor,d)
    if n<min_n: raise Inadmissible(f'class group size {n} < min admissible {min_n}')
    candidates=[]
    for k in range(1,min(settings.k_max,n//min_n)+1):
        try:
            model=GaussianMixture(n_components=k,covariance_type='diag',
                       reg_covar=settings.gmm_reg,n_init=settings.gmm_n_init,
                       random_state=seed+k,max_iter=250)
            model.fit(X)
            assigned=component_assign(model,X,'equal')
            counts=np.bincount(assigned,minlength=k)
            if counts.min()<min_n: continue
            candidates.append({'k':k,'model':model,'bic':float(model.bic(X)),
                               'n':counts.tolist()})
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            LOG.warning('GMM k=%s inadmissible: %s',k,exc)
    if not candidates: raise Inadmissible('no GMM candidate meets hard-assigned minimum size')
    best_bic=min(c['bic'] for c in candidates)
    eligible=[c for c in candidates if c['bic']<=best_bic+settings.bic_tolerance+1e-9]
    for c in eligible:
        k=c['k']; base=component_assign(c['model'],X)
        scores=[]
        for m in np.unique(env):
            retained=np.where(env!=m)[0]
            if len(retained)<k: raise Inadmissible('too few observations for stability refit')
            refit=GaussianMixture(n_components=k,covariance_type='diag',
                         reg_covar=settings.gmm_reg,n_init=settings.gmm_n_init,
                         random_state=seed+10000+101*k+int(m),max_iter=250)
            try:
                refit.fit(X[retained])
                scores.append(float(adjusted_rand_score(base[retained],
                                          component_assign(refit,X[retained]))))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                raise Inadmissible(f'GMM project-held-out stability refit failed: {exc}') from exc
        c['stability']=float(np.mean(scores))
        c['fold_ari']=scores
    if settings.gmm_selection == 'bic_first':
        chosen=sorted(eligible,key=lambda c:(c['bic'], -c['stability'], c['k']))[0]
    else:
        best_stability=max(c['stability'] for c in eligible)
        tied=[c for c in eligible if best_stability-c['stability'] <= 1e-6]
        chosen=min(tied,key=lambda c:(c['k'],c['bic']))
    trace=[{'k':c['k'],'bic':c['bic'],'eligible':int(c in eligible),
            'selected':int(c is chosen),'stability':c.get('stability',np.nan),
            'counts':json.dumps(c['n'])} for c in candidates]
    return chosen['model'],trace


def discover(Xs,ys,env,settings,seed):
    models={}; groups=np.full(len(ys),-1,dtype=int); group_rows=[]; candidate_rows=[]
    for cls in (0,1):
        idx=np.flatnonzero(ys==cls)
        if not len(idx): raise Inadmissible(f'missing source class {cls}')
        m,trace=fit_gmm_class(Xs[idx],env[idx],settings,seed+cls*100000)
        models[cls]=m
        groups[idx]=component_assign(m,Xs[idx],prior='equal')
        for record in trace: candidate_rows.append({'class':cls,**record})
        for g in range(m.n_components):
            chosen=idx[groups[idx]==g]
            group_rows.append({'class':cls,'group':g,'n_yg':int(len(chosen)),
                'b_yg':int(len(np.unique(env[chosen]))),
                'source_fraction':float(len(chosen)/len(idx)),
                'mixture_weight':float(m.weights_[g])})
    if np.any(groups<0): raise AssertionError('source assignment incomplete')
    return models,groups,group_rows,candidate_rows


def make_probe(name,seed,settings):
    if name=='LR': return LogisticRegression(C=settings.lr_c,max_iter=3000,solver='lbfgs',random_state=seed)
    if name=='RF': return RandomForestClassifier(n_estimators=settings.rf_trees,
            random_state=seed,n_jobs=-1,min_samples_leaf=1)
    raise ValueError(name)


def raw_logit(model,X,clip):
    p=np.asarray(model.predict_proba(X))[:,list(model.classes_).index(1)]
    p=np.clip(p,clip,1-clip)
    return np.log(p)-np.log1p(-p)


def calibrated_heldout_probs(Xraw,y,env,held_env,probe,settings,seed):
    """Outer LOPO predictions; calibration from INNER source-project-grouped OOF scores.

    Each inner pipeline fits only on inner training projects. Then sigmoid
    Platt calibrator is fit on OOF logits and applied to a base fitted on
    all outer training sources. No outer held-project row enters this fit.
    """
    tr=np.flatnonzero(env!=held_env); te=np.flatnonzero(env==held_env)
    if not len(te): raise Inadmissible('empty held-out source project')
    yt=y[tr]; envt=env[tr]
    if len(np.unique(yt))!=2: raise Inadmissible('outer probe fold lacks two training classes')
    if len(np.unique(envt))<2: raise Inadmissible('grouped calibration needs at least two source projects')
    oof=np.full(len(tr),np.nan)
    for m in np.unique(envt):
        inner_train=np.flatnonzero(envt!=m); inner_val=np.flatnonzero(envt==m)
        if len(np.unique(yt[inner_train]))!=2:
            raise Inadmissible(f'inner calibration fold {int(m)} lacks both training classes')
        t=fit_transformer(Xraw[tr[inner_train]])
        clf=make_probe(probe,seed+1000+int(m),settings)
        clf.fit(t.transform(Xraw[tr[inner_train]]),yt[inner_train])
        oof[inner_val]=raw_logit(clf,t.transform(Xraw[tr[inner_val]]),settings.probability_clip)
    if not np.isfinite(oof).all() or len(np.unique(yt))!=2:
        raise Inadmissible('incomplete sigmoid calibration OOF data')
    calibrator=LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000,random_state=seed)
    calibrator.fit(oof.reshape(-1,1),yt)
    t=fit_transformer(Xraw[tr]); clf=make_probe(probe,seed+9000,settings)
    clf.fit(t.transform(Xraw[tr]),yt)
    held_logit=raw_logit(clf,t.transform(Xraw[te]),settings.probability_clip)
    calibrated=calibrator.predict_proba(held_logit.reshape(-1,1))[:,list(calibrator.classes_).index(1)]
    return te,np.clip(calibrated,settings.probability_clip,1-settings.probability_clip)


def transfer_losses(view,settings,seed):
    losses=[]; trace=[]
    for name in ('LR','RF'):
        prob=np.full(len(view.ys),np.nan)
        for m in np.unique(view.env):
            indices,p=calibrated_heldout_probs(view.Xs_raw,view.ys,view.env,m,name,settings,seed+77*int(m)+(0 if name=='LR' else 20000))
            prob[indices]=p
            trace.append({'probe':name,'held_project':view.source_names[int(m)],'n':len(indices),
                          'mean_true_class_logloss':float(np.mean(-np.log(np.where(view.ys[indices]==1,p,1-p))))})
        if not np.isfinite(prob).all(): raise Inadmissible(f'incomplete project-held-out probabilities: {name}')
        truth=np.where(view.ys==1,prob,1-prob)
        losses.append(-np.log(np.clip(truth,settings.probability_clip,1-settings.probability_clip)))
    return np.mean(losses,axis=0),trace


def source_ambiguity(X,y,epsilon):
    out=np.empty(len(y),dtype=float)
    for cls in (0,1):
        idx=np.flatnonzero(y==cls); neg=np.flatnonzero(y!=cls)
        if len(idx)<2 or not len(neg): raise Inadmissible('insufficient source observations for same/opposite-class neighbors')
        knn=NearestNeighbors(n_neighbors=2,n_jobs=-1).fit(X[idx])
        dist,ind=knn.kneighbors(X[idx],n_neighbors=2)
        # With duplicates, self may not be included in 2 returned neighbors.
        # For each point, exclude its exact index explicitly, requesting extra tie
        # neighbors when necessary. This is a deterministic fixed two-neighbor
        # correction; if self absent the closest returned is already another row.
        local=[]
        for j in range(len(idx)):
            valid=dist[j,ind[j]!=j]
            local.append(float(valid[0]) if len(valid) else float(dist[j,1]))
        plus=np.asarray(local)
        minus=NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(X[neg]).kneighbors(X[idx])[0][:,0]
        ambiguity=plus/(plus+minus+epsilon)
        ambiguity[(plus==0)&(minus==0)]=.5
        out[idx]=ambiguity
    return out


def reference_distances(Xs,Xt,q):
    """q-th other-source neighbor for sources; q-th source neighbor for targets."""
    if q<1 or q>=len(Xs): raise Inadmissible(f'q={q} invalid for {len(Xs)} source observations')
    model=NearestNeighbors(n_neighbors=q+1,n_jobs=-1).fit(Xs)
    sdist,sindex=model.kneighbors(Xs,n_neighbors=q+1)
    src=np.empty(len(Xs),dtype=float)
    for i in range(len(Xs)):
        ds=sdist[i,sindex[i]!=i]
        src[i]=float(ds[q-1]) # extra neighbor guarantees >=q after excluding self
    tgt=model.kneighbors(Xt,n_neighbors=q)[0][:,q-1]
    return src,tgt


def proximity(source_ref,target_dist,alpha):
    tau=float(np.quantile(source_ref,1-alpha))
    covered=(target_dist<=tau)
    return covered,tau,1-float(np.mean(covered))


def source_profile(group_rows,groups,y,loss,a,lam):
    rows=[]
    for row in group_rows:
        cls,g=int(row['class']),int(row['group'])
        idx=(y==cls)&(groups==g)
        L=float(np.mean(loss[idx]));D=float(-np.expm1(-L));A=float(np.mean(a[idx]));H=lam*D+(1-lam)*A
        rows.append({**row,'L_tr':L,'D':D,'A':A,'H':H})
    return rows


def relevance(models,Xt,covered,prior='equal'):
    if not np.any(covered): return None
    return {cls:component_responsibilities(models[cls],Xt[covered],prior).mean(axis=0)
            for cls in (0,1)}


def risk_rows(src_rows,T,lam):
    if T is None: return [],{0:np.nan,1:np.nan}
    risk={0:0.,1:0.}; result=[]
    for row in src_rows:
        cls,g=int(row['class']),int(row['group']); t=float(T[cls][g])
        rtr=t*row['D']; ramb=t*row['A']; r=lam*rtr+(1-lam)*ramb
        result.append({**row,'T':t,'R_tr':rtr,'R_amb':ramb,'R':r})
        risk[cls]+=r
    for cls in (0,1):
        if abs(sum(r['T'] for r in result if r['class']==cls)-1)>1e-8:
            raise AssertionError('T relevance normalization broken')
    return result,risk


def comparison_diagnostics(source_rows,groups,ys,loss,a,models,Xt,covered,T,lam):
    """Ablations in same target run; no target labels or prediction retraining."""
    alternative=[]
    for cls in (0,1):
        clsrows=[r for r in source_rows if r['class']==cls]
        srcfreq=np.asarray([r['n_yg'] for r in clsrows],float);srcfreq/=srcfreq.sum()
        uniform=np.full(len(clsrows),1/len(clsrows))
        weighting={'TREAD':T[cls] if T is not None else None,
                   'source_frequency':srcfreq,'uniform':uniform}
        for method,weights in weighting.items():
            score=sum(float(w)*r['H'] for w,r in zip(weights,clsrows)) if weights is not None else np.nan
            alternative.append({'class':cls,'method':method,'score':score})
        idx=ys==cls
        L=float(np.mean(loss[idx])); D=float(-np.expm1(-L)); A=float(np.mean(a[idx]))
        alternative.append({'class':cls,'method':'single_component', 'score':lam*D+(1-lam)*A})
        if T is not None:
            alternative.append({'class':cls,'method':'transfer_only',
                                'score':sum(T[cls][r['group']]*r['D'] for r in clsrows)})
            alternative.append({'class':cls,'method':'ambiguity_only',
                                'score':sum(T[cls][r['group']]*r['A'] for r in clsrows)})
            fitted=component_responsibilities(models[cls],Xt[covered],'fitted').mean(axis=0)
            alternative.append({'class':cls,'method':'fitted_prior_affinity',
                                'score':sum(fitted[r['group']]*r['H'] for r in clsrows)})
        else:
            for method in ('transfer_only','ambiguity_only','fitted_prior_affinity'):
                alternative.append({'class':cls,'method':method,'score':np.nan})
    return alternative


def mmd_source_target(Xs,Xt,seed,max_n=600):
    """Biased Gaussian kernel MMD; median distance bandwidth from source ONLY."""
    rng=np.random.default_rng(seed)
    A=Xs[rng.choice(len(Xs),min(max_n,len(Xs)),replace=False)]
    B=Xt[rng.choice(len(Xt),min(max_n,len(Xt)),replace=False)]
    dist=pdist(A,'euclidean'); positives=dist[dist>0]
    if not len(positives): return np.nan,np.nan
    bw=float(np.median(positives)); denom=2*bw*bw
    Kxx=np.exp(-cdist(A,A,'sqeuclidean')/denom).mean()
    Kyy=np.exp(-cdist(B,B,'sqeuclidean')/denom).mean()
    Kxy=np.exp(-cdist(A,B,'sqeuclidean')/denom).mean()
    return float(np.sqrt(max(0.,Kxx+Kyy-2*Kxy))),bw


def build_diagnosis(view:SourceTarget,settings:Settings,seed=None):
    """Only sources (Xs_raw,ys,env) and unlabeled target metrics (Xt_raw) accepted."""
    seed=settings.seed if seed is None else seed
    transform=fit_transformer(view.Xs_raw)
    Xs=transform.transform(view.Xs_raw);Xt=transform.transform(view.Xt_raw)
    models,group_id,group_rows,candidates=discover(Xs,view.ys,view.env,settings,seed)
    project_composition=[]
    for g in group_rows:
        cls,component=g['class'],g['group']
        mask=(view.ys==cls)&(group_id==component)
        for env_index,project in enumerate(view.source_names):
            count=int(np.sum(mask & (view.env==env_index)))
            project_composition.append({'class':cls,'group':component,'source_project':project,
                 'n':count,'fraction_of_group':count/g['n_yg']})
    individual_losses,probe_trace=transfer_losses(view,settings,seed)
    ai=source_ambiguity(Xs,view.ys,settings.epsilon)
    src=source_profile(group_rows,group_id,view.ys,individual_losses,ai,settings.lam)
    dsrc,dtgt=reference_distances(Xs,Xt,settings.q)
    covered,tau,U=proximity(dsrc,dtgt,settings.alpha)
    T=relevance(models,Xt,covered)
    final,risk=risk_rows(src,T,settings.lam)
    comparisons=comparison_diagnostics(src,group_id,view.ys,individual_losses,ai,models,Xt,covered,T,settings.lam)
    mmd,bw=mmd_source_target(Xs,Xt,seed,settings.mmd_sample_max)
    sensitivity=[]
    for q in settings.q_grid:
        dr,dt=reference_distances(Xs,Xt,q)
        for alpha in settings.alpha_grid:
            c,t,u=proximity(dr,dt,alpha)
            tt=relevance(models,Xt,c)
            _,rr=risk_rows(src,tt,settings.lam)
            sensitivity.append({'dimension':'q_alpha','q':q,'alpha':alpha,
                'lambda':settings.lam,'gmm_seed':seed,'U':u,'R0':rr[0],'R1':rr[1],'N_c':int(c.sum())})
    for lam in settings.lambda_grid:
        sr=source_profile(group_rows,group_id,view.ys,individual_losses,ai,lam)
        _,rr=risk_rows(sr,T,lam)
        sensitivity.append({'dimension':'lambda','q':settings.q,'alpha':settings.alpha,
            'lambda':lam,'gmm_seed':seed,'U':U,'R0':rr[0],'R1':rr[1],'N_c':int(covered.sum())})
    # Run GMM-only seed sensitivity without refitting LOPO probes.
    for sseed in settings.gmm_sensitivity_seeds:
        try:
            sm,sg,srows,_=discover(Xs,view.ys,view.env,settings,sseed)
            ss=source_profile(srows,sg,view.ys,individual_losses,ai,settings.lam)
            st=relevance(sm,Xt,covered)
            _,rr=risk_rows(ss,st,settings.lam)
            sensitivity.append({'dimension':'gmm_seed','q':settings.q,'alpha':settings.alpha,
                'lambda':settings.lam,'gmm_seed':sseed,'U':U,'R0':rr[0],'R1':rr[1],
                'N_c':int(covered.sum()),'K0':sm[0].n_components,'K1':sm[1].n_components})
        except Inadmissible as exc:
            sensitivity.append({'dimension':'gmm_seed','gmm_seed':sseed,'status':f'unavailable: {exc}'})
    # Source class-prior reference: diagnostic score for class y.
    source_prior=float(view.ys.mean())
    prior_ll={0:float(-np.log(np.clip(1-source_prior,settings.probability_clip,1))),
              1:float(-np.log(np.clip(source_prior,settings.probability_clip,1)))}
    return {'models':models,'transformation':transform,'Xs':Xs,'Xt':Xt,'source_group':group_id,
       'source_rows':src,'risk_rows':final,'project_composition':project_composition,
       'candidates':candidates,'probe_trace':probe_trace,
       'individual_loss':individual_losses,'individual_ambiguity':ai,
       'covered':covered,'source_ref':dsrc,'target_distance':dtgt,'threshold':tau,'U':U,
       'N_c':int(covered.sum()),'T':T,'risk':risk,'comparisons':comparisons,'sensitivity':sensitivity,
       'mmd':mmd,'mmd_bandwidth':bw,'prior_ll':prior_ll,'d':Xs.shape[1]}

def make_downstream(name,seed,settings):
    if name=='GBT':
        return GradientBoostingClassifier(n_estimators=200,learning_rate=.05,
                      max_depth=3,random_state=seed)
    return make_probe(name,seed,settings)   # LR and RF from same prespecified families


def binary_loss(y,p,clip):
    prob=np.clip(np.asarray(p,float),clip,1-clip)
    return -np.where(np.asarray(y)==1,np.log(prob),np.log1p(-prob))


def rates(y,p):
    y=np.asarray(y); p=np.asarray(p)
    if not len(y): return {'recall':np.nan,'fpr':np.nan}
    predicted=p>=.5
    pos=y==1; neg=y==0
    return {'recall':float(predicted[pos].mean()) if pos.any() else np.nan,
            'fpr':float(predicted[neg].mean()) if neg.any() else np.nan}


def evaluate_target(view,truth,diagnosis,settings,seed_count):
    """Target y is supplied HERE, after complete diagnosis() returned."""
    y=np.asarray(truth,int)
    if len(y)!=len(view.Xt_raw): raise ValueError('target labels length mismatch')
    covered=diagnosis['covered']
    outer_rows=[]; subsets=[]; predictions=[]
    for name in ('GBT','LR','RF'):
        for nseed in range(seed_count):
            seed=settings.seed+nseed  # prespecified seeds 2027--2036 (learner names distinguish runs)
            model=make_downstream(name,seed,settings)
            # FULL outer source preprocessing is fixed BEFORE target is used.
            model.fit(diagnosis['Xs'],view.ys)
            prob=np.clip(model.predict_proba(diagnosis['Xt'])[:,list(model.classes_).index(1)],
                         settings.probability_clip,1-settings.probability_clip)
            loss=binary_loss(y,prob,settings.probability_clip)
            overall=rates(y,prob)
            outer_rows.append({'learner':name,'seed':seed,'n':len(y),'full_logloss':float(loss.mean()),
                     'full_recall':overall['recall'],'full_fpr':overall['fpr'],
                     'full_auprc':float(average_precision_score(y,prob)) if len(np.unique(y))==2 else np.nan,
                     'full_mcc':float(matthews_corrcoef(y,prob>=.5)) if len(np.unique(y))==2 else np.nan})
            for cls in (0,1):
                for selection,mask in [('proximate',covered),('exceedance',~covered),('all',np.ones(len(y),bool))]:
                    chosen=mask&(y==cls)
                    subsets.append({'learner':name,'seed':seed,'class':cls,'subset':selection,
                       'n':int(chosen.sum()),'low_support':int(0<chosen.sum()<settings.min_primary_class_n), 'trueclass_logloss':float(loss[chosen].mean()) if chosen.any() else np.nan,
                       'class_success':float(np.mean((prob[chosen]>=.5)==y[chosen])) if chosen.any() else np.nan})
            # Per-target aggregate losses for the exceedance comparison, with no class conditioning.
            for selection,mask in [('proximate',covered),('exceedance',~covered)]:
                subsets.append({'learner':name,'seed':seed,'class':-1,'subset':selection,
                     'n':int(mask.sum()),'trueclass_logloss':float(loss[mask].mean()) if mask.any() else np.nan,
                     'class_success':float(np.mean((prob[mask]>=.5)==y[mask])) if mask.any() else np.nan})
            for j in range(len(y)):
                predictions.append({'learner':name,'seed':seed,'target_row':j,
                    'true_label':int(y[j]),'p_defect':float(prob[j]),'trueclass_logloss':float(loss[j]),
                    'source_proximate':int(covered[j]),
                    'source_qnn_distance':float(diagnosis['target_distance'][j])})
    return outer_rows,subsets,predictions


def controlled_cases(settings):
    """Deterministic construct tests; do NOT treat synthetic scores as real-world evidence."""
    rng=np.random.default_rng(59019)
    # Two dense source clouds, with low within-cloud nearest-neighbor distances.
    xs=np.vstack((rng.normal(loc=-6,scale=.35,size=(160,2)),
                  rng.normal(loc=6,scale=.35,size=(160,2))))
    near=rng.normal(loc=-6,scale=.35,size=(90,2))
    gap=rng.normal(loc=0,scale=.25,size=(90,2))
    far=rng.normal(loc=20,scale=.25,size=(90,2))
    rows=[]
    for q in settings.q_grid:
        for name,xt in [('in_source_cloud',near),('inter_source_gap',gap),('distant',far)]:
            ds,dt=reference_distances(xs,xt,q)
            for alpha in settings.alpha_grid:
                c,t,u=proximity(ds,dt,alpha)
                rows.append({'scenario':name,'q':q,'alpha':alpha,'U':u,'threshold':t,
                             'median_target_qnn_distance':float(np.median(dt))})
    return pd.DataFrame(rows)


def clean_json(x):
    if isinstance(x,dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean_json(t) for t in x]
    if isinstance(x,np.ndarray): return clean_json(x.tolist())
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return None if not np.isfinite(x) else float(x)
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def dump_json(path,data):
    Path(path).write_text(json.dumps(clean_json(data),indent=2,sort_keys=True))


def write_frame(path,records):
    pd.DataFrame(records).to_csv(path,index=False)


def save_target(folder,view,diag,outer,subsets,predictions):
    folder.mkdir(parents=True,exist_ok=True)
    ids={'family':view.family,'target':view.target}
    finished={**ids,'status':'ok','n_sources':len(view.ys),
      'n_target':len(view.Xt_raw),'source_projects':view.source_names,
      'n_features_after_selection':diag['d'],'q_threshold':diag['threshold'],
      'N_c':diag['N_c'],'U':diag['U'],'R0':diag['risk'][0],'R1':diag['risk'][1],
      'mmd':diag['mmd'],'mmd_source_bandwidth':diag['mmd_bandwidth'],
      'source_defective_fraction':float(view.ys.mean()),'prior_ll_y0':diag['prior_ll'][0],
      'prior_ll_y1':diag['prior_ll'][1]}
    for fname,rows in [('groups.csv',diag['risk_rows'] if diag['T'] is not None else diag['source_rows']),
             ('gmm_candidates.csv',diag['candidates']),('group_project_counts.csv',diag['project_composition']),
             ('probe_folds.csv',diag['probe_trace']),
             ('comparators.csv',diag['comparisons']),('sensitivity.csv',diag['sensitivity']),
             ('full_metrics.csv',outer),('subset_metrics.csv',subsets),('module_predictions.csv',predictions)]:
        write_frame(folder/fname,[{**ids,**r} for r in rows])
    # Independently audit exactly how target proximity is calculated.
    write_frame(folder/'proximity.csv',[{**ids,'target_row':j,'qnn_distance':float(d),
        'threshold':diag['threshold'],'c_j':int(diag['covered'][j])}
        for j,d in enumerate(diag['target_distance'])])
    # Complete source-only feature-selection audit, including imputation and
    # removal of constants after source pooling; target labels are never read.
    trans=diag['transformation']
    source_feature_rows=[]
    position=0
    for k,name in enumerate(view.feature_names):
        retained=bool(trans.keep[k])
        source_feature_rows.append({'family':view.family,'target':view.target,
           'feature_name':name,'kept_for_this_target':int(retained),
           'source_median_imputation':float(trans.median[k]),
           'source_center':float(trans.center[position]) if retained else np.nan,
           'source_iqr_divisor':float(trans.scale[position]) if retained else np.nan})
        if retained:position+=1
    write_frame(folder/'source_feature_preprocessing.csv',source_feature_rows)
    # Completion marker written last: resume cannot confuse partial output with a complete run.
    dump_json(folder/'diagnosis.json',finished)


def read_completed(output,table):
    records=[]
    for f in sorted(Path(output).glob(f'targets/*/*/{table}')):
        try:
            df=pd.read_csv(f)
            if not df.empty: records.append(df)
        except pd.errors.EmptyDataError: continue
    return pd.concat(records,ignore_index=True) if records else pd.DataFrame()


def bootstrap_spearman(x,y,n,seed):
    """Descriptive target-project bootstrap; outer runs share sources."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y)
    x=x[valid];y=y[valid]
    if len(x)<4 or len(np.unique(x))<2 or len(np.unique(y))<2:return np.nan,np.nan
    rng=np.random.default_rng(seed); correlations=[]
    for _ in range(n):
        chosen=rng.integers(0,len(x),len(x))
        if len(np.unique(x[chosen]))>1 and len(np.unique(y[chosen]))>1:
            r=spearmanr(x[chosen],y[chosen]).statistic
            if np.isfinite(r):correlations.append(float(r))
    return (float(np.quantile(correlations,.025)),float(np.quantile(correlations,.975))) if correlations else (np.nan,np.nan)


def project_correlations(dataset, score_cols,outcome,settings,kind):
    results=[]
    for family in ('overall',*BENCHMARKS):
        part=dataset if family=='overall' else dataset[dataset.family==family]
        for col in score_cols:
            if col not in part or outcome not in part: continue
            selected=part[[col,outcome]].replace([np.inf,-np.inf],np.nan).dropna()
            x=selected[col].to_numpy(float); y=selected[outcome].to_numpy(float)
            n=len(x)
            rho=np.nan
            if n>=4 and len(np.unique(x))>1 and len(np.unique(y))>1:
                rho=float(spearmanr(x,y).statistic)
            low,high=bootstrap_spearman(x,y,settings.bootstrap_replicates,settings.seed+len(results))
            results.append({'type':kind,'family':family,'diagnostic':col,'outcome':outcome,
                            'n_targets':n,'spearman_rho':rho,'bootstrap95_low':low,'bootstrap95_high':high})
    return results


def analysis_pipeline(output,settings):
    output=Path(output)
    diag=read_completed(output,'diagnosis.csv') # intentionally unused: main status is JSON
    summary=[]
    for file in sorted(output.glob('targets/*/*/diagnosis.json')):
        summary.append(json.loads(file.read_text()))
    if not summary:
        raise ValueError('No target diagnosis files; run the pipeline first')
    summary_df=pd.DataFrame(summary)
    summary_df.to_csv(output/'target_diagnoses.csv',index=False)
    avail=summary_df.groupby(['family','status']).size().reset_index(name='n_targets')
    avail.to_csv(output/'diagnostic_availability.csv',index=False)
    for src,dst in [('source_feature_preprocessing.csv','source_feature_preprocessing.csv'),
                    ('groups.csv','representation_profiles.csv'),('group_project_counts.csv','representation_project_composition.csv'),('gmm_candidates.csv','gmm_selection.csv'),
                    ('probe_folds.csv','probe_fold_diagnostics.csv'),('comparators.csv','ablation_diagnostics.csv'),
                    ('sensitivity.csv','sensitivity_results.csv'),('full_metrics.csv','downstream_full_metrics.csv'),
                    ('subset_metrics.csv','downstream_subset_metrics.csv')]:
        read_completed(output,src).to_csv(output/dst,index=False)
    a=read_completed(output,'subset_metrics.csv')
    if a.empty:
        LOG.warning('No target outcome metrics to analyze')
        return
    a=a[a['learner']=='GBT'].copy()
    # Each target is ONE observation; independent learner seeds are averaged.
    primary=a.groupby(['family','target','class','subset'],as_index=False)[['trueclass_logloss','n']].mean()
    primary.to_csv(output/'primary_outcomes.csv',index=False)
    diag_valid=summary_df[summary_df.status=='ok'].copy()
    correlations=[]; ablation=[]
    cmp=read_completed(output,'comparators.csv')
    metrics=read_completed(output,'full_metrics.csv')
    metric_primary=metrics[metrics.learner=='GBT'].groupby(['family','target'],as_index=False)[['full_logloss','full_recall','full_fpr','full_auprc','full_mcc']].mean()
    p=primary[(primary['class'].isin([0,1]))&(primary.subset=='proximate')]
    pwide=p.pivot(index=['family','target'],columns='class',values='trueclass_logloss').reset_index()
    pwide=pwide.rename(columns={0:'proximate_logloss_y0',1:'proximate_logloss_y1'})
    for cname in ('proximate_logloss_y0','proximate_logloss_y1'):
        if cname not in pwide: pwide[cname]=np.nan
    frame=diag_valid.merge(pwide,on=['family','target'],how='left').merge(metric_primary,on=['family','target'],how='left')
    frame.to_csv(output/'project_rq2_analysis.csv',index=False)
    for cls in (0,1):
        correlations+=project_correlations(frame,[f'R{cls}','U','mmd',f'prior_ll_y{cls}'],
                      f'proximate_logloss_y{cls}',settings,'RQ2')
    correlations+=project_correlations(frame,['U','R0','R1','mmd'],'full_logloss',settings,'RQ2_secondary')
    # Under the exact outcome correspondence, investigate source-proximity exceedance.
    gap=primary[(primary['class']==-1)&primary.subset.isin(['proximate','exceedance'])]
    gapw=gap.pivot(index=['family','target'],columns='subset',values='trueclass_logloss').reset_index()
    if {'proximate','exceedance'}.issubset(gapw):
        gapw['exceedance_minus_proximate']=gapw['exceedance']-gapw['proximate']
        gapw=gapw.merge(diag_valid[['family','target','U']],on=['family','target'],how='left')
        gapw.to_csv(output/'proximity_loss_comparison.csv',index=False)
        correlations+=project_correlations(gapw,['U'],'exceedance_minus_proximate',settings,'RQ2_support')
    # All ablation diagnostics paired to the same target and SAME observed outcome.
    if not cmp.empty:
        for cls in (0,1):
            cp=cmp[cmp['class']==cls]
            wide=cp.pivot_table(index=['family','target'],columns='method',values='score',aggfunc='first').reset_index()
            observed=frame[['family','target',f'proximate_logloss_y{cls}']]
            valid=wide.merge(observed,on=['family','target'],how='left')
            av=[c for c in wide.columns if c not in ('family','target')]
            ablation+=project_correlations(valid,av,f'proximate_logloss_y{cls}',settings,f'RQ3_y{cls}')
        pd.DataFrame(ablation).to_csv(output/'ablation_associations.csv',index=False)
    pd.DataFrame(correlations).to_csv(output/'diagnostic_associations.csv',index=False)
    # One simple OUTCOME ANALYSIS ONLY (not used in TREAD construction):
    # Is R_y informative in project-held-out regularized regression beyond
    # source-prior log loss and MMD? Report descriptive MAE and n.
    rows=[]
    for cls in (0,1):
        outcome=f'proximate_logloss_y{cls}'
        feature_sets={'simple':['mmd',f'prior_ll_y{cls}'],
                      'simple_plus_tread':['mmd',f'prior_ll_y{cls}',f'R{cls}','U']}
        for label,cols in feature_sets.items():
            chosen=frame[['family','target',outcome,*cols]].dropna()
            if len(chosen)<8: continue
            y=chosen[outcome].to_numpy(float);X=chosen[cols].to_numpy(float)
            errors=[]
            for i in range(len(y)):
                fit=np.arange(len(y))!=i
                avg=X[fit].mean(axis=0); sd=X[fit].std(axis=0);sd[sd==0]=1
                Xtr=(X[fit]-avg)/sd
                clf=Ridge(alpha=1.0).fit(Xtr,y[fit])
                pred=clf.predict(((X[i:i+1]-avg)/sd))[0]
                errors.append(abs(pred-y[i]))
            rows.append({'class':cls,'feature_set':label,'n':len(y),'loo_mae':float(np.mean(errors)),
                         'median_absolute_error':float(np.median(errors))})
    pd.DataFrame(rows,columns=['class','feature_set','n','loo_mae','median_absolute_error']).to_csv(output/'exploratory_incremental_prediction.csv',index=False)
    # Project/family counts and failure availability are part of scientific reporting.
    dump_json(output/'analysis_manifest.json',{'project_count':len(summary),'analyzable_targets':int(len(diag_valid)),
       'units':'Target-project. Seeds averaged BEFORE comparisons',
       'interval_interpretation':'Descriptive; dependent source pools violate independent-run assumptions',
       'primary_learner':'GradientBoostingClassifier(n_estimators=200, learning_rate=.05, max_depth=3)',
        'primary_outcome':'Class-specific mean true-class log loss, source-proximate subset',
       'missing_metrics':'Remain NaN (never converted to zero)',
       'incremental_regression':'Exploratory leave-one-target-out Ridge; outcome labels used only in posthoc analysis'})



def verify_run_integrity(output,projects,settings,mode):
    """Return a reproducible integrity manifest; distinguish unavailability from errors."""
    out=Path(output)
    cases=[]
    for family,ps in projects.items():
        for index,p in enumerate(ps):
            if mode=='smoke' and index>0: continue
            folder=out/'targets'/family/p.name
            status_file=folder/'diagnosis.json'
            status=json.loads(status_file.read_text()) if status_file.exists() else {}
            status_value=status.get('status','missing')
            problems=[]
            if (folder/'fatal_error.json').exists():problems.append('unexpected_fatal_error')
            if status_value=='ok':
                for filename in ('groups.csv','gmm_candidates.csv','probe_folds.csv',
                     'comparators.csv','sensitivity.csv','full_metrics.csv','subset_metrics.csv',
                     'module_predictions.csv','proximity.csv','group_project_counts.csv',
                     'source_feature_preprocessing.csv'):
                    if not (folder/filename).exists():problems.append('missing_'+filename)
                try:
                    pred=pd.read_csv(folder/'module_predictions.csv')
                    full=pd.read_csv(folder/'full_metrics.csv')
                    prox=pd.read_csv(folder/'proximity.csv')
                    reps=pd.read_csv(folder/'groups.csv')
                    expected=len(p.y)*3*settings.downstream_seeds
                    if len(pred)!=expected:problems.append(f'prediction_count:{len(pred)}_expected:{expected}')
                    if len(full)!=3*settings.downstream_seeds:problems.append('full_metric_count')
                    if len(prox)!=len(p.y):problems.append('proximity_count')
                    if abs(float(status['U'])-(1-int(status['N_c'])/len(p.y)))>1e-10:
                        problems.append('U_identity')
                    for cls in (0,1):
                        r=reps[reps['class']==cls]
                        if int(r['n_yg'].sum())!=sum(q.y.sum() if cls==1 else (len(q.y)-q.y.sum())
                                for j,q in enumerate(ps) if j!=index):
                            problems.append(f'source_group_count_y{cls}')
                        if int(status['N_c'])>0:
                            if abs(float(r['T'].sum())-1)>1e-8:problems.append(f'T_normalization_y{cls}')
                            if abs(float(r['R'].sum())-float(status[f'R{cls}']))>1e-8:
                                problems.append(f'R_sum_y{cls}')
                    if pred.duplicated(['learner','seed','target_row']).any():
                        problems.append('duplicate_prediction_keys')
                except Exception as exc:
                    problems.append(f'output_read_or_validation_error:{exc}')
            elif status_value=='unavailable':
                pass # Methodology prescribes reporting, not imputing a score.
            else:
                problems.append('target_output_missing')
            cases.append({'family':family,'target':p.name,'status':status_value,
                          'checks_pass':not problems,'problems':problems})
    report={'requested_targets':len(cases),
            'completed_ok':sum(c['status']=='ok' for c in cases),
            'unavailable':sum(c['status']=='unavailable' for c in cases),
            'missing_or_failed':sum(c['status'] not in ('ok','unavailable') for c in cases),
            'integrity_errors':sum(not c['checks_pass'] for c in cases),
            'all_outputs_verified':all(c['checks_pass'] for c in cases),
            'cases':cases}
    dump_json(out/'integrity_report.json',report)
    return report

def run_all(args):
    settings=Settings(downstream_seeds=(1 if args.mode=='smoke' else args.seeds))
    if args.mode=='smoke':
        # Non-publication TEST ONLY. Full mode always preserves manuscript settings.
        settings=replace(settings,k_max=2,gmm_n_init=1,rf_trees=10,
                         gmm_sensitivity_seeds=(),q_grid=(5,),alpha_grid=(.05,),
                         lambda_grid=(.5,),mmd_sample_max=100,bootstrap_replicates=100)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    projects,inventory=load_benchmarks(args.data_root)
    manifest_path=output/'run_manifest.json'
    old_manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    hashes={f'{r.family}/{r.project}':r.sha256 for r in inventory.itertuples()}
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.resume:
        if old_manifest is None: raise ValueError('--resume requires an existing run_manifest.json')
        if (old_manifest.get('settings')!=clean_json(asdict(settings))
                or old_manifest.get('input_hashes')!=hashes
                or old_manifest.get('script_sha256')!=script_sha256):
            raise ValueError('Resume blocked: source code, input hashes, or research settings changed. '
                             'Use a NEW results folder instead of mixing configurations.')
    elif old_manifest is not None and list(output.glob('targets/*/*/diagnosis.json')):
        raise FileExistsError('Output already contains target results. Use --resume or a NEW output folder.')
    inventory.to_csv(output/'dataset_inventory.csv',index=False)
    load_benchmarks.feature_alignment.to_csv(output/'feature_alignment.csv',index=False)
    load_benchmarks.excluded_columns.to_csv(output/'excluded_columns.csv',index=False)
    dump_json(output/'family_feature_schemas.json',load_benchmarks.family_schemas)
    LOG.warning('Feature dimensions and project counts come from CSV files, not the manuscript table. '
                'Compare dataset_inventory.csv with the paper before submission.')
    dump_json(output/'run_manifest.json',{'method':'TREAD, not TRIAD','mode':args.mode,
      'requested_full_research_protocol':args.mode=='full',
      'settings':asdict(settings),'input_hashes':hashes,'script_sha256':script_sha256,
      'versions':{'python':platform.python_version(),'numpy':np.__version__,
      'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':SKLEARN_VERSION},
      'target_label_policy':'Only passed to evaluate_target after build_diagnosis finishes',
      'benchmark_reference':'Project CSV headers and observed labels; no fixed workbook counts',
      'data_root':str(Path(args.data_root).resolve())})
    controlled_cases(settings).to_csv(output/'controlled_construct_checks.csv',index=False)
    total=0
    for family,ps in projects.items():
        for index,p in enumerate(ps):
            if args.mode=='smoke' and index>0:continue
            total+=1
            folder=output/'targets'/family/p.name
            done=folder/'diagnosis.json'
            if args.resume and done.exists():
                LOG.info('Resume: skip previously completed %s/%s',family,p.name);continue
            LOG.info('Target %s/%s: %d sources, %d target modules',family,p.name,len(ps)-1,len(p.y))
            view=make_view(family,ps,index)
            try:
                diagnosis=build_diagnosis(view,settings)
                # The next call is FIRST point where target labels are provided.
                outer,subsets,pred=evaluate_target(view,p.y,diagnosis,settings,settings.downstream_seeds)
                save_target(folder,view,diagnosis,outer,subsets,pred)
                LOG.info('Completed %s/%s: R0=%s R1=%s U=%.3f',family,p.name,
                  f'{diagnosis["risk"][0]:.4f}',f'{diagnosis["risk"][1]:.4f}',diagnosis['U'])
            except Inadmissible as e:
                folder.mkdir(parents=True,exist_ok=True)
                dump_json(done,{'family':family,'target':p.name,'status':'unavailable','reason':str(e),
                                 'n_sources':len(view.ys),'n_target':len(p.y)})
                LOG.warning('Diagnosis unavailable: %s/%s: %s',family,p.name,e)
                # A missing diagnostic does not imply that downstream CPDP
                # evaluation must disappear. Predict the complete target with
                # independently trained learners whenever feasible.
                try:
                    evaluate_when_unavailable(folder,view,p.y,settings)
                except Exception as downstream_error:
                    LOG.warning('Full-target downstream evaluation also unavailable for %s/%s: %s',
                        family,p.name,downstream_error)
            except Exception as e:
                folder.mkdir(parents=True,exist_ok=True)
                dump_json(folder/'fatal_error.json',{'family':family,'target':p.name,'error':str(e),
                         'traceback':traceback.format_exc()})
                LOG.exception('Unexpected computation failure %s/%s',family,p.name)
                if not args.continue_on_error:raise
    report=verify_run_integrity(output,projects,settings,args.mode)
    if not report['all_outputs_verified']:
        raise RuntimeError('Integrity verification failed; inspect integrity_report.json')
    analysis_pipeline(output,settings)
    LOG.info('Finished %d target attempts (OK %d, unavailable %d); results: %s',
        total,report['completed_ok'],report['unavailable'],output)




# -------------------------- SELF CONTAINED PLOTTING --------------------------
def _save_chart(fig, base):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(str(base.with_suffix('.pdf')), bbox_inches='tight')
    fig.savefig(str(base.with_suffix('.png')), dpi=180, bbox_inches='tight')
    plt.close(fig)


def generate_figures(output):
    """Create reproducible manuscript-oriented figures from real result CSVs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    out = output / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    src = output / 'representation_profiles.csv'
    if src.exists() and src.stat().st_size:
        x = pd.read_csv(src)
        if len(x) and {'D', 'A', 'class', 'family'}.issubset(x):
            fig, ax = plt.subplots(figsize=(6.4, 4))
            for (family, y), rows in x.groupby(['family', 'class']):
                ax.scatter(rows.D, rows.A, s=24, label=f'{family.upper()}, Y={y}', alpha=.75)
            ax.set(xlabel='Transfer difficulty $D$', ylabel='Local ambiguity $A$',
                   title='RQ1: Within-class representation characteristics')
            ax.legend(fontsize=7, ncol=2)
            base = out / 'rq1_representation_hardness'
            _save_chart(fig, base)
            figures.append(base.name)
    src = output / 'project_rq2_analysis.csv'
    if src.exists() and src.stat().st_size:
        x = pd.read_csv(src)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        any_data = False
        for y in (0, 1):
            xr, yr = f'R{y}', f'proximate_logloss_y{y}'
            if not {xr, yr, 'family'}.issubset(x):
                continue
            v = x[['family', xr, yr]].dropna()
            for family, rows in v.groupby('family'):
                axes[y].scatter(rows[xr], rows[yr], label=family.upper(), alpha=.85)
            axes[y].set(xlabel=f'TREAD $R_{y}$',
                        ylabel=f'Observed true-class log loss, Y={y}',
                        title=f'RQ2: Class {y}')
            if len(v):
                axes[y].legend(fontsize=8)
                any_data = True
        if any_data:
            base = out / 'rq2_class_relative_validation'
            _save_chart(fig, base)
            figures.append(base.name)
        else:
            plt.close(fig)
    src = output / 'ablation_associations.csv'
    if src.exists() and src.stat().st_size:
        x = pd.read_csv(src)
        if {'family', 'type', 'diagnostic', 'spearman_rho'}.issubset(x):
            x = x[(x.family == 'overall') & x.type.isin(['RQ3_y0', 'RQ3_y1'])]
            if len(x):
                fig, ax = plt.subplots(figsize=(8, 3.6))
                categories = sorted(x.diagnostic.unique())
                for y in (0, 1):
                    v = (x[x.type == f'RQ3_y{y}'].set_index('diagnostic')
                         .spearman_rho.reindex(categories))
                    ax.scatter(np.arange(len(categories)) + (y-.5)*.19,
                               v, label=f'Y={y}')
                ax.axhline(0, lw=.8)
                ax.set_xticks(range(len(categories)), categories, rotation=35,
                              ha='right', fontsize=8)
                ax.set(ylabel='Target-project Spearman correlation',
                       title='RQ3: Diagnostic comparison')
                ax.legend()
                base = out / 'rq3_ablation_associations'
                _save_chart(fig, base)
                figures.append(base.name)
    src = output / 'sensitivity_results.csv'
    if src.exists() and src.stat().st_size:
        x = pd.read_csv(src)
        if {'dimension', 'U', 'q', 'alpha'}.issubset(x):
            x = x[(x.dimension == 'q_alpha') & x.U.notna()]
            if len(x):
                x = x.groupby(['q', 'alpha'], as_index=False).U.median()
                pivot = x.pivot(index='q', columns='alpha', values='U')
                fig, ax = plt.subplots(figsize=(5, 3.2))
                image = ax.imshow(pivot.to_numpy(), aspect='auto', vmin=0, vmax=1)
                ax.set_xticks(range(len(pivot.columns)),
                              [str(z) for z in pivot.columns])
                ax.set_yticks(range(len(pivot.index)),
                              [str(int(z)) for z in pivot.index])
                ax.set(xlabel=r'Threshold fraction $\alpha$',
                       ylabel='Neighbor order $q$',
                       title='Sensitivity: median source-proximity exceedance')
                fig.colorbar(image, ax=ax, label='Median $U$')
                base = out / 'rq3_q_alpha_sensitivity'
                _save_chart(fig, base)
                figures.append(base.name)
    LOG.info('Created %d figure sets in %s', len(figures), out)
    return figures


# ------------------------- BUILT-IN INTEGRITY TESTS -------------------------
def basic_self_tests():
    """Fast mathematical tests, not substitutes for actual benchmark runs."""
    s = np.array([[1., 4., 1.], [2., 5., 1.],
                  [4., np.nan, 1.], [5., 7., 1.]])
    trans = fit_transformer(s)
    assert int(trans.keep.sum()) == 2
    assert np.isfinite(trans.transform(np.array([[3., 6., 1.]]))).all()
    assert encode_labels(pd.Series([0, 1, 2, 0])).tolist() == [0, 1, 1, 0]
    x = np.array([[0., 0.], [0., 0.], [2., 0.], [0., 0.], [5., 0.]])
    y = np.array([0, 0, 0, 1, 1])
    a = source_ambiguity(x, y, 1e-8)
    assert abs(a[0] - .5) < 1e-12
    source = np.array([[0.], [0.], [0.], [10.], [10.], [10.]])
    sr, tr = reference_distances(source, np.array([[0.], [6.]]), 2)
    assert sr[0] == 0 and tr[0] == 0 and tr[1] > 0
    ds, dt = reference_distances(np.linspace(-1, 1, 40)[:, None],
                                 np.array([[100.], [101.]]), 5)
    covered, threshold, U = proximity(ds, dt, .05)
    assert not np.any(covered) and U == 1.
    _, risks = risk_rows([{'class': 0, 'group': 0, 'H': .3},
                          {'class': 1, 'group': 0, 'H': .2}], None, .5)
    assert np.isnan(risks[0]) and np.isnan(risks[1])
    assert 'yt' not in SourceTarget.__dataclass_fields__, 'Target labels leaked into diagnosis input'
    LOG.info('Built-in mathematical/leakage smoke checks passed')


# --------------------------- ONE-CLICK LAUNCHER -----------------------------
def _script_folder():
    return Path(__file__).resolve().parent


def locate_datasets(configured):
    """Prefer script location, then current directory; optionally open folder dialog."""
    proposed = Path(configured).expanduser()
    script = _script_folder()
    choices = ([proposed] if proposed.is_absolute() else
               [script / proposed, Path.cwd() / proposed,
                script.parent / proposed, Path.cwd().parent / proposed])
    for item in choices:
        if item.is_dir() and any(p.is_dir() and canon(p.name) in BENCHMARKS
                                 for p in item.iterdir()):
            return item.resolve()
    # GUI directory picker makes a double-click recoverable if folder is elsewhere.
    if sys.stdin.isatty() or (sys.platform.startswith('win')):
        try:
            import tkinter as tk
            from tkinter import filedialog
            app = tk.Tk()
            app.withdraw()
            folder = filedialog.askdirectory(title='Select Full Dataset-MSR folder (AEEEM, JIRA, PROMISE inside)')
            app.destroy()
            if folder:
                return Path(folder).resolve()
        except Exception as exc:
            LOG.warning('Dataset-folder picker unavailable: %s', exc)
    raise FileNotFoundError('Dataset folder not found. Place "Full Dataset-MSR" next to this '
                            'single Python file, with the dataset-family subfolders, '
                            'or edit FULL_DATASET_FOLDER at the top of the file.')


def configure_logger(output):
    output.mkdir(parents=True, exist_ok=True)
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(output / 'execution.log', encoding='utf-8')])


def _open_results(folder):
    try:
        if sys.platform.startswith('win'):
            os.startfile(str(folder))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(folder)])
        elif os.environ.get('DISPLAY'):
            subprocess.Popen(['xdg-open', str(folder)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    except Exception:
        pass


def single_click_main():
    import argparse
    parser = argparse.ArgumentParser(description='One-click TREAD MSR full pipeline; no CLI args required')
    parser.add_argument('--data-root', default=FULL_DATASET_FOLDER)
    parser.add_argument('--output', default='')
    parser.add_argument('--mode', choices=['full', 'smoke'], default='full')
    parser.add_argument('--seeds', type=int, default=FULL_DOWNSTREAM_SEEDS)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    script_dir = _script_folder()
    folder_name = RESULTS_FOLDER if args.mode == 'full' else 'TREAD_SMOKE_RESULTS'
    output = Path(args.output).expanduser().resolve() if args.output else script_dir / folder_name
    configure_logger(output)
    LOG.info('TREAD MSR SINGLE-FILE RELEASE / %s', time.strftime('%Y-%m-%d %H:%M:%S'))
    if RUN_BUILT_IN_TESTS and not args.skip_tests:
        basic_self_tests()
    if args.analyze_only:
        analysis_pipeline(output, Settings())
        LOG.info('Reconstructed RQ analysis from completed outputs')
    else:
        args.data_root = str(locate_datasets(args.data_root))
        args.output = str(output)
        args.resume = AUTO_RESUME_SAME_RUN and (output / 'run_manifest.json').exists()
        args.continue_on_error = False
        LOG.info('Dataset directory: %s', args.data_root)
        LOG.info('Mode: %s, downstream seeds: %d', args.mode, args.seeds)
        if args.mode == 'smoke':
            LOG.warning('SMOKE MODE IS NOT VALID FOR MANUSCRIPT RESULTS')
        run_all(args)
    if GENERATE_PUBLICATION_FIGURES and not args.no_figures:
        try:
            generate_figures(output)
        except Exception:
            LOG.exception('Optional figure rendering failed; validated numerical results remain intact')
            dump_json(output/'figure_generation_error.json',
                      {'error':traceback.format_exc()})
    # A compact landing document within the generated results (not a prerequisite).
    overview = [
        'TREAD — MSR TARGET-RELATIVE EVIDENCE DIAGNOSIS',
        f'Output directory: {output}',
        'Primary files:',
        '  target_diagnoses.csv             Class scores R0/R1 and U for each target',
        '  representation_profiles.csv      Per-component D, A, H, T, R scores',
        '  project_rq2_analysis.csv         Diagnostic-to-target-outcome data',
        '  diagnostic_associations.csv     RQ2 project-level correlations',
        '  ablation_associations.csv       RQ3 baseline/ablation correlations',
        '  oracle_affinity_comparison.csv   Retrospective class-conditioned oracle',
        '  fixed_source_summary.csv        Fixed historical pool, changing target',
        '  representation_project_validity.csv Component project concentration',
        '  external_diagnostic_associations.csv External baseline comparisons',
        '  model_variant_associations.csv   BIC, scaling, covariance alternatives',
        '  class_count_sensitivity.csv      >=10 target-class observation check',
        '  dataset_quality_audit.csv       Duplicate and conflicting metric vectors',
        '  sensitivity_results.csv         Parameter and seed sensitivity',
        '  integrity_report.json           Completion, missingness, failures',
        '  dataset_inventory.csv            Actual project counts/dimensions from CSVs',
        '  feature_alignment.csv           Exactly which numeric metrics are common',
        '  source_feature_preprocessing.csv Metric use per outer source-fitted run',
        '  excluded_columns.csv            Non-metric/invalid columns with reasons',
        '  family_feature_schemas.json     Used metric schema for each family',
        '  figures/                         Auto-generated figures (if available)',
        '  targets/                         Full per-project and per-module audit traces',
        '',
        'All labels used for outcome evaluation only after TREAD diagnosis.',
        'A successful script run does not establish empirical utility of TREAD.',
        'IMPORTANT: compare actual CSV metric counts and shared features to the manuscript table.',
        'SMOKE-mode results are NOT suitable for manuscript reporting.',
    ]
    (output/'START_HERE.txt').write_text('\n'.join(overview)+'\n',encoding='utf-8')
    LOG.info('ALL DONE. Results are in: %s', output)
    _open_results(output)



# ====================== MANUSCRIPT VALIDATION EXTENSIONS ======================
# Additional analyses below NEVER change the primary TREAD equations above.
# All target-label-based controls run after evaluate_target has completed.

_base_load_benchmarks = load_benchmarks
_base_build_diagnosis = build_diagnosis
_base_save_target = save_target
_base_analysis_pipeline = analysis_pipeline
_base_run_all = run_all


def load_benchmarks(root, strict=None):
    families, inventory = _base_load_benchmarks(root, strict)
    audits = []
    for family, ps in families.items():
        for p in ps:
            frame = pd.DataFrame(p.X)
            duplicated = frame.duplicated(keep=False)
            conflict = frame.assign(_defect_label=p.y).groupby(
                list(range(frame.shape[1])), dropna=False, sort=False
            )['_defect_label'].transform('nunique').to_numpy() > 1
            audits.append({'family': family, 'project': p.name,
                'shared_numeric_features': int(p.X.shape[1]),
                'duplicate_feature_rows': int(duplicated.sum()),
                'conflicting_label_feature_rows': int(conflict.sum()),
                'source_metric_missing_cells': int(np.isnan(p.X).sum())})
    load_benchmarks.quality_audit = pd.DataFrame(audits)
    # Original loader attaches feature reports to the currently bound function.
    return families, inventory


def project_concentration(view, groups, source_rows):
    results = []
    grouped = {0: [], 1: []}
    for row in source_rows:
        cls, g = int(row['class']), int(row['group'])
        mask = (view.ys == cls) & (groups == g)
        vals = np.bincount(view.env[mask], minlength=len(view.source_names)).astype(float)
        fractions = vals / max(1., vals.sum())
        active = fractions[fractions > 0]
        entropy = float(-(active * np.log(active)).sum())
        normalized = float(entropy / np.log(len(view.source_names))) if len(view.source_names)>1 else 0.
        row.update({'dominant_project_share': float(fractions.max()),
                    'project_entropy': entropy, 'normalized_project_entropy': normalized,
                    'effective_source_projects': float(np.exp(entropy))})
        grouped[cls].append(row)
        results.append({'class':cls,'group':g,'dominant_project_share':float(fractions.max()),
            'normalized_project_entropy':normalized,'effective_source_projects':float(np.exp(entropy))})
    association = []
    for cls in (0,1):
        mask = view.ys == cls
        association.append({'class': cls,
            'group_project_adjusted_rand':float(adjusted_rand_score(view.env[mask],groups[mask])),
            'group_project_adjusted_mutual_info':float(adjusted_mutual_info_score(view.env[mask],groups[mask])),
            'K':len(grouped[cls])})
    return results, association


def cross_project_source_reference(Xs, env, q):
    refs = np.empty(len(Xs))
    for group in np.unique(env):
        idx = np.flatnonzero(env==group)
        others = np.flatnonzero(env!=group)
        if len(others)<q:raise Inadmissible('too few other-project source neighbors')
        refs[idx] = NearestNeighbors(n_neighbors=q,n_jobs=-1).fit(Xs[others]).kneighbors(Xs[idx])[0][:,q-1]
    return refs


def other_project_ambiguity(Xs, ys, env, epsilon):
    values = np.empty(len(ys))
    for project in np.unique(env):
        for cls in (0,1):
            idx = np.flatnonzero((env==project)&(ys==cls))
            if not len(idx):continue
            pos = np.flatnonzero((env!=project)&(ys==cls))
            neg = np.flatnonzero((env!=project)&(ys!=cls))
            if not len(pos) or not len(neg):
                raise Inadmissible('cross-project ambiguity requires both classes in other projects')
            dp = NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(Xs[pos]).kneighbors(Xs[idx])[0][:,0]
            dn = NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(Xs[neg]).kneighbors(Xs[idx])[0][:,0]
            vals=dp/(dp+dn+epsilon)
            vals[(dp==0)&(dn==0)]=.5
            values[idx]=vals
    return values


def project_similarity_baseline(view, diag):
    if not diag['N_c']: return [{'class':cls,'method':'project_similarity_weighted','score':np.nan} for cls in (0,1)]
    out=[]
    scale=float(np.median(diag['source_ref'][diag['source_ref']>0])) if np.any(diag['source_ref']>0) else 1.
    X=diag['Xt'][diag['covered']]
    for cls in (0,1):
        scores=[]; hardness=[]
        for project in np.unique(view.env):
            inds=np.flatnonzero((view.env==project)&(view.ys==cls))
            if not len(inds):continue
            d=NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(diag['Xs'][inds]).kneighbors(X)[0][:,0]
            scores.append(-d/max(scale,1e-10))
            l=float(np.mean(diag['individual_loss'][inds]))
            dval=float(-np.expm1(-l))
            aval=float(np.mean(diag['individual_ambiguity'][inds]))
            hardness.append(diag['settings_lambda']*dval+(1-diag['settings_lambda'])*aval)
        if not scores:
            out.append({'class':cls,'method':'project_similarity_weighted','score':np.nan})
            continue
        responsibilities = np.exp(np.column_stack(scores)-logsumexp(np.column_stack(scores),axis=1,keepdims=True))
        w=responsibilities.mean(axis=0)
        out.append({'class':cls,'method':'project_similarity_weighted',
                    'score':float(np.dot(w,hardness))})
    return out


def domain_separability(Xs,Xt,seed,max_each=1000):
    # Target's DOMAIN identity is unlabeled target information, not defect labels.
    rng=np.random.default_rng(seed)
    i=rng.choice(len(Xs),min(max_each,len(Xs)),replace=False)
    j=rng.choice(len(Xt),min(max_each,len(Xt)),replace=False)
    X=np.vstack([Xs[i],Xt[j]])
    domain=np.r_[np.zeros(len(i),dtype=int),np.ones(len(j),dtype=int)]
    if min(len(i),len(j))<5:return np.nan
    oof=np.zeros(len(domain),float)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    for tr,te in cv.split(X,domain):
        classifier=LogisticRegression(max_iter=1000,C=1.0,random_state=seed)
        classifier.fit(X[tr],domain[tr])
        oof[te]=classifier.predict_proba(X[te])[:,1]
    auc=float(roc_auc_score(domain,oof))
    return float(max(0,2*auc-1))


def hard_target_score(diag):
    if not diag['N_c']:return {0:np.nan,1:np.nan}
    result={}
    for cls in (0,1):
        model=diag['models'][cls]
        choice=component_assign(model,diag['Xt'][diag['covered']])
        w=np.bincount(choice,minlength=model.n_components).astype(float)/len(choice)
        rows=[r for r in diag['source_rows'] if r['class']==cls]
        result[cls]=float(sum(w[r['group']]*r['H'] for r in rows))
    return result


def class_specific_proximity(diag,view,settings):
    out=[]
    for cls in (0,1):
        X=diag['Xs'][view.ys==cls]
        if len(X)<=settings.q:
            out.append({'class':cls,'status':'unavailable', 'U_class_support':np.nan})
            continue
        sr,tr=reference_distances(X,diag['Xt'],settings.q)
        c,thresh,u=proximity(sr,tr,settings.alpha)
        out.append({'class':cls,'status':'ok','U_class_support':u,'threshold':thresh,'N_close':int(c.sum())})
    return out


def transformed_sensitivity(view,diag,settings,variant):
    """An ALTERNATIVE diagnostic, never used to choose the primary method."""
    st=settings
    raw=view.Xs_raw; target=view.Xt_raw
    if variant=='signed_log':
        raw=np.sign(raw)*np.log1p(np.abs(raw))
        target=np.sign(target)*np.log1p(np.abs(target))
    elif variant=='covariance_reg_1e-4':
        st=replace(settings,gmm_reg=1e-4)
    elif variant=='bic_first':
        st=replace(settings,gmm_selection='bic_first')
    else:raise ValueError(variant)
    trans=fit_transformer(raw)
    xs=trans.transform(raw);xt=trans.transform(target)
    models,groups,rows,_=discover(xs,view.ys,view.env,st,settings.seed+25)
    if variant=='signed_log':
        a=source_ambiguity(xs,view.ys,settings.epsilon)
        alternative_view=replace(view,Xs_raw=raw,Xt_raw=target)
        losses,_=transfer_losses(alternative_view,st,st.seed+900)
    else:
        a=diag['individual_ambiguity'];losses=diag['individual_loss']
    src=source_profile(rows,groups,view.ys,losses,a,settings.lam)
    dr,dt=reference_distances(xs,xt,settings.q)
    accepted,_,u=proximity(dr,dt,settings.alpha)
    t=relevance(models,xt,accepted)
    _,r=risk_rows(src,t,settings.lam)
    return {'variant':variant,'status':'ok','U':u,'R0':r[0],'R1':r[1],
        'K0':models[0].n_components,'K1':models[1].n_components,'N_c':int(accepted.sum())}


def build_diagnosis(view,settings,seed=None):
    diag=_base_build_diagnosis(view,settings,seed)
    diag['settings_lambda']=settings.lam
    project_rows,relations=project_concentration(view,diag['source_group'],diag['source_rows'])
    diag['project_quality']=project_rows
    diag['project_group_associations']=relations
    diag['comparisons']+=project_similarity_baseline(view,diag)
    for cls in (0,1):
        base=float(np.mean(diag['individual_loss'][view.ys==cls]))
        diag['comparisons'].append({'class':cls,'method':'source_heldout_loss',
                    'score':base})
        diag['comparisons'].append({'class':cls,'method':'source_class_prevalence',
                    'score':float(np.mean(view.ys==cls))})
    diag['domain_separability']=domain_separability(diag['Xs'],diag['Xt'],settings.seed)
    diag['project_proximity']={}
    try:
        src_alt=cross_project_source_reference(diag['Xs'],view.env,settings.q)
        alt_c,alt_t,alt_u=proximity(src_alt,diag['target_distance'],settings.alpha)
        alt_rel=relevance(diag['models'],diag['Xt'],alt_c)
        _,alt_risk=risk_rows(diag['source_rows'],alt_rel,settings.lam)
        diag['project_proximity']={'status':'ok','U':alt_u,'threshold':alt_t,
            'N_c':int(alt_c.sum()),'R0':alt_risk[0],'R1':alt_risk[1],
            'proximity_agreement':float(np.mean(alt_c==diag['covered']))}
    except Inadmissible as exc:
        diag['project_proximity']={'status':'unavailable','reason':str(exc)}
    diag['class_proximity']=class_specific_proximity(diag,view,settings)
    diag['responsibility_concentration']=[]
    if diag['N_c']:
        for cls in (0,1):
            w=component_responsibilities(diag['models'][cls],diag['Xt'][diag['covered']])
            maximum=w.max(axis=1)
            diag['responsibility_concentration'].append({'class':cls,'mean_max_responsibility':float(maximum.mean()),
                'frac_max_gt_0_99':float(np.mean(maximum>.99)),
                'median_max_responsibility':float(np.median(maximum))})
    hard=hard_target_score(diag)
    for cls in (0,1):
        diag['comparisons'].append({'class':cls,'method':'hard_target_affinity','score':hard[cls]})
    try:
        ai=other_project_ambiguity(diag['Xs'],view.ys,view.env,settings.epsilon)
        alt_src=source_profile([dict(r) for r in diag['source_rows']],diag['source_group'],
              view.ys,diag['individual_loss'],ai,settings.lam)
        _,r=risk_rows(alt_src,diag['T'],settings.lam)
        diag['other_project_ambiguity']={'status':'ok','R0':r[0],'R1':r[1]}
    except Inadmissible as exc:
        diag['other_project_ambiguity']={'status':'unavailable','reason':str(exc)}
    diag['model_variant_sensitivity']=[]
    for variant in ('bic_first','covariance_reg_1e-4','signed_log'):
        try:
            diag['model_variant_sensitivity'].append(transformed_sensitivity(view,diag,settings,variant))
        except (Inadmissible,ValueError,FloatingPointError) as exc:
            diag['model_variant_sensitivity'].append({'variant':variant,'status':'unavailable','reason':str(exc)})
    return diag


def posthoc_oracle(view, truth, diag):
    """Only call AFTER the label-free diagnosis is complete and predictions exist."""
    y=np.asarray(truth)
    rows=[]
    for cls in (0,1):
        mask=diag['covered'] & (y==cls)
        if not mask.any():
            rows.append({'class':cls,'n_oracle_target':0,'status':'unavailable',
                         'R_oracle':np.nan,'R_deployable':diag['risk'][cls]})
            continue
        resp=component_responsibilities(diag['models'][cls],diag['Xt'][mask])
        weights=resp.mean(axis=0)
        group=[r for r in diag['source_rows'] if r['class']==cls]
        oracle=float(sum(weights[r['group']]*r['H'] for r in group))
        rows.append({'class':cls,'n_oracle_target':int(mask.sum()),'status':'ok',
              'R_oracle':oracle,'R_deployable':diag['risk'][cls],
              'oracle_minus_deployable':oracle-diag['risk'][cls]})
    return rows


def prediction_entropy(predictions, settings):
    p=pd.DataFrame(predictions)
    if p.empty:return []
    p=p[p.learner=='GBT'].copy()
    if p.empty:return []
    collected=[]
    for _,row in p.groupby('target_row'):
        probs=np.clip(row.p_defect.to_numpy(float),settings.probability_clip,1-settings.probability_clip)
        entropy=-probs*np.log(probs)-(1-probs)*np.log(1-probs)
        collected.append({'entropy':float(entropy.mean()),'disagreement':float(probs.std()),
            'source_proximate':bool(row.source_proximate.iloc[0])})
    return [{'scope':name,'mean_predictive_entropy':float(np.mean([r['entropy'] for r in collected if fn(r)])) if any(fn(r) for r in collected) else np.nan,
            'mean_prediction_disagreement':float(np.mean([r['disagreement'] for r in collected if fn(r)])) if any(fn(r) for r in collected) else np.nan}
        for name,fn in [('all',lambda r:True),('proximate',lambda r:r['source_proximate'])]]


def calibrated_downstream_probability(view,diag,settings,seed):
    """Supplementary source-grouped sigmoid calibration for primary downstream GBT."""
    ys,env=view.ys,view.env
    out=np.full(len(ys),np.nan)
    for group in np.unique(env):
        tr=np.flatnonzero(env!=group)
        te=np.flatnonzero(env==group)
        if len(np.unique(ys[tr]))<2:raise Inadmissible('insufficient classes for GBT calibration training')
        transformer=fit_transformer(view.Xs_raw[tr])
        clf=make_downstream('GBT',seed+int(group),settings)
        clf.fit(transformer.transform(view.Xs_raw[tr]),ys[tr])
        out[te]=raw_logit(clf,transformer.transform(view.Xs_raw[te]),settings.probability_clip)
    if not np.isfinite(out).all() or len(np.unique(ys))<2:raise Inadmissible('invalid downstream OOF logits')
    cal=LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000,random_state=seed)
    cal.fit(out.reshape(-1,1),ys)
    fit=make_downstream('GBT',seed,settings)
    fit.fit(diag['Xs'],ys)
    z=raw_logit(fit,diag['Xt'],settings.probability_clip)
    pred=cal.predict_proba(z.reshape(-1,1))[:,list(cal.classes_).index(1)]
    return np.clip(pred,settings.probability_clip,1-settings.probability_clip)


def save_target(folder,view,diag,outer,subsets,predictions):
    _base_save_target(folder,view,diag,outer,subsets,predictions)
    ids={'family':view.family,'target':view.target}
    for name,records in (
       ('group_project_validity.csv',diag['project_quality']),
       ('project_group_associations.csv',diag['project_group_associations']),
       ('responsibility_concentration.csv',diag['responsibility_concentration']),
       ('class_specific_proximity.csv',diag['class_proximity']),
       ('model_variant_sensitivity.csv',diag['model_variant_sensitivity']),
       ('predictive_uncertainty.csv',prediction_entropy(predictions,Settings()))):
        write_frame(folder/name,[{**ids,**r} for r in records])
    # Target truth is exposed ONLY by the already completed downstream predictions.
    if predictions:
        exemplar={int(r['target_row']):int(r['true_label']) for r in predictions if r['learner']=='GBT'}
        if len(exemplar)!=len(view.Xt_raw):raise AssertionError('Missing target labels in posthoc evaluation')
        truth=np.asarray([exemplar[i] for i in range(len(exemplar))],dtype=int)
        write_frame(folder/'oracle_affinity.csv', [{**ids,**r} for r in posthoc_oracle(view,truth,diag)])
        try:
            cp=calibrated_downstream_probability(view,diag,Settings(),Settings().seed)
            ls=binary_loss(truth,cp,Settings().probability_clip)
            write_frame(folder/'downstream_calibration_sensitivity.csv',[{**ids,'status':'ok',
                'learner':'GBT','n':len(truth),'calibrated_full_logloss':float(ls.mean()),
                'calibrated_proximate_logloss':float(np.mean(ls[diag['covered']])) if diag['N_c'] else np.nan}])
        except (Inadmissible,ValueError) as exc:
            write_frame(folder/'downstream_calibration_sensitivity.csv',[{**ids,'status':'unavailable','reason':str(exc)}])
    doc=json.loads((folder/'diagnosis.json').read_text())
    doc.update({'domain_separability':diag['domain_separability'],
        'project_proximity':diag['project_proximity'],
        'other_project_ambiguity':diag['other_project_ambiguity'],
        'class_specific_proximity':diag['class_proximity'],
        'K0':int(diag['models'][0].n_components),
        'K1':int(diag['models'][1].n_components)})
    dump_json(folder/'diagnosis.json',doc)


def extra_research_analysis(output,settings):
    output=Path(output)
    for src,dst in [('oracle_affinity.csv','oracle_affinity_comparison.csv'),
           ('project_group_associations.csv','representation_project_associations.csv'),
           ('group_project_validity.csv','representation_project_validity.csv'),
           ('class_specific_proximity.csv','class_specific_proximity.csv'),
           ('responsibility_concentration.csv','responsibility_concentration.csv'),
           ('model_variant_sensitivity.csv','model_variant_sensitivity.csv'),
           ('predictive_uncertainty.csv','predictive_uncertainty.csv'),
           ('downstream_calibration_sensitivity.csv','downstream_calibration_sensitivity.csv')]:
        read_completed(output,src).to_csv(output/dst,index=False)
    diagnosis=output/'target_diagnoses.csv'
    if not diagnosis.exists():return
    d=pd.read_csv(diagnosis)
    oracle=read_completed(output,'oracle_affinity.csv')
    outcome=output/'primary_outcomes.csv'
    if not outcome.exists() or oracle.empty:return
    primary=pd.read_csv(outcome)
    rows=[]
    for cls in (0,1):
        o=oracle[oracle['class']==cls].copy()
        loss=primary[(primary['class']==cls)&(primary.subset=='proximate')]
        joined=o.merge(loss[['family','target','trueclass_logloss','n']],on=['family','target'],how='inner')
        if joined.empty:continue
        joined['low_support']=joined['n']<settings.min_primary_class_n
        for family in ('overall',*sorted(joined.family.unique())):
            part=joined if family=='overall' else joined[joined.family==family]
            for score in ('R_oracle','R_deployable'):
                for valid_policy in ('all_nonempty','n_at_least_10'):
                    sub=part if valid_policy=='all_nonempty' else part[~part.low_support]
                    values=sub[[score,'trueclass_logloss']].dropna()
                    rho=float(spearmanr(values[score],values.trueclass_logloss).statistic) if len(values)>=4 and values[score].nunique()>1 and values.trueclass_logloss.nunique()>1 else np.nan
                    rows.append({'class':cls,'family':family,'score':score,'policy':valid_policy,
                        'n_targets':len(values),'spearman_rho':rho})
    write_frame(output/'oracle_affinity_associations.csv',rows)
    # Test whether any single target substantially changes the family association.
    merged=pd.read_csv(output/'project_rq2_analysis.csv') if (output/'project_rq2_analysis.csv').exists() else pd.DataFrame()
    influence=[]
    if not merged.empty:
        for family,part in merged.groupby('family'):
            for cls in (0,1):
                score=f'R{cls}'; goal=f'proximate_logloss_y{cls}'
                for excluded in part.target.unique():
                    sub=part[part.target!=excluded][[score,goal]].dropna()
                    rho=float(spearmanr(sub[score],sub[goal]).statistic) if len(sub)>=3 and sub[score].nunique()>1 and sub[goal].nunique()>1 else np.nan
                    influence.append({'family':family,'class':cls,'excluded_target':excluded,
                        'remaining_targets':len(sub),'spearman_rho':rho})
        write_frame(output/'leave_one_target_influence.csv',influence)
        # Family-centered exploratory association controls between-family shifts.
        centered=merged.copy()
        for cls in (0,1):
            sc=f'R{cls}'; loss=f'proximate_logloss_y{cls}'
            centered[sc+'_family_centered']=centered[sc]-centered.groupby('family')[sc].transform('mean')
            centered[loss+'_family_centered']=centered[loss]-centered.groupby('family')[loss].transform('mean')
        centered.to_csv(output/'family_centered_project_data.csv',index=False)
        center_assoc=[]
        for cls in (0,1):
            left=f'R{cls}_family_centered';right=f'proximate_logloss_y{cls}_family_centered'
            sub=centered[[left,right]].dropna()
            rho=float(spearmanr(sub[left],sub[right]).statistic) if len(sub)>=8 and sub[left].nunique()>1 and sub[right].nunique()>1 else np.nan
            center_assoc.append({'class':cls,'n_targets':len(sub),'family_centered_spearman':rho,
                 'interpretation':'descriptive within-family centered association; shared source projects'})
        write_frame(output/'family_centered_associations.csv',center_assoc)
        # Retrospective ranking utility; not an operational failure-probability threshold.
        triage=[]
        for cls in (0,1):
            score=f'R{cls}'; loss=f'proximate_logloss_y{cls}'
            candidates=[score,f'prior_ll_y{cls}','mmd']
            for family,sub in merged.groupby('family'):
                for candidate in candidates:
                    frame=sub[['target',candidate,loss]].dropna()
                    if len(frame)<4:continue
                    top_n=max(1,int(np.ceil(0.30*len(frame))))
                    predicted=set(frame.nlargest(top_n,candidate).target)
                    observed=set(frame.nlargest(top_n,loss).target)
                    triage.append({'family':family,'class':cls,'score':candidate,'n_targets':len(frame),
                        'top_fraction_requested':.30,'top_n':top_n,'top_n_overlap':len(predicted&observed),
                        'top_n_precision':len(predicted&observed)/top_n})
        write_frame(output/'retrospective_project_triage.csv',triage)
    # Class-specific outcome denominators and ten-module support sensitivity.
    if not merged.empty:
        observed=pd.read_csv(outcome)
        support_rows=[]
        for cls in (0,1):
            part=observed[(observed['class']==cls)&(observed.subset=='proximate')]
            table=merged.merge(part[['family','target','n']],on=['family','target'],how='left')
            for policy in ('all_nonempty','n_at_least_10'):
                filtered=table[table['n']>0] if policy=='all_nonempty' else table[table['n']>=settings.min_primary_class_n]
                for family in ('overall',*sorted(table.family.unique())):
                    sub=filtered if family=='overall' else filtered[filtered.family==family]
                    selected=sub[[f'R{cls}',f'proximate_logloss_y{cls}']].dropna()
                    rho=float(spearmanr(selected.iloc[:,0],selected.iloc[:,1]).statistic) if len(selected)>=4 and selected.iloc[:,0].nunique()>1 and selected.iloc[:,1].nunique()>1 else np.nan
                    support_rows.append({'class':cls,'family':family,'policy':policy,'n_targets':len(selected),
                        'spearman_rho':rho})
        write_frame(output/'class_count_sensitivity.csv',support_rows)
    variants=read_completed(output,'model_variant_sensitivity.csv')
    if not variants.empty and not merged.empty:
        rows=[]
        for cls in (0,1):
            for variant,part in variants.groupby('variant'):
                frame=part.merge(merged[['family','target',f'proximate_logloss_y{cls}']],on=['family','target'],how='inner')
                rows+=project_correlations(frame,[f'R{cls}'],f'proximate_logloss_y{cls}',settings,f'RQ3_{variant}_y{cls}')
        write_frame(output/'model_variant_associations.csv',rows)
    uncertainty=read_completed(output,'predictive_uncertainty.csv')
    if not uncertainty.empty and not merged.empty:
        unc=uncertainty[uncertainty.scope=='proximate'][['family','target','mean_predictive_entropy','mean_prediction_disagreement']]
        joined=merged.merge(unc,on=['family','target'],how='left')
        external=[]
        for cls in (0,1):
            external+=project_correlations(joined,['domain_separability','mean_predictive_entropy',
                'mean_prediction_disagreement','mmd',f'prior_ll_y{cls}',f'R{cls}'],
                f'proximate_logloss_y{cls}',settings,f'External_y{cls}')
        external+=project_correlations(joined,['domain_separability','mean_predictive_entropy','mmd','U'],
                'full_logloss',settings,'External_full')
        write_frame(output/'external_diagnostic_associations.csv',external)
    # Label changes are posthoc and cannot alter label-free TREAD scores.
    negative=[]
    for f in sorted(output.glob('targets/*/*/module_predictions.csv')):
        df=pd.read_csv(f)
        filtered=df[df.learner=='GBT'].drop_duplicates('target_row')
        if filtered.empty:continue
        flipped=1-filtered.true_label.to_numpy(dtype=int)
        original=filtered.true_label.to_numpy(dtype=int)
        probs=filtered.p_defect.to_numpy(dtype=float)
        negative.append({'family':f.parts[-3],'target':f.parts[-2],
            'n_target':len(original),'original_defect_rate':float(np.mean(original)),
            'flipped_defect_rate':float(np.mean(flipped)),
            'original_logloss':float(binary_loss(original,probs,settings.probability_clip).mean()),
            'flipped_label_logloss':float(binary_loss(flipped,probs,settings.probability_clip).mean()),
            'diagnosis_label_independent_by_interface':int('yt' not in SourceTarget.__dataclass_fields__),
            'diagnosis_recomputed':0,'interpretation':'posthoc outcome intervention; original diagnosis fixed'})
    write_frame(output/'evaluation_only_label_negative_control.csv',negative)

    # Prevalence-shift control uses target truth ONLY post hoc.
    counts=[]
    prediction_files=output.glob('targets/*/*/module_predictions.csv')
    for file in sorted(prediction_files):
        df=pd.read_csv(file)
        subset=df[df.learner=='GBT'].drop_duplicates('target_row')
        if len(subset):
            counts.append({'family':file.parts[-3], 'target':file.parts[-2],
                           'target_defect_prevalence':float(subset.true_label.mean())})
    if counts:
        prev=pd.DataFrame(counts)
        src=d.merge(prev,on=['family','target'],how='left')
        src['source_target_prevalence_difference']=src.target_defect_prevalence-src.source_defective_fraction
        src.to_csv(output/'retrospective_prevalence_checks.csv',index=False)


def analysis_pipeline(output,settings):
    _base_analysis_pipeline(output,settings)
    extra_research_analysis(output,settings)


def run_fixed_source(output,projects,settings,mode):
    """Evaluation-only separate experiment; identical historical source pool for each target."""
    jira=projects.get('jira',[])
    if len(jira)<=5:
        write_frame(Path(output)/'fixed_source_summary.csv',[{'status':'unavailable',
            'reason':'JIRA requires more than five projects for fixed-source targets'}])
        return
    ordered=sorted(jira,key=lambda p:canon(p.name))
    source=ordered[:5]
    remaining=ordered[5:]
    if mode=='smoke':remaining=remaining[:1]
    rows=[]
    for target in remaining:
        all_ps=source+[target]
        view=make_view('jira',all_ps,len(all_ps)-1)
        try:
            diagnosis=build_diagnosis(view,settings)
            # Identical source records, preprocessing, GMMs, and hardness across targets.
            outer,sub,pred=evaluate_target(view,target.y,diagnosis,settings,settings.downstream_seeds)
            folder=Path(output)/'fixed_source'/'targets'/target.name
            save_target(folder,view,diagnosis,outer,sub,pred)
            prim=[r for r in sub if r['learner']=='GBT' and r['subset']=='proximate']
            record={'family':'jira','target':target.name,'status':'ok',
                'source_projects':'|'.join(p.name for p in source),
                'R0':diagnosis['risk'][0],'R1':diagnosis['risk'][1],
                'U':diagnosis['U'],'N_c':diagnosis['N_c']}
            for cls in (0,1):
                matched=[r for r in prim if r['class']==cls]
                record[f'proximate_logloss_y{cls}']=float(np.nanmean([r['trueclass_logloss'] for r in matched])) if matched and any(np.isfinite(r['trueclass_logloss']) for r in matched) else np.nan
                record[f'n_proximate_y{cls}']=int(matched[0]['n']) if matched else 0
            rows.append(record)
        except Inadmissible as exc:
            rows.append({'family':'jira','target':target.name,'status':'unavailable','reason':str(exc)})
    write_frame(Path(output)/'fixed_source_summary.csv',rows)
    # Audit equality of class hardness and source metric configuration in fixed-source analysis.
    profiles=[]
    for folder in (Path(output)/'fixed_source'/'targets').glob('*'):
        f=folder/'groups.csv'
        if f.exists():
            frame=pd.read_csv(f)
            profiles.append((folder.name,frame[['class','group','D','A','H','n_yg']].sort_values(['class','group']).reset_index(drop=True)))
    equal=all(profiles[0][1].equals(df) for _,df in profiles[1:]) if profiles else False
    dump_json(Path(output)/'fixed_source_integrity.json',{'sources':[p.name for p in source],
        'targets':[p.name for p in remaining],'identical_source_hardness':equal,
        'checked_source_profiles':len(profiles)})
    if len(profiles)>1 and not equal:raise AssertionError('Fixed-source profiles differ across targets')


def run_all(args):
    _base_run_all(args)
    output=Path(args.output)
    settings=Settings(downstream_seeds=(1 if args.mode=='smoke' else args.seeds))
    if args.mode=='smoke':
        settings=replace(settings,k_max=2,gmm_n_init=1,rf_trees=10,
            gmm_sensitivity_seeds=(),q_grid=(5,),alpha_grid=(.05,),
            lambda_grid=(.5,),mmd_sample_max=100,bootstrap_replicates=100)
    families,_=load_benchmarks(args.data_root)
    load_benchmarks.quality_audit.to_csv(output/'dataset_quality_audit.csv',index=False)
    run_fixed_source(output,families,settings,args.mode)
    analysis_pipeline(output,settings)
    dump_json(output/'validation_extensions_manifest.json',{
        'primary_equations_unchanged':True,
        'oracle_affinity':'Evaluation labels only after deployable diagnosis and predictions',
        'fixed_source':'First five JIRA projects ordered by canonical project name',
        'project_group_validity':'Group entropy, concentration, ARI, AMI',
        'proximity_sensitivity':'Project-held-out and class-specific comparisons',
        'alternative_models':'BIC-first, signed log scaling, 1e-4 covariance, hard target affinity',
        'externals':'Project similarity, source-only difficulty, domain separability, predictive entropy',
        'primary_outcomes':'True-class loss on proximate members of each target class',
        'analysis_unit':'Target project (seeds averaged)'
    })

def evaluate_when_unavailable(folder,view,target_labels,settings):
    """Full-target outcome data without imputing an unavailable R0/R1 diagnosis."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if len(np.unique(view.ys))<2:raise Inadmissible('downstream training requires both source classes')
    transform=fit_transformer(view.Xs_raw)
    xs=transform.transform(view.Xs_raw)
    xt=transform.transform(view.Xt_raw)
    covered=np.ones(len(xt),dtype=bool)
    distances=np.full(len(xt),np.nan)
    try:
        dr,dt=reference_distances(xs,xt,settings.q)
        covered,_,_=proximity(dr,dt,settings.alpha)
        distances=dt
    except Inadmissible:
        # No invented proximity result: all-target predictions still remain valid.
        pass
    surrogate={'Xs':xs,'Xt':xt,'covered':covered,'target_distance':distances}
    outer,subsets,predictions=evaluate_target(view,target_labels,surrogate,settings,settings.downstream_seeds)
    ids={'family':view.family,'target':view.target}
    for name,rows in [('full_metrics.csv',outer),('subset_metrics.csv',subsets),
                      ('module_predictions.csv',predictions)]:
        write_frame(folder/name,[{**ids,**r} for r in rows])


# ====================== END VALIDATION EXTENSIONS =======================


if __name__ == '__main__':
    try:
        single_click_main()
    except BaseException:
        # A double-click run should never hide failure from the researcher.
        import traceback as _traceback
        _traceback.print_exc()
        print('\nTREAD DID NOT COMPLETE. Read execution.log and any fatal_error.json files.',
              flush=True)
        if sys.stdin.isatty() and '--no-pause' not in sys.argv:
            try:
                input('Press Enter to close this window...')
            except (EOFError, KeyboardInterrupt):
                pass
        raise
