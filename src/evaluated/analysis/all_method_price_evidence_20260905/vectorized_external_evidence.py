"""Array-only equivalent of the sealed local CLARA evidence gathering loop."""
from __future__ import annotations

import numpy as np
import s09_minimum_external_core as core

def evidence_for_profile(*,states,mapped_levels,selected_levels,nu,beta):
    n=len(states);a=core.ACTIONS
    means=np.stack([f[[f'{k}__errf_mean' for k in a]].to_numpy(float) for f in mapped_levels])
    counts=np.stack([f.event_count.to_numpy(float) for f in mapped_levels])
    nu_values=np.full(n,float(nu)) if np.ndim(nu)==0 else np.asarray(nu,float)
    parent=means[-1].copy()
    current_mean=parent.copy();current_parent=parent.copy();current_shrunken=parent.copy()
    for level in range(len(mapped_levels)-2,-1,-1):
        mask=selected_levels<=level
        ns=counts[level,mask,None]
        vs=nu_values[mask,None]
        m=means[level,mask]
        p=parent[mask]
        denom=ns+vs
        with np.errstate(invalid='ignore',divide='ignore'):
            shrunk=np.where(denom>0,(ns*m+vs*p)/denom,m)
        current_mean[mask]=m;current_parent[mask]=p;current_shrunken[mask]=shrunk
        parent[mask]=shrunk
    rows=np.arange(n)
    def selected(suffix):
        arrays=np.stack([f[[f'{k}__{suffix}' for k in a]].to_numpy(float) for f in mapped_levels])
        return arrays[selected_levels,rows]
    risk_se=selected('risk_day_cluster_se')
    coverage=selected('covered_guardrail_mean')
    cold=states.rolling_state.astype(str).eq('cold_start').to_numpy()
    coverage[cold]=selected('covered_mean')[cold]
    return dict(risk_mean=current_mean,risk_parent=current_parent,risk_shrunken=current_shrunken,risk_se=risk_se,
                risk_score=current_shrunken+float(beta)*risk_se,coverage=coverage,tuwr=selected('tuwr_mean'),ard=selected('ard_mean'))
