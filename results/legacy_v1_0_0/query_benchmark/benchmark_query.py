"""Reproduce the public DataFrame query measurement using supplied fixtures.

Requires the accompanying CLARA source repository and its Python dependencies.
The default only validates mappings/endpoints. --measure records fresh timings.
"""
from pathlib import Path
import argparse, os, sys, json, time, threading, hashlib, platform
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repository-root',type=Path,required=True)
p.add_argument('--fixture',type=Path,default=Path(__file__).parent/'resident_query_fixture.parquet')
p.add_argument('--mapping',type=Path,default=Path(__file__).parent/'current_lookup_mapping.parquet')
p.add_argument('--thresholds',type=Path,default=Path(__file__).parent/'width_thresholds.parquet')
p.add_argument('--measure',action='store_true')
p.add_argument('--output',type=Path,default=Path('query_benchmark_output'))
args=p.parse_args()
scripts=args.repository_root/'src/evaluated/workspace/0427/0620/提交版本0703 Energy/一审/一审修订实验/测试CLARA/scripts'
if not scripts.is_dir():raise SystemExit('The supplied repository does not contain the evaluated six-action source tree.')
sys.path.insert(0,str(scripts.resolve()))
import numpy as np,pandas as pd,pyarrow as pa
import gefcom_six_action_selector_core_v1 as core
import source_tuning_facts as facts
pa.set_cpu_count(1);pa.set_io_thread_count(1)
raw=pd.read_parquet(args.fixture);mapping=pd.read_parquet(args.mapping);thresholds=pd.read_parquet(args.thresholds)
enc=facts.attach_raw_width_states(raw,thresholds=thresholds)
assert len(raw)==220 and len(mapping)==6600
probe=mapping[list(core.STATE_FIELDS)].copy();probe.insert(0,'event_id',[f'probe-{i}' for i in range(len(probe))])
np.testing.assert_array_equal(core.predict_clara_actions(probe,mapping),mapping.selected_action)
ix=pd.MultiIndex.from_frame(mapping[list(core.STATE_FIELDS)]).get_indexer(pd.MultiIndex.from_frame(enc[list(core.STATE_FIELDS)]))
assert (ix>=0).all()
actions=core.predict_clara_actions(enc,mapping)
np.testing.assert_array_equal(actions,mapping.selected_action.to_numpy()[ix])
def endpoints(frame,selected):
    return tuple(core.StreamingMetricAccumulator._selected_values(frame,selected,'candidate_'+edge) for edge in ['lower','upper'])
lower,upper=endpoints(enc,actions)
for i,a in enumerate(actions):
    assert lower[i]==enc.iloc[i][f'{a}__candidate_lower'] and upper[i]==enc.iloc[i][f'{a}__candidate_upper']
correct={'status':'PASS','state_mappings_checked':6600,'query_actions_checked':220,'selected_endpoints_checked':440,'measurement_performed':args.measure}
print(json.dumps(correct))
if not args.measure:sys.exit(0)
import psutil
from threadpoolctl import threadpool_info
args.output.mkdir(parents=True,exist_ok=False)
def write(name,obj):(args.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
write('correctness_receipt.json',correct)
write('environment.json',{'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(),'ram_bytes':psutil.virtual_memory().total,'packages':{'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__},'threadpools':[{k:v for k,v in x.items() if k!='filepath'} for x in threadpool_info()]})
banks={1:[enc.iloc[[i]].copy() for i in range(220)],220:[enc]};rawbanks={1:[raw.iloc[[i]].copy() for i in range(220)],220:[raw]}
def run(kind,f):
    if kind.startswith('width'):f=facts.attach_raw_width_states(f,thresholds=thresholds)
    a=core.predict_clara_actions(f,mapping)
    return a if kind=='lookup_only' else endpoints(f,a)
workloads=[(k,b,220 if b==1 else 100) for k in ['lookup_only','lookup_interval','width_encoding_lookup_interval'] for b in [1,220]]
for kind,batch,n in workloads:
    bank=rawbanks[batch] if kind.startswith('width') else banks[batch]
    for i in range(5):run(kind,bank[i%len(bank)])
load=[];stop=threading.Event()
def monitor():
    while not stop.is_set():load.append(psutil.cpu_percent(interval=.5))
t=threading.Thread(target=monitor,daemon=True);t.start();rows=[]
try:
    for r in range(220):
        for j in range(len(workloads)):
            kind,batch,n=workloads[(j+r)%len(workloads)]
            if r>=n:continue
            bank=rawbanks[batch] if kind.startswith('width') else banks[batch]
            f=bank[r%len(bank)];start=time.perf_counter_ns();result=run(kind,f);ns=time.perf_counter_ns()-start
            rows.append({'workload':kind,'batch_size':batch,'repeat':r,'elapsed_ns':ns})
            assert result is not None
finally:stop.set();t.join(timeout=2)
d=pd.DataFrame(rows);d.to_csv(args.output/'query_measurements.csv',index=False)
summary=[]
for (kind,batch),g in d.groupby(['workload','batch_size'],sort=False):
    q=np.quantile(g.elapsed_ns/1e6,[.5,.95,.99]);summary.append({'workload':kind,'batch_size':int(batch),'repeats':len(g),'median_ms':float(q[0]),'p95_ms':float(q[1]),'p99_ms':float(q[2])})
pd.DataFrame(summary).to_csv(args.output/'query_summary.csv',index=False)
write('receipt.json',{'status':'BACKGROUND_LOAD_DESCRIPTIVE','raw_calls':len(d),'slow_calls_removed':0,'system_cpu_load_percent':load,'idle_gate_passed':None,'scope':'Preloaded state lookup and selected interval retrieval; optional width classification. No forecasting, candidate generation, full history maintenance, sequential updates or policy fitting.','batch_scope':'Per-call total for 220 distinct resident queries, not standalone latency.','input_sha256':{key:hashlib.sha256(value.read_bytes()).hexdigest() for key,value in [('fixture',args.fixture),('mapping',args.mapping),('thresholds',args.thresholds)]}})
print(pd.DataFrame(summary).to_string(index=False))
