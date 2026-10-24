from pathlib import Path
import subprocess,os,json,time
b=Path('/Data/u40902/TYX/ppi_audit_20261024');assert json.loads((b/'outputs/progress.json').read_text())['completed']==48
assert not (b/'outputs/analysis_pid.json').exists()
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
with (b/'logs/analysis.log').open('xb') as f:
 p=subprocess.Popen(['timeout','900',str(b/'environment/runtime/bin/python'),'-u',str(b/'code/analyze_matched_results.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd=b)
r={'pid':p.pid,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(b/'outputs/analysis_pid.json').write_text(json.dumps(r)+'\n');print(r)
