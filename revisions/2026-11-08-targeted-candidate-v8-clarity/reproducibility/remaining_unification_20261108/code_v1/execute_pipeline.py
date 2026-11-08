"""Run the already authorized bounded protocol; no server or credential handling."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
B=Path(os.environ.get('PPI_UNIFICATION_ROOT','/root/autodl-tmp/ppi_unification_20261108'))
O=B/'outputs';O.mkdir(parents=True,exist_ok=True)
def update(phase,**kw):
    (O/'pipeline_status.json').write_text(json.dumps({'phase':phase,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw},indent=2)+'\n')
try:
    for script in ['run_unified.py','analyze_unified.py']:
        update('running',script=script,pid=os.getpid())
        subprocess.run([sys.executable,'-u',str(B/'code'/script)],check=True,cwd=B)
    update('complete',classifier_fits=39,bootstrap_repeats=2000)
except Exception:
    update('failed',traceback=traceback.format_exc());raise
