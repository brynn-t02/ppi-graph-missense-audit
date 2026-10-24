import os,sys,json,subprocess,time,traceback
from pathlib import Path
B=Path('/Data/u40902/TYX/ppi_audit_20261024');E=B/'environment/runtime'
env=os.environ.copy();env.update(CONDA_PKGS_DIRS=str(B/'cache/conda'),PIP_CACHE_DIR=str(B/'cache/pip'),PIP_DISABLE_PIP_VERSION_CHECK='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
status={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'state':'running','commands':[]}
def save(): (B/'outputs/environment_status.json').write_text(json.dumps(status,indent=2)+'\n')
def run(cmd,timeout):
 status['commands'].append(cmd);save(); print('COMMAND',cmd,flush=True)
 p=subprocess.run(cmd,env=env,timeout=timeout)
 if p.returncode:raise RuntimeError('Command failed with exit '+str(p.returncode))
try:
 save()
 run(['/Data/u40902/miniconda3/bin/conda','create','--yes','--prefix',str(E),'--override-channels','--channel','https://repo.anaconda.com/pkgs/main','python=3.10.8','pip=22.3.1','setuptools=65.5.0','wheel=0.37.1'],600)
 run([str(E/'bin/python'),'-m','pip','install','--timeout','45','--retries','2','--index-url','https://pypi.org/simple','-r',str(B/'code/requirements-restored.txt')],1800)
 run([str(E/'bin/python'),'-m','pip','check'],60)
 p=subprocess.run([str(E/'bin/python'),'-m','pip','freeze'],env=env,text=True,capture_output=True,timeout=60,check=True);(B/'outputs/restored_pip_freeze.txt').write_text(p.stdout)
 status['state']='installed_not_runtime_validated'
except Exception:
 status['state']='failed';status['error']=traceback.format_exc();print(status['error'],flush=True)
finally:
 status['ended_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()
