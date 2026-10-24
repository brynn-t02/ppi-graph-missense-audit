import json,subprocess,os,time,traceback,re,importlib.metadata
from pathlib import Path
B=Path('/Data/u40902/TYX/ppi_audit_20261024');E=B/'environment/runtime/bin/python';p=B/'outputs/environment_status.json';s=json.loads(p.read_text());assert s['state']=='installed_not_runtime_validated';env=os.environ.copy();env['PIP_CACHE_DIR']=str(B/'cache/pip')
try:
 s['state']='finishing_base_python_pins';p.write_text(json.dumps(s,indent=2)+'\n')
 cmd=[str(E),'-m','pip','install','--timeout','45','--retries','2','--index-url','https://pypi.org/simple','-r',str(B/'code/requirements-restored.txt'),'-r',str(B/'code/requirements-conda-python-libs.txt')];s['commands'].append(cmd);subprocess.run(cmd,env=env,check=True,timeout=600)
 subprocess.run([str(E),'-m','pip','check'],env=env,check=True,timeout=60)
 z=subprocess.run([str(E),'-m','pip','freeze'],capture_output=True,text=True,check=True,env=env,timeout=60);(B/'outputs/restored_pip_freeze.txt').write_text(z.stdout)
 # Verify every requested pin against the installed distribution metadata.
 code="import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"
 actual=json.loads(subprocess.run([str(E),'-c',code],capture_output=True,text=True,check=True,timeout=30).stdout)
 norm=lambda x:re.sub('[-_.]+','-',x).lower();actual={norm(k):v for k,v in actual.items()};pins={}
 for name in ['requirements-restored.txt','requirements-conda-python-libs.txt']:
  for line in (B/'code'/name).read_text().splitlines():
   if '==' in line:k,v=line.split('==');pins[norm(k)]=v
 pins.update(torch='2.1.2+cu118',torchvision='0.16.2+cu118')
 conflicts={n:{'expected':v,'actual':actual.get(n)} for n,v in pins.items() if actual.get(n)!=v};assert not conflicts,conflicts
 s['pin_verification']={'checked':len(pins),'conflicts':conflicts};s['state']='installed_not_runtime_validated'
except Exception:
 s['state']='failed';s['error']=traceback.format_exc();print(s['error'],flush=True)
finally:p.write_text(json.dumps(s,indent=2)+'\n')
