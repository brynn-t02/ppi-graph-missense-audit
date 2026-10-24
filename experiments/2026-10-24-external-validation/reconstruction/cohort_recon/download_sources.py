from pathlib import Path
import urllib.request,json,datetime,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parent
SOURCES={
 'humsavar_2026_03.txt':'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/humsavar.txt',
 'uniprot_human_reviewed_canonical_2026_03.fasta.gz':'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29',
}
def get(item):
 name,url=item;p=ROOT/name
 with urllib.request.urlopen(url,timeout=120) as r:
  headers=dict(r.headers); total=0
  with p.with_suffix(p.suffix+'.part').open('wb') as f:
   while b:=r.read(1048576):
    total+=len(b)
    if total>50_000_000:raise RuntimeError('bounded download limit exceeded')
    f.write(b)
  p.with_suffix(p.suffix+'.part').rename(p)
 data={'url':url,'filename':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'headers':headers}
 (ROOT/(name+'.http.json')).write_text(json.dumps(data,indent=2))
 print(json.dumps(data),flush=True)
 return data
with concurrent.futures.ThreadPoolExecutor(2) as e:
 results=list(e.map(get,SOURCES.items()))
(ROOT/'download_manifest.json').write_text(json.dumps(results,indent=2))
