import json, glob, os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import fetch_tamarind_results as F
F.OUT_DIR='outputs/rfd3/native'

def prune(dest):
    for patt in ('*_full_data_sample_*.json','*.a3m','*_results.zip'):
        for f in glob.glob(os.path.join(dest,'**',patt), recursive=True):
            try: os.remove(f)
            except OSError: pass

jobs=['ifmhc_rfd3_native_6AMU','ifmhc_rfd3_native_6AM5']
# for f in glob.glob('outputs/rfd3/native/protenix/*.json'):
#     d=json.load(open(f)); jn=d.get('request',{}).get('jobName')
#     if jn and d.get('ok'): jobs.append(jn)

def proc(jn):
    model='rfdiffusion3'
    if F.already_downloaded(jn, model): return 'already'
    if F.get_status(jn)!='Complete': return 'not_complete'
    url,err=F.result_url(jn)
    if not url: return 'urlfail'
    try:
        dest,n=F.download_and_extract(jn, model, url); prune(dest); return 'downloaded'
    except Exception as e:
        return f'dlfail:{type(e).__name__}'

with ThreadPoolExecutor(max_workers=6) as ex:
    res=list(ex.map(proc, jobs))
print('lean fetch result:', dict(Counter(r.split(':')[0] for r in res)))
print('local protenix results now:', len(glob.glob('outputs/rfd3/native/protenix/native/*.downloaded')))
