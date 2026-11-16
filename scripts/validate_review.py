from pathlib import Path
import hashlib, io, json, platform, subprocess, sys, time, traceback
from unittest.mock import patch
import argparse
parser=argparse.ArgumentParser(description='Full review validation; prepare the dataset first or supply --source-project.')
parser.add_argument('--source-project',type=Path)
args=parser.parse_args()
PROJECT=Path(__file__).resolve().parents[1]
BASE=PROJECT
SOURCE=args.source_project.resolve() if args.source_project else PROJECT
OUT = PROJECT / 'evidence'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT))
import pandas as pd
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
torch.set_num_threads(2)
checks=[]
def check(name, condition, actual=None):
    checks.append(dict(name=name,status='PASS' if condition else 'FAIL',actual=actual))
    print(checks[-1]['status'],name,flush=True)
def command(args, name):
    p=subprocess.run([sys.executable,*map(str,args)],cwd=PROJECT,capture_output=True,text=True,encoding='utf-8')
    (OUT/name).write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    return p
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def app(): return AppTest.from_file(str(PROJECT/'streamlit_app.py'),default_timeout=120).run(timeout=120)
def click(a,label):
    next(b for b in a.button if b.label==label).click()
    return a.run(timeout=120)
def upload(a,data,name='sample.png'):
    a.get('file_uploader')[0].set_value((name,data,'image/png'))
    return a.run(timeout=120)
try:
    p=command(['-m','pytest','tests/test_core.py','-q','-p','no:cacheprovider'],'core_tests.log')
    check('TC01-TC05 core suite',p.returncode==0,p.stdout.strip())
    p=command(['scripts/audit_isolation.py'],'isolation_audit.json')
    check('TC06 packaged checkpoint isolation audit',p.returncode==0)
    parents=pd.read_csv(SOURCE/'data/manifest.csv'); lines=pd.read_csv(SOURCE/'data/line_manifest.csv')
    joined=lines.merge(parents[['filename','split']],left_on='parent_filename',right_on='filename',suffixes=('_line','_parent'),how='left')
    check('TC06 parent counts and uniqueness',parents.filename.is_unique and parents.split.value_counts().to_dict()=={'train':1680,'test':240,'validation':123},parents.split.value_counts().to_dict())
    mismatch=int((joined.split_line!=joined.split_parent).sum())
    crossing=int((lines.groupby('source_group').split.nunique()>1).sum())
    check('TC06 crop split inheritance and group isolation',len(lines)==6019 and mismatch==0 and crossing==0,dict(crops=len(lines),mismatches=mismatch,crossing_groups=crossing))
    buckets={}
    for row in parents.itertuples():
        h=sha(SOURCE/'data/images'/row.filename); buckets.setdefault(h,set()).add(row.split)
    duplicates=sum(len(v)>1 for v in buckets.values())
    check('TC06 exact-image cross-split SHA256 check',duplicates==0,dict(cross_split_duplicate_hashes=duplicates))
    baseline=json.loads((PROJECT/'outputs/baseline_seed_2026/evaluation_test/metrics.json').read_text())
    check('TC07 baseline saved evaluation verified',baseline['samples']==240 and abs(baseline['character_error_rate']-0.8793825437228016)<1e-12,baseline)
    # Use a temporary checkpoint so evaluation never overwrites the recorded run.
    import shutil
    fresh=PROJECT/'tmp/fresh_evaluation'; fresh.mkdir(parents=True,exist_ok=True)
    shutil.copy2(PROJECT/'outputs/crnn_main_seed_2026/best.pt',fresh/'best.pt')
    p=command(['scripts/evaluate_crnn.py','--checkpoint',fresh/'best.pt','--manifest',SOURCE/'data/line_manifest.csv','--split','test'],'heldout_evaluation.log')
    fresh_metrics=json.loads((fresh/'evaluation_test/metrics.json').read_text()) if p.returncode==0 else {}
    saved=json.loads((PROJECT/'outputs/crnn_main_seed_2026/evaluation_test/metrics.json').read_text())
    check('TC08 full held-out CRNN evaluation reproduces metrics',p.returncode==0 and fresh_metrics==saved,fresh_metrics)
    captured={}; original=MemoryMediaFileStorage.load_and_get_id
    def capture(storage,path_or_data,mimetype,kind,filename=None):
        fid=original(storage,path_or_data,mimetype,kind,filename)
        if filename: captured[filename]=storage._files_by_id[fid].content
        return fid
    with patch.object(MemoryMediaFileStorage,'load_and_get_id',capture):
        a=app(); check('TC10 app starts without exception',not a.exception)
        click(a,'Use example'); start=time.perf_counter(); click(a,'Recognize Modi text'); elapsed=time.perf_counter()-start
        result=a.session_state['recognition_result']
        check('TC09-TC10 actual demo recognition and line order',not a.exception and not a.error and result['line_count']==3 and result['combined_text']=='\n'.join(result['lines']),dict(lines=result['lines'],wall_seconds=round(elapsed,4),device=result['device']))
        txt=captured['modi_transliteration.txt'].decode('utf-8'); payload=json.loads(captured['modi_transliteration.json'])
        check('TC10 actual TXT JSON PNG downloads',txt==result['combined_text'] and payload['combined_text']==txt and len(payload['lines'])==3 and Image.open(io.BytesIO(captured['modi_detected_lines.png'])).size==result['overlay'].size)
        for name,data in captured.items(): (OUT/name).write_bytes(data)
        result['overlay'].save(OUT/'demo_overlay.png')
        upload(a,(PROJECT/'data/images/mode_00099.png').read_bytes(),'different.png')
        check('TC10 changed image clears stale result',len(a.text_area)==0 and len(a.get('download_button'))==0)
    a=app()
    upload(a,b'not an image'); check('TC11 corrupt image rejected',any('could not be read' in e.value for e in a.error))
    buf=io.BytesIO(); Image.new('RGB',(600,300),'white').save(buf,format='PNG')
    upload(a,buf.getvalue()); click(a,'Recognize Modi text')
    check('TC11 blank image rejected',not a.text_area and any('No readable text lines' in e.value for e in a.error))
    raw=(PROJECT/'data/images/mode_00001.png').read_bytes(); boundary=raw+b'\0'*(25*1024*1024-len(raw))
    upload(a,boundary); check('TC12 25 MiB input boundary accepted',not a.exception and not a.error)
    upload(a,boundary+b'\0'); check('TC12 25 MiB plus one byte rejected',any('larger than 25 MB' in e.value for e in a.error))
except Exception:
    check('Validation runner completes',False,traceback.format_exc())
finally:
    output=dict(date='2026-11-16',scope='Local packaged source and Streamlit AppTest; no external host certification',environment=dict(python=platform.python_version(),torch=torch.__version__,streamlit=__import__('streamlit').__version__),checks=checks,passed=sum(c['status']=='PASS' for c in checks),failed=sum(c['status']=='FAIL' for c in checks),checkpoint_sha256={n:sha(PROJECT/'outputs'/n/'best.pt') for n in ['baseline_seed_2026','crnn_main_seed_2026']},external_deployment='NOT EXECUTED: Docker unavailable; no separate host configured')
    (OUT/'review_validation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=output['passed'],failed=output['failed'])),flush=True)
