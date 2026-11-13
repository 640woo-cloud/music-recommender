"""Re-evaluate the shared recommendation snapshots. Run from any directory."""
from pathlib import Path
import sys, zipfile, argparse, subprocess, io
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-zip', type=Path, required=True,
                        help='Local content_based_result.zip received from the content teammate')
    args = parser.parse_args()
    folder = ROOT / 'evaluation'
    train = pd.read_csv(ROOT / 'outputs/train.csv')
    test = pd.read_csv(ROOT / 'outputs/test.csv')
    assert train.merge(test, on=['user_id','artist_name']).empty
    cases = [('popular','baseline',pd.read_csv(ROOT/'outputs/recs_popular.csv'))]
    for method, filename in [('all','recs_memory_cf.csv'),('top50','recs_memory_cf_selected.csv')]:
        data = subprocess.check_output(['git','-C',str(ROOT),'show',
            f'177a8e3aa44471097ae75a7e01ed3de2f24429e0:outputs/{filename}'])
        cases.append(('memory_cf',method,pd.read_csv(io.BytesIO(data))))
    with zipfile.ZipFile(args.content_zip) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('.csv'):
                tag = Path(name).stem.removeprefix('content_based_recs_')
                for method, recs in pd.read_csv(archive.open(name)).groupby('method'):
                    cases.append((method,tag,recs))
    results=[]
    for method,setting,recs in cases:
        assert not recs.duplicated(['user_id','artist_name']).any()
        assert recs.merge(train,on=['user_id','artist_name']).empty
        assert set(recs.user_id)==set(test.user_id)
        assert set(recs.artist_name)<=set(train.artist_name)
        assert recs.groupby('user_id')['rank'].apply(lambda x: sorted(x)==list(range(1,11))).all()
        p,r=evaluate(recs,test,k=10)
        results.append(dict(method=method,setting=setting,precision_at_10=p,recall_at_10=r))
    pd.DataFrame(results).to_csv(folder/'results/recomputed_metrics.csv',index=False)

if __name__=='__main__':main()
