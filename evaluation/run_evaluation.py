"""Re-evaluate the shared recommendation snapshots. Run from any directory."""
from pathlib import Path
import sys, zipfile, argparse, subprocess, io, hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-zip', type=Path,
                        help='Optional local ZIP to also evaluate all historical content experiments')
    args = parser.parse_args()
    folder = ROOT / 'evaluation'
    train = pd.read_csv(ROOT / 'outputs/train.csv')
    test = pd.read_csv(ROOT / 'outputs/test.csv')
    keys = ['user_id','artist_name']
    assert not train.duplicated(keys).any() and not test.duplicated(keys).any()
    assert train.merge(test, on=keys).empty
    assert set(train.user_id) == set(test.user_id)
    assert set(test.artist_name) <= set(train.artist_name)
    cases = [('popular','baseline',pd.read_csv(ROOT/'outputs/recs_popular.csv'))]
    for method, filename in [('all','recs_memory_cf.csv'),('top50','recs_memory_cf_selected.csv')]:
        cases.append(('memory_cf',method,pd.read_csv(ROOT/'outputs'/filename)))
    content = pd.read_csv(ROOT/'content_based_result/content_based_recs.csv')
    assert set(content.method) == {'cosine_learned'}
    cases.append(('content_based','s_log_cosine_learned',content))
    if args.content_zip:
        with zipfile.ZipFile(args.content_zip) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith('.csv') and not name.startswith('__MACOSX/'):
                    tag = Path(name).stem.removeprefix('content_based_recs_')
                    for method, recs in pd.read_csv(archive.open(name)).groupby('method'):
                        cases.append((method,tag,recs))
    results=[]
    for method,setting,recs in cases:
        assert not recs[['user_id','rank','artist_name']].isna().any().any()
        assert not recs.duplicated(['user_id','artist_name']).any()
        assert recs.merge(train,on=['user_id','artist_name']).empty
        assert set(recs.user_id)==set(test.user_id)
        assert set(recs.artist_name)<=set(train.artist_name)
        assert recs.groupby('user_id')['rank'].apply(lambda x: sorted(x)==list(range(1,11))).all()
        p,r=evaluate(recs,test,k=10)
        truth = test.groupby('user_id').artist_name.apply(set).to_dict()
        hits = [len(set(group.artist_name) & truth[user]) for user,group in recs.groupby('user_id')]
        results.append(dict(method=method,setting=setting,precision_at_10=p,recall_at_10=r,
                            mean_hits_at_10=sum(hits)/len(hits),users=len(hits),
                            users_with_hits=sum(h>0 for h in hits),
                            user_hit_rate=sum(h>0 for h in hits)/len(hits),
                            distinct_recommended_artists=recs.artist_name.nunique(),
                            catalog_coverage=recs.artist_name.nunique()/train.artist_name.nunique()))
    table = pd.DataFrame(results)
    table.iloc[:4].to_csv(folder/'results/final_metrics.csv',index=False,encoding='utf-8-sig')
    if args.content_zip:
        table.iloc[4:].to_csv(folder/'results/content_experiment_metrics.csv',index=False,encoding='utf-8-sig')
    paths = ['outputs/train.csv','outputs/test.csv','evaluate.py','outputs/recs_popular.csv',
             'outputs/recs_memory_cf.csv','outputs/recs_memory_cf_selected.csv',
             'content_based_result/content_based_recs.csv']
    metadata = dict(source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                    train_rows=len(train),test_rows=len(test),users=train.user_id.nunique(),
                    k=10,aggregation='macro average over all test users',
                    input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (folder/'results/final_evaluation_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(table.to_string(index=False))

if __name__=='__main__':main()
