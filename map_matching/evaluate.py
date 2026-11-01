from localization import Localizer, make_query, ROOT
import cv2
import json
import numpy as np

def evaluate():
    engine = Localizer()
    rng = np.random.default_rng(42)
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    records=[]
    for idx,path in enumerate(engine.paths):
        for trial in range(3):
            query,truth = make_query(engine.images[idx],rng)
            result = engine.match(query)
            correct = result.get('image') == path.name and result['accepted']
            error = float(np.linalg.norm(np.array(result['polygon'])-truth['polygon'],axis=1).mean()) if correct else None
            record=dict(source=path.name,trial=trial,**truth,retrieved=result.get('image'),accepted=result['accepted'],correct=correct,corner_error_px=error,inliers=result.get('inliers',0))
            records.append(record)
            name=f'{idx:02d}_{trial}'
            cv2.imwrite(str(out/(name+'_query.png')),query)
            if idx < 3 and trial == 0 and result['accepted']:
                for key in ('localized','aligned','matches_image'):
                    cv2.imwrite(str(out/(name+'_'+key+'.png')),result[key])
            print(f'{name}: correct={correct}, corner error={error}',flush=True)
    blank=engine.match(np.full((150,150,3),255,np.uint8))
    summary=dict(mode='Exhaustive SIFT/FLANN/RANSAC',seed=42,total=len(records),correct=sum(r['correct'] for r in records),localized_within_5px=sum(r['correct'] and r['corner_error_px']<5 for r in records),blank_rejected=not blank['accepted'],records=records)
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    print({k:v for k,v in summary.items() if k!='records'})
    return summary

if __name__=='__main__':
    evaluate()
