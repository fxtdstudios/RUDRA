"""Render two known failures with fixed SDR display mapping; not independent QA."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from training.infer_sdr2hdr import load_models, predict_image
from training.sdr2hdr_dataset import load_rgb
from training.export_bench_pairs import degrade_like_eval
from rudra.delivery.bench import pu21_encode
from ui.export_formats import srgb_pixels


def main():
    out=Path('outputs/backbone_visual_review_20260927')
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    device=torch.device('cpu')
    paths=['checkpoints/sdr2hdr_shadow_v1.pt','outputs/robust_backbone_20260926_192336/candidate.pt']
    models=[load_models(p,None,device)[0].eval() for p in paths]
    rows=[json.loads(s) for s in Path('outputs/model_upgrade_audit_20260927/diagnostic_review.jsonl').read_text().splitlines()]
    chosen=[next(r for r in rows if r['condition']==condition) for condition in ('clean','hard')]
    results=[]
    for row in chosen:
        x=torch.from_numpy(load_rgb(row['sdr_path'],False)).permute(2,0,1)
        if row['condition']=='hard':
            x=degrade_like_eval(x,int(hashlib.sha256(row['asset_id'].encode()).hexdigest()[:7],16))
        ref=load_rgb(row['hdr_path'],True,ceiling=1000)*10000
        with torch.inference_mode():
            predictions=[predict_image(m,x[None],True,512,64,'all')[0].permute(1,2,0).numpy()*10000 for m in models]
        y=ref @ np.array([.2627,.678,.0593])
        masks={'shadows':y<12,'midtones':(y>=12)&(y<203),'highlights':y>=203}
        regional={}
        for name,mask in masks.items():
            regional[name]={'pixels':int(mask.sum()),'pu_mse':[
                float(np.mean((pu21_encode(p)[mask]-pu21_encode(ref)[mask])**2)) if mask.any() else None
                for p in predictions]}
        images=[np.rint(x.permute(1,2,0).numpy()*255).astype(np.uint8)]+[srgb_pixels(p) for p in [ref,*predictions]]
        canvas=Image.new('RGB',(1280,350),'#202020'); draw=ImageDraw.Draw(canvas)
        for i,(name,pixels) in enumerate(zip(['Input SDR','HDR reference (fixed SDR map)','Shipped (same map)','Candidate (same map)'],images)):
            image=Image.fromarray(pixels); image.thumbnail((320,310))
            canvas.paste(image,(320*i,35)); draw.text((320*i+5,8),name,fill='white')
        file=out/(row['condition']+'.png'); canvas.save(file)
        result=dict(asset_id=row['asset_id'],condition=row['condition'],regions=regional,image=str(file))
        results.append(result); print(json.dumps(result),flush=True)
    (out/'review.json').write_text(json.dumps(dict(results=results,checkpoints=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in paths],limitation='Two selected reused-validation failures; fixed SDR mapping, no calibrated HDR or independent evaluation.'),indent=2))


if __name__=='__main__': main()
