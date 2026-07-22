from sparc import load_model
import argparse
import tqdm
from pathlib import Path
import numpy as np
import soundfile as sf

parser = argparse.ArgumentParser()
#parser.add_argument("--rank",type=int,default=0)
#parser.add_argument("--n",type=int,default=1)
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--sparc_dir", type=str, )
parser.add_argument("--save_dir", type=str, )
parser.add_argument("--config_path", type=str, default=None)
#parser.add_argument("--batch_size",type=int,default=1)

if __name__ == "__main__":
    
    args = parser.parse_args()
    device = args.device
    sparc_dir = Path(args.sparc_dir)
    save_dir = Path(args.save_dir)
    spk_emb_dir = sparc_dir/"spk_emb"
    ft_dir = sparc_dir/"emasrc"
    save_dir.mkdir(parents=True, exist_ok=True)
    coder = load_model("en+",
                       config=args.config_path, 
                       device=device) 
    
    feats = sorted([f for f in ft_dir.glob("*.npy")])
    spk_embs = sorted([f for f in spk_emb_dir.glob("*.npy")])

    for feat_path, spk_emb_path in tqdm.tqdm(list(zip(feats, spk_embs))):

        save_name = str(feat_path).replace(str(ft_dir), "")
        save_name = save_dir/(Path(save_name).stem+".wav")
        
        if not Path(save_name).parent.exists():
            Path(save_name).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            feat = np.load(feat_path)
            spk_emb = np.load(spk_emb_path)
            wav = coder.decode(feat[:,:12], feat[:,12], feat[:,13], spk_emb)
            sf.write(save_name, wav, 16000)
        except Exception as e:
            print(f"Error processing {feat_path}, {spk_emb_path}: {e}")
        
            
            
        
        
        