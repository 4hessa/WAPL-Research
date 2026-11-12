"""UTF-8 byte model readiness: smoke training only, no dev/test model evaluation."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

class ByteTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.token=nn.Embedding(256,32);self.pos=nn.Embedding(64,32)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(32,4,64,dropout=0.,batch_first=True) for _ in range(2)])
        self.head=nn.Linear(32,256)
    def forward(self,x):
        if x.ndim!=2 or not 1<=x.shape[1]<=64:raise ValueError('Expected batch of 1–64 byte tokens')
        h=self.token(x)+self.pos(torch.arange(x.shape[1],device=x.device))
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device),diagonal=1)
        for block in self.blocks:h=block(h,src_mask=mask)
        return self.head(h)

def byte_chunks(blob):
    # Chunks may split a UTF-8 code point: inputs are bytes, not Unicode tokens.
    a=np.frombuffer(blob,dtype=np.uint8).astype(np.int64);n=len(a)//65
    return torch.from_numpy(a[:n*65].reshape(n,65).copy())

def byte_loss(model,batch):
    return nn.functional.cross_entropy(model(batch[:,:-1]).reshape(-1,256),batch[:,1:].reshape(-1))

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
    manifest=json.loads(Path('data/multilingual_sources.json').read_text())
    for language in ['ar','en']:
        source=manifest['languages'][language]['splits']['train'];path=Path(source['path']);blob=path.read_bytes()
        assert hashlib.sha256(blob).hexdigest()==source['sha256']
        data=byte_chunks(blob);assert len(data)>16;probe=data[:16]  # Training monitor only.
        for seed in [22000,22001]:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ByteTransformer();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            with torch.no_grad():before=float(byte_loss(model,probe))
            start=time.perf_counter();losses=[]
            for _ in range(32):
                batch=data[rng.integers(len(data),size=4)];opt.zero_grad();loss=byte_loss(model,batch);assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
            seconds=time.perf_counter()-start
            with torch.no_grad():after=float(byte_loss(model,probe))
            rows.append(dict(language=language,seed=seed,parameters=sum(p.numel() for p in model.parameters()),train_chunks=len(data),discarded_tail_bytes=len(blob)%65,training_monitor_before=before,training_monitor_after=after,all_losses_finite=bool(np.isfinite(losses).all()),steps=32,seconds=seconds))
    files=['experiments/multilingual_readiness.py','tools/prepare_multilingual.py','data/multilingual_sources.json','data/ud_ar_train.txt','data/ud_en_train.txt']
    result={'status':'training_only_readiness_not_efficacy','dev_or_test_model_evaluation':False,'rows':rows,'torch':torch.__version__,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/multilingual-readiness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
