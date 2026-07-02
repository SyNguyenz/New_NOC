"""Localize the CTRL drift from probe_lora_forensic: when we LoRA-fine-tune the frozen base on the train set
with a single ASL loss, test N5 drifts 0.788 -> ~0.75. WHERE does it come from?
  DEC-only : LoRA only on cls_decoder -> if this drops, the decoder is RE-FITTING train combos
             (combinatorial overfit at the readout = a REAL wall property the train gradient pulls toward).
  ENC-only : LoRA everywhere EXCEPT the decoder -> if this drops, it's encoder-feature perturbation
             (closer to a 'dropped multi-task balance' harness effect).
Both start at base (B zero-init -> pre-N5=0.788). 3 epochs is enough: full-LoRA dropped to 0.661 by ep1."""
import numpy as np, torch, time
import probe_lora_forensic as P

def ctrl(name_filter, name_exclude, tag, epochs=3, bs=48, lr=1e-3):
    torch.manual_seed(0); np.random.seed(0)
    m=P.fresh(name_filter=name_filter, name_exclude=name_exclude)
    asl=P.AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    tok,mkt,yt,N=P.tok,P.mkt,P.yt,P.N
    print(f"[{tag}] trainable={sum(p.numel() for p in params):,}  pre-N5={P.test_n5(m)[5]}", flush=True)
    traj=[]
    for ep in range(epochs):
        m.train(); perm=np.random.permutation(N)
        for s in range(0,N,bs):
            bi=perm[s:s+bs]; x=tok[bi].to(P.DEV); mk=mkt[bi].to(P.DEV); y=yt[bi].to(P.DEV)
            mb=mk.bool(); drop=(torch.rand_like(mb,dtype=torch.float)<0.15)&mb; kept=mb&~drop
            mk2=torch.where(kept.sum(1,keepdim=True)>=8,kept,mb).to(mk.dtype)
            lg=m(x,mk2)["logits_cls"]; loss=asl(lg,y)
            opt.zero_grad(); loss.backward(); opt.step()
        r=P.test_n5(m); traj.append(r[5]); print(f"[{tag}] ep{ep+1}: N5={r[5]} N4={r[4]}", flush=True)
    print(f"[{tag}] best={max(traj)}  (base=0.788)", flush=True); return max(traj)

t0=time.time()
print("base test N5=0.788\n--- DEC-only LoRA (decoder re-fit?) ---")
bd=ctrl("cls_decoder", None, "DEC-only")
print("\n--- ENC-only LoRA (encoder perturbation?) ---")
be=ctrl(None, "cls_decoder", "ENC-only")
print(f"\nLOCATE  base=0.788 | DEC-only best={bd} | ENC-only best={be} | ({time.time()-t0:.0f}s)")
print("read: the arm that DROPS below base carries the drift.")
