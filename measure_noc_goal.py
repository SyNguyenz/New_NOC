"""
measure_noc_goal.py — on the FULL real test, per NOC: oracle EM (ID ceiling), joint EM
(count-decoded), and count accuracy. For p5 also intrinsic(CORN) vs id_profile count.
Shows the binding constraint for the "oracle AND EM > 0.9 every NOC" goal.
"""
import json
from pathlib import Path
import numpy as np, torch
from train_set_transformer import topk_decode, DEVICE
from features.enrich import enrich_tokens
from measure_real_allcombos_multi import load_model  # reuse robust builder

D = Path("data_insilico_w")
tok3 = np.load(D / "tokens_test.npy").astype(np.float32)  # already full real test
mask = np.load(D / "mask_test.npy").astype(bool)
y = np.load(D / "y_test_set.npy").astype(int)
noc = np.clip(np.load(D / "noc_test.npy").astype(int), 1, 5)
en = enrich_tokens(tok3, mask)

def fwd(model, n_tok):
    P, C, CORN = [], [], []
    has_corn = False
    with torch.no_grad():
        for i in range(0, len(en), 256):
            o = model(torch.from_numpy(en[i:i+256, :, :n_tok]).to(DEVICE),
                      torch.from_numpy(mask[i:i+256]).to(DEVICE))
            P.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
            C.append(o["logits_card"].cpu().numpy())
            if "logits_corn" in o:
                has_corn = True
                from models.ordinal import corn_probs
                CORN.append(corn_probs(o["logits_corn"], 6).cpu().numpy())
    P = np.concatenate(P); C = np.concatenate(C)
    corn = np.concatenate(CORN) if has_corn else None
    return P, C, corn

def per_noc(em):
    return {k: float(em[noc == k].mean()) for k in [1, 2, 3, 4, 5]}

print(f"FULL real test n={len(y)}  per-NOC={[int((noc==k).sum()) for k in range(1,6)]}\n")
for arm in ["inc4_p1_stack", "inc4_p3_irm", "inc4_p5_noc_intrinsic"]:
    model, n_tok = load_model(arm)
    P, C, corn = fwd(model, n_tok)
    k_id = C.argmax(1) + 1
    oracle = per_noc((topk_decode(P, noc) == y).all(1))
    joint = per_noc((topk_decode(P, k_id) == y).all(1))
    cnt_id = {k: float((np.clip(k_id, 1, 5)[noc == k] == k).mean()) for k in [1, 2, 3, 4, 5]}
    print(f"=== {arm} ===")
    print(f"  {'':10}{'N1':>7}{'N2':>7}{'N3':>7}{'N4':>7}{'N5':>7}")
    print(f"  {'oracle':10}" + "".join(f"{oracle[k]:>7.3f}" for k in range(1, 6)))
    print(f"  {'joint-EM':10}" + "".join(f"{joint[k]:>7.3f}" for k in range(1, 6)))
    print(f"  {'count(id)':10}" + "".join(f"{cnt_id[k]:>7.3f}" for k in range(1, 6)))
    if corn is not None:
        k_in = corn.argmax(1) + 1
        cnt_in = {k: float((np.clip(k_in, 1, 5)[noc == k] == k).mean()) for k in [1, 2, 3, 4, 5]}
        print(f"  {'count(intr)':10}" + "".join(f"{cnt_in[k]:>7.3f}" for k in range(1, 6)))
    print()
