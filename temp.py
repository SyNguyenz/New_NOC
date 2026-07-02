import numpy as np, os
d='results/inc21_sos_aslot_seed42'
yp=np.load(d+'/y_test_pred.npy'); yt=np.load(d+'/y_test_true.npy')
phip=np.load(d+'/phi_pred_test.npy')
print('phi_pred', phip.shape)
true_phi=None
for c in [os.environ.get('STR_DATA_DIR'),'data','.']:
    if c and os.path.exists(os.path.join(c,'phi_test.npy')):
        true_phi=np.load(os.path.join(c,'phi_test.npy')); print('TRUE phi_test from',c,true_phi.shape); break
if true_phi is None: print('no true phi_test.npy -> predicted phi only')
tn=yt.sum(1).astype(int)
def collect(phi, sel):
    c=[];m=[];cr=[];mr=[]
    for i in np.where(sel)[0]:
        order=np.argsort(-phi[i]); rank=np.empty(phi.shape[1],int); rank[order]=np.arange(phi.shape[1])
        for k in np.where(yt[i]==1)[0]:
            (c if yp[i,k]==1 else m).append(phi[i,k])
            (cr if yp[i,k]==1 else mr).append(rank[k])
    return map(np.array,(c,m,cr,mr))
for label,phi in [('PRED',phip)]+([('TRUE',true_phi)] if true_phi is not None else []):
    print('\n=== phi:',label,'(rank 0 = strongest; higher = fainter) ===')
    for grp,sel in [('NOC5',tn==5),('NOC4',tn==4),('NOC2-3',(tn>=2)&(tn<=3))]:
        c,m,cr,mr=collect(phi,sel)
        if len(m)==0: print('%-8s no missed'%grp); continue
        print('%-8s caught n=%d phi_med=%.4f rankMed=%.1f | MISSED n=%d phi_med=%.4f rankMed=%.1f'%(
            grp,len(c),np.median(c),np.median(cr),len(m),np.median(m),np.median(mr)))