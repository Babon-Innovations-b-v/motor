"""Random SOMA-shape coefficient vectors (sd units) and what each body measures: stature, leg ratio, volume; plus
how far the shaped bind's joint frames turn from the template's."""
import json, os, numpy as np, torch, soma.assets
from soma.body import SOMALayer
from scipy.spatial.transform import Rotation
T=os.environ['DEMO_DIR'].rstrip('/')+'/'
rig=dict(np.load(T+'clips-rig.npz'))
layer=SOMALayer(data_root=soma.assets.get_assets_dir(),device='cpu',identity_model_type='soma',lod='low',mode='dense')
names=list(layer.public_joint_names); kept=[names.index(str(x)) for x in rig['names']]
rn=[str(x) for x in rig['names']]; faces=rig['faces'].astype(int)
def shaped(c):
    layer.prepare_identity(torch.from_numpy(np.asarray(c)).float()[None])
    w=layer.public_bind_transforms_world(); w=(w[0] if w.ndim==4 else w).detach().numpy().astype(float)[kept]
    v=layer._cached_rest_shape; v=(v[0] if v.ndim==3 else v).detach().numpy().astype(float)
    return w,v
def measures(w,v):
    low=v[:,1].min(); h=v[:,1].max()-low
    tri=v[faces]; vol=abs(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
    thigh=[rn.index(s) for s in ('LeftLeg','RightLeg')]
    leg=np.mean(w[thigh,1,3])-low
    return {'stature':h,'leg':float(leg),'leg_ratio':float(leg/h),'volume':float(vol),'bmi_vol':float(vol*1000/h**2)}
if __name__=='__main__':
    print(rn)
    w0,v0=shaped(np.zeros(128)); tb=layer.bind_pose_world[layer.public_transform_joint_indices].detach().numpy()[kept]
    ang=[Rotation.from_matrix(tb[j,:3,:3].T@w0[j,:3,:3]).magnitude()*57.3 for j in range(len(kept))]
    print('zero vs template frame turn deg max %.2f'%max(ang), 'zero', measures(w0,v0))
    rng=np.random.default_rng(0); C=rng.standard_normal((300,128)); rows=[]
    for c in C:
        w,v=shaped(c); m=measures(w,v)
        m['turn_max']=max(Rotation.from_matrix(w0[j,:3,:3].T@w[j,:3,:3]).magnitude()*57.3 for j in range(len(kept)))
        rows.append(m)
    np.savez('sample.npz',C=C,**{k:np.array([r[k] for r in rows]) for k in rows[0]})
    for k in rows[0]:
        a=np.array([r[k] for r in rows]); print(k,'p5 %.4f p50 %.4f p95 %.4f'%tuple(np.percentile(a,[5,50,95])))
