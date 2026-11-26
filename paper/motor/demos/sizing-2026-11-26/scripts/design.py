"""Each test body's SOMA-shape coefficients (sd units, 128): the least-norm vector whose body measures its target
stature, leg ratio (hip joint height over stature) and volume BMI (mesh volume x 1000 kg/m3 over stature squared),
found by Newton steps on the model itself through the sampled regression's Jacobian. Writes <out>/<body>/build.npz in
prop_rig's rig.npz form (bind, offsets, source_points) with the coefficients and measures, stood on the floor as
people/body.py own_bind_pose stands a fitted person (lowest vertex at y = 0)."""
import json, pathlib, sys
import numpy as np
from sample import shaped, measures, rig
OUT = pathlib.Path(sys.argv[1])
KEYS = ('stature', 'leg_ratio', 'bmi_vol')
TARGETS = {'tall': (1.93, 0.511, 24.5), 'heavy': (1.69, 0.511, 35.0), 'short': (1.50, 0.511, 24.5)}
sample = np.load(pathlib.Path(__file__).with_name('sample.npz'))
Y = np.stack([sample[k] for k in KEYS], 1)
X = np.c_[sample['C'], np.ones(len(Y))]
A = np.linalg.lstsq(X, Y, rcond=None)[0][:128].T  # 3 x 128
fit_r2 = [1 - ((Y[:, i] - X @ np.linalg.lstsq(X, Y[:, i], rcond=None)[0]) ** 2).sum() / ((Y[:, i] - Y[:, i].mean()) ** 2).sum() for i in range(3)]
print('regression r2', np.round(fit_r2, 4))
pinv = np.linalg.pinv(A)
names, parents = [str(n) for n in rig['names']], rig['parents']
summary = {}
for body, target in TARGETS.items():
    coeffs = np.zeros(128)
    if target is not None:
        for _ in range(8):
            got = measures(*shaped(coeffs))
            coeffs = coeffs + pinv @ (np.array(target) - np.array([got[k] for k in KEYS]))
    bind, points = shaped(coeffs)
    got = measures(bind, points)
    lift = -points[:, 1].min()
    points[:, 1] += lift
    bind[:, 1, 3] += lift
    offsets = np.array([bind[j, :3, 3] if parents[j] < 0 else
                        bind[parents[j], :3, :3].T @ (bind[j, :3, 3] - bind[parents[j], :3, 3]) for j in range(len(names))])
    folder = OUT / body
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'build.npz', names=np.array(names), parents=parents, bind=bind, offsets=offsets,
                        source_points=points, faces=rig['faces'], weights=rig['weights'], soma_shape=coeffs)
    summary[body] = {'target': target, **{k: round(float(v), 4) for k, v in got.items()},
                     'coeff norm sd': round(float(np.linalg.norm(coeffs)), 3),
                     'largest coeff sd': round(float(np.abs(coeffs).max()), 3)}
    print(body, summary[body])
(OUT / 'designs.json').write_text(json.dumps({'regression r2': [round(r, 4) for r in fit_r2], 'bodies': summary}, indent=1))
