"""The demo's MOTOR jobs: each body (bodies/<body>/build.npz, design.py) on walking (people-clips-made's) and squat
(squat.py), as sim_body.make writes a job (flat floor at y = 0), with Kimodo's foot contacts (kimodo/motions);
the person's kg is the body's mesh volume at 1000 kg/m3 (design.py's volume BMI), as generated bodies have no
measured mass.    score/.venv/bin/python jobs.py <score checkout> <demo dir>"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from score.stages import sim_body  # noqa: E402
W = pathlib.Path(sys.argv[2])
FLOOR = np.array([[[-30, 0, -30], [30, 0, 30], [30, 0, -30]], [[-30, 0, -30], [-30, 0, 30], [30, 0, 30]]], float)
clips = dict(np.load(W / 'clips-rig.npz'))
made = np.load(W / 'squat-rig.npz')
assert np.allclose(made['bind'], clips['bind']) and list(made['names']) == list(clips['names'])
clips['clip_squat'] = made['clip_squat']
CLIPS = ['walking', 'squat']
contacts = {clip: found for clip, found in sim_body.contacts_of(W / 'kimodo', clips).items() if clip in CLIPS}
model = str(sim_body.fetched())
designs = json.loads((W / 'bodies/designs.json').read_text())['bodies']
for body in designs:
    built = dict(np.load(W / 'bodies' / body / 'build.npz'))
    rig, size = sim_body.person_rig(clips, built, contacts)
    names = [str(n) for n in rig['names']]
    kept = sim_body.segments_of(rig)[0]
    height = sim_body.standing_height(rig)
    kg = round(designs[body]['volume'] * 1000.0, 1)
    for clip in CLIPS:
        folder = W / 'jobs' / f'{body}_{clip}'
        folder.mkdir(parents=True, exist_ok=True)
        worlds = sim_body.clip_frames(rig, clip, kept)
        np.savez_compressed(folder / 'targets.npz', world=worlds, ground=FLOOR, points=rig['points'],
                            faces=rig['faces'], weights=rig['weights'])
        job = {'rate': sim_body.RATE, 'gravity': 9.80665, 'model': model,
               'residual': {'force': sim_body.RESIDUAL_FORCE, 'moment': sim_body.RESIDUAL_MOMENT},
               'adjust': {'move most m': sim_body.MOVE_MOST, 'turn most deg': sim_body.FOLLOWED,
                          'follows deg': sim_body.FOLLOWED},
               'people': [{'name': f'{body}_{clip}', 'kg': kg, 'height': height,
                           'rig': {'names': names, 'parents': np.asarray(rig['parents']).tolist(),
                                   'bind': np.asarray(rig['bind']).tolist()},
                           'segments': [names[j] for j in kept], 'targets': str(folder / 'targets.npz')}]}
        (folder / 'job.json').write_text(json.dumps(job))
        print(f'{body}_{clip}', 'frames', len(worlds), 'height', round(height, 3), 'kg', kg,
              'finite', bool(np.isfinite(worlds).all()), 'contacts', clip in contacts, flush=True)
