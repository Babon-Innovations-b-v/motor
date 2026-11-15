# motor/volumes/

SOMA's segment volumes (#70; pure numpy, its test runs in the score venv): the person's closed mesh at its bind
filled with 5 mm cells (`filled`, columns along y between surface crossings), each cell given to the bone group with
the nearest SOMA bone line (`shares`), each group's volume, centre and spread (`measured`), its mass, centre of mass
and inertia's trace at Dempster's densities brought to the person's mass (`weighed`).

- Densities are assumed: Dempster 1955 as Winter 2009 Table 4.1 tabulates them; Chandler et al. 1975 measured up to
  7% less (`DENSITY_RANGE`). Never tune a density to pass a gate.
- Measured 2026-10-28 on the template rig: 0.0603 m3 in 0.3 s; at 70 kg the thigh 7.9 kg, shin 3.3, foot and toes
  0.9, upper arm 2.9, forearm 1.2, hand 0.4, head and neck 5.7.
- A cell goes to its nearest bone line, not by skin weights: deep cells are far from every vertex.
- `slenderness` (2026-11-01): a group's girth for its length, its mean cross-section's width (its cells between
  `GIRTH_MIDDLE` of its stretch, clear of the cells its ends share) over the stretch's length; `sizing` widens a limb's
  bones by it.
