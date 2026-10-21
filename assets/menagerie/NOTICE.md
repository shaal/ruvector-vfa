# Third-party notice — MuJoCo Menagerie Franka Emika Panda

`franka_emika_panda/` is vendored, unmodified, from MuJoCo Menagerie:

- Upstream: <https://github.com/google-deepmind/mujoco_menagerie/tree/feadf76d42f8a2162426f7d226a3b539556b3bf5/franka_emika_panda>
- Commit: `feadf76d42f8a2162426f7d226a3b539556b3bf5`
- License: Apache License 2.0 (`franka_emika_panda/LICENSE`, copied verbatim)
- Copyright: the MuJoCo Menagerie authors (Google DeepMind). The model is
  derived from Franka Emika's public `franka_description` URDF.

## What was copied

`panda.xml`, `hand.xml`, `LICENSE`, and the whole `assets/` directory
(67 mesh files, every one referenced by `panda.xml`/`hand.xml`). Each file's git
blob hash equals the upstream blob at the commit above. The MJX variants,
`scene.xml`, `panda_nohand.xml`, `README.md`, `CHANGELOG.md`, and `panda.png`
were not copied.

## Modifications

None. Do not edit these files. The scene changes (torque motors, table, cube,
sites, sensors) are made in code by `src/vfa/sim/scene.py` through
`mujoco.MjSpec`. `tests/test_scene.py` pins the content with this tree hash
(SHA-256 over the sorted lines `<relative path>\0<sha256 of file>\n`):

```
1418e49508d92ec0a2ecf1c9f7a62d458b1d2ba940a19f980b1231e68b55f274
```

`.gitattributes` marks the directory `-text` so line-ending conversion cannot
change the bytes on checkout.
