"""T01 smoke test: the core dependencies import and MuJoCo actually steps.

If this fails on a clean ``uv sync`` the whole project is unbuildable, so it is
the first gate every later task inherits.
"""

from __future__ import annotations


def test_vfa_imports_and_version() -> None:
    import vfa

    assert isinstance(vfa.__version__, str)
    assert vfa.__version__


def test_core_dependencies_import() -> None:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    import numpy  # noqa: F401


def test_mujoco_builds_and_steps_trivial_model() -> None:
    import mujoco

    # A single free-falling box: the simplest model that still integrates.
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="box" pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.05" mass="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    z_start = float(data.qpos[2])
    for _ in range(100):
        mujoco.mj_step(model, data)

    # 100 steps of gravity must move the box down measurably.
    assert float(data.qpos[2]) < z_start
