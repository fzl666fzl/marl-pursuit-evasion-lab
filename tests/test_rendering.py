from __future__ import annotations

import numpy as np

from pursuit_lab.envs import make_env
from pursuit_lab.rendering import render_world_frame


def test_custom_render_frame_has_clear_legend_and_entity_colors() -> None:
    env = make_env(seed=0)
    try:
        env.reset(seed=0)
        frame = render_world_frame(env, size=360)
    finally:
        env.close()

    assert frame.shape == (360, 360, 3)
    assert frame.dtype == np.uint8
    assert (frame == np.array([220, 60, 60], dtype=np.uint8)).all(axis=2).any()
    assert (frame == np.array([40, 180, 80], dtype=np.uint8)).all(axis=2).any()
    assert (frame == np.array([120, 120, 120], dtype=np.uint8)).all(axis=2).any()
