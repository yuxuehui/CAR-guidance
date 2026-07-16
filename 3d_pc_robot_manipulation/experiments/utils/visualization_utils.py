import numpy as np
import sapien
from typing import List, Optional
from mani_skill.utils.building.actors.common import build_sphere

def add_energy_center_markers(
    scene,
    energy_centers: List[List[float]],
    radius: float = 0.015,
    color: Optional[List[float]] = None,
    marker_prefix: str = "energy_center_",
):
    if color is None:
        color = [1.0, 0.0, 0.0, 0.7]

    markers = []
    for i, center in enumerate(energy_centers):
        center_array = np.array(center, dtype=np.float32)

        marker = build_sphere(
            scene=scene,
            radius=radius,
            color=color,
            name=f"{marker_prefix}{i}",
            body_type="kinematic",
            add_collision=False,
        )

        from sapien import Pose
        marker.set_pose(Pose(p=center_array))

        markers.append(marker)

    return markers

def hide_energy_center_markers(scene, marker_prefix: str = "energy_center_"):
    for name, actor in scene.actors.items():
        if name.startswith(marker_prefix):
            try:
                actor.hide_visual()
            except Exception as e:

                import warnings
                warnings.warn(f"无法隐藏标记 {name}: {e}")

def show_energy_center_markers(scene, marker_prefix: str = "energy_center_"):
    for name, actor in scene.actors.items():
        if name.startswith(marker_prefix):
            try:
                actor.show_visual()
            except Exception as e:

                import warnings
                warnings.warn(f"无法显示标记 {name}: {e}")

def remove_energy_center_markers(scene, marker_prefix: str = "energy_center_"):

    markers_to_remove = []
    for name, actor in scene.actors.items():
        if name.startswith(marker_prefix):
            markers_to_remove.append(actor)

    for marker in markers_to_remove:
        scene.remove_actor(marker)
