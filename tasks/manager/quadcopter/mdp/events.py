from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass


@configclass
class QuadEventCfg:
    """Eventos para aleatorización y resets."""
    # Resetear el estado del robot al inicio
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.5, 2.0)},
            "velocity_range": {},
        },
    )
    
    # Randomización de física (CRUCIAL para Sim-to-Real)
    # Aleatorizar masa del dron
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (0.8, 1.2), "operation": "scale"},
    )