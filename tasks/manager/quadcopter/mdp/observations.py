from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass


@configclass
class QuadObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        # Estado base
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity) # Importante para saber dónde es "abajo"
        
        # El comando que debe seguir (para que el agente sepa qué hacer)
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = PolicyCfg()