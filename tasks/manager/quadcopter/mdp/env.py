from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .observations import QuadObservationsCfg
from .actions import QuadActionsCfg
from .commands import QuadCommandsCfg
from .rewards import QuadRewardsCfg
from .terminations import QuadTerminationsCfg
from .events import QuadEventCfg
from .scene import QuadcopterSceneCfg


@configclass
class QuadcopterTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuración principal del entorno."""
    
    # 1. Escena
    scene = QuadcopterSceneCfg(num_envs=4096, env_spacing=5.0)
    
    # 2. Managers definidos arriba
    observations = QuadObservationsCfg()
    actions = QuadActionsCfg()
    commands = QuadCommandsCfg()
    rewards = QuadRewardsCfg()
    terminations = QuadTerminationsCfg()
    events = QuadEventCfg()
    
    def __post_init__(self):
        # Configuraciones generales de simulación
        self.decimation = 4  # Control a 50Hz si sim es 120Hz
        self.sim.render_interval = self.decimation
        self.sim.dt = 0.005 # 120Hz physics
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** 20
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** 21

        self.episode_length_s = 12.0