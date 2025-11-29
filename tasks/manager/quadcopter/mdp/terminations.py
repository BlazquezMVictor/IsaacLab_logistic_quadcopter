from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg


@configclass
class QuadTerminationsCfg:
    """Cuándo termina el episodio."""
    # Si se inclina demasiado (Roll/Pitch > 80 grados), se ha estrellado
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={"limit_angle": 1.4} # ~80 grados
    )
    # Si el tiempo se acaba
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Si choca contra el suelo (altura < umbral)
    crash = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={
            "minimum_height": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])
        }
    )