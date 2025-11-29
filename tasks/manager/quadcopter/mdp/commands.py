from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class QuadCommandsCfg:
    """Configuración de los comandos de velocidad."""
    # Objetivo: Velocidad Lineal (X, Y, Z) y Angular (Yaw/Z)
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),  # Cambiar objetivo cada 2.0-4.0s
        debug_vis=True,
        heading_command=True,
        # Rangos: [Min_X, Max_X, Min_Y, Max_Y, Min_Z, Max_Z, Min_Roll, Max_Roll, Min_Pitch, Max_Pitch, Min_Yaw, Max_Yaw]
        # Nota: IsaacLab suele usar estructura (lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_z) para este generador específico si se configura así, 
        # pero aquí definimos rangos completos para asegurar claridad.
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
            heading=(-3.14, 3.14),
        )
    )