from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import torch
import isaaclab.envs.mdp as mdp


# Definimos funciones custom rápidas para IsaacLab si no existen predefinidas con esta lógica exacta
def track_lin_velocity_exp(env, command_name: str, sigma: float = 0.5, asset_cfg=SceneEntityCfg("robot")):
    """Recompensa exponencial por seguimiento de velocidad lineal."""
    # Obtener velocidad actual y deseada
    vel_curr = env.scene[asset_cfg.name].data.root_lin_vel_b  # Velocidad en base frame
    vel_target = env.command_manager.get_command(command_name)[:, :]
    
    # Calcular error cuadrático
    error = torch.sum(torch.square(vel_target - vel_curr), dim=1)
    return torch.exp(-error / sigma)

def track_ang_velocity_z_exp(env, command_name: str, sigma: float = 0.5, asset_cfg=SceneEntityCfg("robot")):
    """Recompensa exponencial por seguimiento de velocidad angular (Yaw)."""
    ang_curr = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2] # Solo Z
    ang_target = env.command_manager.get_command(command_name)[:, 2] # Asumiendo index 3 es yaw_rate
    
    error = torch.square(ang_target - ang_curr)
    return torch.exp(-error / sigma)


@configclass
class QuadRewardsCfg:
    """Gestor de recompensas diseñado para convergencia rápida."""
    
    # -- Recompensas Principales (Objetivo) --
    
    # Tracking de velocidad lineal (X, Y, Z)
    # Peso alto porque es la misión principal
    lin_vel_tracking = RewTerm(
        func=track_lin_velocity_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "sigma": 0.1} 
    )
    
    # Tracking de velocidad angular (Yaw)
    ang_vel_tracking = RewTerm(
        func=track_ang_velocity_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "sigma": 0.1}
    )
    
    # -- Regularización (Estabilidad y Suavidad) --
    
    # Penalización por acciones bruscas (suaviza el vuelo)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )
    
    # Penalización por aceleraciones lineales bruscas (ahorro de energía/vibración)
    lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.005
    )

    # Penalización por colisiones (con el suelo o entorno)
    #collision = RewTerm(
    #    func=mdp.undesired_contacts,
    #    weight=-2.0, # Penalización fuerte instantánea
    #    params={"threshold": 0.02}
    #)

    # "Bonus de vida": Pequeña recompensa constante por no estrellarse.
    # Ayuda en las primeras etapas para priorizar la supervivencia.
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.5
    )