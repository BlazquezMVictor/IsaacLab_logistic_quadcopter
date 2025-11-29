from isaaclab.managers import ActionTermCfg, ActionTerm
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.assets import Articulation
import torch


class CrazyflieMixerAction(ActionTerm):
    """
    Convierte comandos de [Roll, Pitch, Yaw, Thrust] a fuerzas de motor individuales
    ajustadas a la geometría del Crazyflie 2.1.
    """

    _asset: Articulation

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 4, device=self.device)

        self.scale = torch.tensor([[0.001, 0.001, 0.0005, 0.4]], device=env.device)
        self.joint_names=[".*"]

        # --- CONSTANTES FÍSICAS DEL CRAZYFLIE ---
        self.arm_length = 0.0397  # 39.7mm del centro al motor
        self.k_drag = 0.006       # Coeficiente de arrastre del rotor (aprox)
        self.num_motors = 4
        
        # Índices de articulaciones (asegúrate que en tu USD coincidan)
        # Usualmente m1, m2, m3, m4 en orden X.
        self._joint_ids = env.scene[cfg.asset_name].find_joints(self.joint_names)[0]

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions):
        # 1. Escalar las acciones de la red neuronal [-1, 1]
        # Actions: [roll_torque, pitch_torque, yaw_torque, total_thrust]
        self._raw_actions = actions * self.scale
        
        roll_cmd = self._raw_actions[:, 0]
        pitch_cmd = self._raw_actions[:, 1]
        yaw_cmd = self._raw_actions[:, 2]
        
        # Thrust: Le sumamos la gravedad para facilitar el aprendizaje (feedforward)
        # 0.03kg * 9.81 = ~0.29N necesarios solo para flotar
        gravity_comp = 0.29 
        thrust_cmd = self._raw_actions[:, 3] + gravity_comp
        
        # 2. Mixer para Configuración en X (Crazyflie)
        # F = T / 4  +/- (moment / (4*L))
        # Simplificación vectorizada:
        
        self._processed_actions = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Coeficientes geométricos
        c_roll_pitch = 1.0 / (4.0 * self.arm_length)
        c_yaw = 1.0 / (4.0 * self.k_drag)
        c_thrust = 0.25
        
        # Motor 1 (Front-Right) - Nota: Verifica el orden de tus motores en el USD
        self._processed_actions[:, 0] = (c_thrust * thrust_cmd) - (c_roll_pitch * roll_cmd) - (c_roll_pitch * pitch_cmd) - (c_yaw * yaw_cmd)
        # Motor 2 (Rear-Right)
        self._processed_actions[:, 1] = (c_thrust * thrust_cmd) - (c_roll_pitch * roll_cmd) + (c_roll_pitch * pitch_cmd) + (c_yaw * yaw_cmd)
        # Motor 3 (Rear-Left)
        self._processed_actions[:, 2] = (c_thrust * thrust_cmd) + (c_roll_pitch * roll_cmd) + (c_roll_pitch * pitch_cmd) - (c_yaw * yaw_cmd)
        # Motor 4 (Front-Left)
        self._processed_actions[:, 3] = (c_thrust * thrust_cmd) + (c_roll_pitch * roll_cmd) - (c_roll_pitch * pitch_cmd) + (c_yaw * yaw_cmd)

        # 3. Clamping (Físico real)
        # El motor no puede dar fuerza negativa ni más de 0.15N (aprox 15g de empuje)
        self._processed_actions = torch.clamp(self._processed_actions, min=0.0, max=0.15)
        
        return self._processed_actions

    def apply_actions(self):
        # IsaacLab espera "Joint Efforts" (Fuerzas/Torques en las articulaciones)
        forces = self.processed_actions
        self._asset.set_joint_effort_target(forces, joint_ids=self._joint_ids)

@configclass
class CrazyflieMixerActionCfg(ActionTermCfg):
    class_type = CrazyflieMixerAction

@configclass
class QuadActionsCfg:
    rotors_vel = CrazyflieMixerActionCfg(asset_name="robot")
    #motor_efforts = mdp.JointEffortActionCfg(
    #    asset_name="robot",
    #    joint_names=[".*"],
    #    scale=0.001,
    #)