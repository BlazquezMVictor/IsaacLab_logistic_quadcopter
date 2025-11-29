# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .cfg import QuadcopterEnvCfg


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Buffers de acción
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros_like(self._actions) # Para calcular action rate
        
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Buffer para guardar la velocidad del paso anterior
        self._last_root_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # --- DEFINICIÓN GEOMÉTRICA DE LOS ROTORES (Configuración en X) ---
        # Crazyflie convencional:
        # Motor 1: Front-Right (+X, -Y)
        # Motor 2: Rear-Right  (-X, -Y)
        # Motor 3: Rear-Left   (-X, +Y)
        # Motor 4: Front-Left  (+X, +Y)
        # (Los signos exactos dependen de tu USD, pero asumiremos X-Forward estándar)
        
        L = self.cfg.arm_length
        # Tensor de posiciones de los motores [4, 3] (x, y, z)
        # Asumimos que están en el plano Z=0 del cuerpo
        self.motor_positions = torch.tensor([
            [ L, -L, 0.0],  # M1 (Front-Right)
            [-L, -L, 0.0],  # M2 (Rear-Right)
            [-L,  L, 0.0],  # M3 (Rear-Left)
            [ L,  L, 0.0]   # M4 (Front-Left)
        ], device=self.device)
        
        # Direcciones de giro para el Torque de Yaw (Drag)
        # +1: Anti-Horario (CCW), -1: Horario (CW)
        # Config estándar: M1(CCW), M2(CW), M3(CCW), M4(CW) (puede variar según firmware)
        self.motor_dirs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)

        # Buffers para cálculos intermedios (evitar allocs en el loop)
        self._motor_forces = torch.zeros(self.num_envs, 4, device=self.device)

        # --- COMANDOS ---
        # [Vx, Vy, Vz, Yaw_Rate]
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel",
                "track_ang_vel",
                "penalty_lin_acc",
                "penalty_action_rate",
            ]
        }
        
        # Física del robot
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # 1. Guardar estado anterior (para penalizaciones)
        self._last_root_lin_vel_b = self._robot.data.root_lin_vel_b.clone()
        self._last_actions = self._actions.clone()
        
        # 2. Clampear acciones [-1, 1]
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # --- MIXER FÍSICO (CRAZYFLIE X-CONFIG) ---
        
        # Mapeos
        # Thrust: [0, 1]. Si la red saca -1 -> 0 thrust. Si saca 1 -> Max thrust.
        cmd_thrust = (self._actions[:, 0] + 1.0) / 2.0  
        
        # Reducimos la autoridad de giro al principio. 
        # Si esto es muy alto, el dron entra en barrena antes de aprender a flotar.
        gain_att = 0.03  # Reducido de 0.05 a 0.03 para más estabilidad inicial
        cmd_roll   = self._actions[:, 1] * gain_att
        cmd_pitch  = self._actions[:, 2] * gain_att
        cmd_yaw    = self._actions[:, 3] * gain_att

        # Feedforward de Gravedad:
        # El Crazyflie pesa ~30g => ~0.294 N.
        # Por motor: 0.294 / 4 = 0.0735 N.
        # Le damos un poco menos (0.07) para que la red tenga que poner un poquito de su parte.
        hover_force = 0.07

        # Fuerza máxima por motor (aprox 15g = 0.15N)
        max_thrust = 0.15

        # Mezcla (Mixer)
        # F = Thrust_base + Gravity_help +/- Roll +/- Pitch +/- Yaw
        # Nota: Ajusta signos si ves que correge al revés
        force_m1 = (cmd_thrust * max_thrust) - cmd_roll - cmd_pitch + cmd_yaw + hover_force
        force_m2 = (cmd_thrust * max_thrust) - cmd_roll + cmd_pitch - cmd_yaw + hover_force
        force_m3 = (cmd_thrust * max_thrust) + cmd_roll + cmd_pitch + cmd_yaw + hover_force
        force_m4 = (cmd_thrust * max_thrust) + cmd_roll - cmd_pitch - cmd_yaw + hover_force

        # Clamping físico (0 a Max)
        self._motor_forces[:, 0] = torch.clamp(force_m1, 0.0, max_thrust)
        self._motor_forces[:, 1] = torch.clamp(force_m2, 0.0, max_thrust)
        self._motor_forces[:, 2] = torch.clamp(force_m3, 0.0, max_thrust)
        self._motor_forces[:, 3] = torch.clamp(force_m4, 0.0, max_thrust)

        # --- APLICACIÓN AL CUERPO ---
        
        # 1. Empuje Total (Eje Z local)
        self._thrust[:, 0, 2] = torch.sum(self._motor_forces, dim=1)
        
        # 2. Momentos (Torques) usando brazos de palanca reales
        # Roll (Eje X) -> Par generado por distancia Y
        m_roll = (self.motor_positions[0, 1] * self._motor_forces[:, 0]) + \
                 (self.motor_positions[1, 1] * self._motor_forces[:, 1]) + \
                 (self.motor_positions[2, 1] * self._motor_forces[:, 2]) + \
                 (self.motor_positions[3, 1] * self._motor_forces[:, 3])

        # Pitch (Eje Y) -> Par generado por distancia -X
        m_pitch = (-self.motor_positions[0, 0] * self._motor_forces[:, 0]) + \
                  (-self.motor_positions[1, 0] * self._motor_forces[:, 1]) + \
                  (-self.motor_positions[2, 0] * self._motor_forces[:, 2]) + \
                  (-self.motor_positions[3, 0] * self._motor_forces[:, 3])

        # Yaw (Eje Z) -> Par aerodinámico
        m_yaw = self.cfg.torque_coefficient * (
            (self._motor_forces[:, 0] * self.motor_dirs[0]) + \
            (self._motor_forces[:, 1] * self.motor_dirs[1]) + \
            (self._motor_forces[:, 2] * self.motor_dirs[2]) + \
            (self._motor_forces[:, 3] * self.motor_dirs[3])
        )

        self._moment[:, 0, 0] = m_roll
        self._moment[:, 0, 1] = m_pitch
        self._moment[:, 0, 2] = m_yaw
        
    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # Aquí definimos qué ve la red neuronal
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,      # [N, 3] Velocidad Lineal Base
                self._robot.data.root_ang_vel_b,      # [N, 3] Velocidad Angular Base
                self._robot.data.projected_gravity_b, # [N, 3] Vector Gravedad (orientación)
                self._commands,                       # [N, 4] El comando deseado [Vx, Vy, Vz, Wz]
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 1. Tracking Lineal (Exponencial)
        # Error = || V_deseada - V_actual ||^2
        # Solo tomamos las primeras 3 componentes del comando para lin vel
        lin_vel_error = torch.sum(torch.square(self._commands[:, :3] - self._robot.data.root_lin_vel_b), dim=1)
        r_lin_vel = torch.exp(-lin_vel_error / 0.25) # Sigma = 0.25
        
        # 2. Tracking Angular (Yaw Rate) (Exponencial)
        # Comando index 3 es Yaw Rate. Estado index 2 es Z angular.
        ang_vel_error = torch.square(self._commands[:, 3] - self._robot.data.root_ang_vel_b[:, 2])
        r_ang_vel = torch.exp(-ang_vel_error / 0.25)

        # 3. Penalizaciones (Regularización)
        # Aceleración lineal brusca (vibración)
        lin_acc = torch.sum(torch.square(self._robot.data.root_lin_vel_b - self._last_root_lin_vel_b) / self.step_dt, dim=1)
        # Cambio brusco de acciones (Action Rate)
        action_rate = torch.sum(torch.square(self._actions - self._last_actions), dim=1)

        rewards = {
            "track_lin_vel": r_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel": r_ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "penalty_lin_acc": lin_acc * self.cfg.lin_acc_penalty_scale * self.step_dt,
            "penalty_action_rate": action_rate * self.cfg.action_rate_penalty_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Muerte por choque o alejarse demasiado en altura
        # Permitimos de 0.05m a 4.0m de altura
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.05, self._robot.data.root_pos_w[:, 2] > 4.0)
        
        # Muerte por mala orientación (opcional, ayuda al principio)
        # Si la gravedad proyectada en Z baja de 0.2, está casi invertido
        flipped = self._robot.data.projected_gravity_b[:, 2] < 0.2
        died = torch.logical_or(died, flipped)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        env_ids_count = len(env_ids)

        # Logging de episodio finalizado
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # 1. Resetear física del robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # Spread out resets
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0

        # Resetear el buffer de velocidad anterior para los entornos reiniciados
        self._last_root_lin_vel_b[env_ids] = 0.0

        # 2. GENERAR NUEVOS COMANDOS (AQUÍ ESTÁ TU CONTROL TOTAL)
        # Queremos velocidades aleatorias para entrenar la red
        
        # Vx, Vy: Rango [-1.0, 1.0]
        self._commands[env_ids, 0] = torch.rand(env_ids_count, device=self.device) * 4.0 - 1.0
        self._commands[env_ids, 1] = torch.rand(env_ids_count, device=self.device) * 4.0 - 1.0
        
        # Vz: Rango [-0.5, 0.5] (Velocidad vertical)
        self._commands[env_ids, 2] = torch.rand(env_ids_count, device=self.device) * 2.0 - 0.5
        
        # Yaw Rate: Rango [-1.5, 1.5] rad/s
        self._commands[env_ids, 3] = torch.rand(env_ids_count, device=self.device) * 3.0 - 1.5

        # 3. Estado inicial del robot
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        
        # Posición inicial aleatoria en el plano
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # Altura inicial segura (1m) para que no empiece chocando
        default_root_state[:, 2] = 1.0 
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)