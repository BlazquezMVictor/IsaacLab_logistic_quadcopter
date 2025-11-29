
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2 # Control a 50Hz (si sim es 100Hz)
    
    # Action Space: [Thrust, Moment_Roll, Moment_Pitch, Moment_Yaw]
    action_space = 4
    
    # Parámetros Físicos del Crazyflie 2.1
    # Distancia del centro al motor (en metros)
    arm_length = 0.0397 
    # Relación entre Torque de Yaw y Empuje (Coeficiente de arrastre aerodinámico)
    # T_yaw = k_drag * Force_thrust
    torque_coefficient = 0.006 
    
    # Escalas para normalizar lo que sale de la red neuronal
    # thrust_to_weight ya no es una fuerza mágica, sino el empuje máx total de los 4 motores
    max_thrust_per_motor = 0.15  # Newtons (aprox 15g por motor)

    # Observation Space: 
    # Lin Vel (3) + Ang Vel (3) + Gravity (3) + Commands (4) = 13
    observation_space = 13
    
    state_space = 0
    debug_vis = True

    # Reward Scales (Pesos de recompensa)
    # Nota: Como usamos funciones exponenciales (0 a 1), los pesos pueden ser positivos
    lin_vel_reward_scale = 5.0
    ang_vel_reward_scale = 2.0
    
    # Penalizaciones (Valores negativos)
    lin_acc_penalty_scale = -0.0001
    action_rate_penalty_scale = -0.001

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100, # 100 Hz physics
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            usd_path="/home/usuario/Escritorio/IsaacLab_logistic_quadcopter/assets/quadcopter_unpacked.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "joint_rotor1": 200.0,
                "joint_rotor2": -200.0,
                "joint_rotor3": 200.0,
                "joint_rotor4": -200.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )