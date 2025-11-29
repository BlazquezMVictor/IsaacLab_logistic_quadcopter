from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuración OPTIMIZADA para Crazyflie con física realista."""
    
    # CAMBIO 1: Horizonte temporal más largo.
    # 24 pasos es muy poco (~0.2s). El dron necesita ver las consecuencias de su inercia.
    # Subimos a 48 o 60 pasos (~0.5 - 0.6 segundos de trayectoria por update).
    num_steps_per_env = 48
    
    # Mantenemos las iteraciones, pero al aumentar los steps por env, 
    # cada iteración procesa el doble de datos, así que entrenará "más" en menos iteraciones.
    max_iterations = 1500  
    save_interval = 100
    experiment_name = "quadcopter_tracking_optimized"
    run_name = "" 

    # -- Configuración de la Política --
    policy = RslRlPpoActorCriticCfg(
        activation="elu", 
        # Esta arquitectura está bien para empezar.
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        init_noise_std=1.0, 
    )

    # -- Configuración del Algoritmo PPO --
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        
        # CAMBIO 2: Aumentar Entropía (Exploración).
        # De 0.01 a 0.02. Esto penaliza que la política se vuelva determinista muy rápido.
        # Evita que el dron diga "siempre apago motores" en la iteración 50.
        entropy_coef=0.02, 
        
        # CAMBIO 3: Learning Rate más conservador.
        # 1e-3 es muy rápido. Para control fino de rotores, 3e-4 es el "punto dulce".
        # Permite ajustar los pesos de la red neuronal suavemente sin romper la física.
        learning_rate=3.0e-4, 
        
        schedule="adaptive", 
        desired_kl=0.01,       
        max_grad_norm=1.0,     
        
        gamma=0.99,
        lam=0.95,
        
        num_mini_batches=4,
        num_learning_epochs=5,
    )