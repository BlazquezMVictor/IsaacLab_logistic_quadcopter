from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuración del Runner para RSL-RL específico para Drones."""
    
    # Cantidad de pasos de simulación por entorno antes de actualizar la política.
    # Para drones: 24 pasos * 0.0083s (dt) ~= 0.2 segundos de experiencia por batch.
    # Al tener 4096 entornos, esto genera muchísima data rápidamente.
    num_steps_per_env = 24  
    
    # Pasos máximos de entrenamiento (ajusta según necesites).
    # 15M es un buen número para empezar a ver resultados sólidos en tracking.
    max_iterations = 1500  
    save_interval = 100     # Guardar checkpoint cada 100 iteraciones
    experiment_name = "quadcopter_tracking_v0"
    run_name = ""          # Se rellena automáticamente con fecha/hora

    # -- Configuración de la Política (Red Neuronal) --
    policy = RslRlPpoActorCriticCfg(
        # Red neuronal: Multilayer Perceptron (MLP)
        # Usamos ELU en lugar de ReLU. ELU es más suave y permite valores negativos,
        # lo cual es crucial para la dinámica de vuelo suave en drones.
        activation="elu", 
        
        # Arquitectura asimétrica o separada es común, pero para tracking simple
        # una red compartida o arquitecturas iguales funcionan bien.
        # [256, 128, 64] es suficiente para tracking. Si añades percepción visual, necesitarás más.
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        
        # Inicialización de la desviación estándar del ruido de acción.
        # 1.0 permite buena exploración inicial. El algoritmo la reducirá solo.
        init_noise_std=1.0, 
    )

    # -- Configuración del Algoritmo PPO --
    algorithm = RslRlPpoAlgorithmCfg(
        # Lógica de "Value Function Bootstrap".
        # True = usa el valor del estado siguiente si hay timeout (más estable para tareas continuas).
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        
        # Entropía: Incentiva la exploración.
        # Para drones, empezamos con algo moderado y dejamos que caiga.
        entropy_coef=0.01, 
        
        # Learning Rate y Schedule
        # "adaptive": RSL-RL ajusta el LR automáticamente para mantener el KL divergence
        # cerca del 'desired_kl'. Esto es ORO para evitar colapsos de la política.
        learning_rate=1.0e-3, 
        schedule="adaptive", 
        desired_kl=0.01,       # Objetivo de cambio entre actualizaciones
        max_grad_norm=1.0,     # Evita gradientes explosivos
        
        # Gamma (Discount Factor): 0.99 es estándar.
        # Lambda (GAE): 0.95 para suavizar la estimación de ventaja.
        gamma=0.99,
        lam=0.95,
        
        # Batching
        # num_mini_batches: divide los (num_envs * num_steps) en trozos.
        # 4096 envs * 24 steps = ~98k muestras. 4 minibatches está bien.
        num_mini_batches=4,
        
        # Cuántas veces reutilizamos los datos recolectados para optimizar (Epochs).
        num_learning_epochs=5,
    )