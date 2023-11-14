import ray
from ray.rllib.algorithms.ppo import PPOConfig
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv, register_env, rllib_training
from godot_rl.main import get_args
from ray import tune
import random
import time

if __name__ == '__main__':
    start_time = time.time()
    ray.init()
    port:int =10008
    # register_env()
    tune.register_env(
        name="godot_environment",
        env_creator=lambda env_ctx: RayVectorGodotEnv(
            env_path='environments/RingPong/bin/RingPong.x86_64',
            port=port + random.randint(1, 100),
            speedup=10,
        )
    )

    # environment: RayVectorGodotEnv = RayVectorGodotEnv(
    #     env_path='environments/RingPong/bin/RingPong.x86_64'
    # )

    config = (  # 1. Configure the algorithm,
        PPOConfig()
        # .environment(env="godot_environment")
        .environment("Taxi-v3")
        .rollouts(num_rollout_workers=2)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    algo = config.build(use_copy=False)

    for _ in range(5):
        print(algo.train())

    print(algo.evaluate())
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")
