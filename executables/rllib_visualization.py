import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm

from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv, register_env, rllib_training
import random
import time
from ray import air, tune
import os

from ray.rllib.algorithms.ppo import PPO

if __name__ == '__main__':
    start_time = time.time()
    ray.init()
    port:int =10008

    tune.register_env(
        name="Pong",
        env_creator=lambda env_ctx: RayVectorGodotEnv(
            # env_path='environments/RingPong/bin/RingPong.x86_64',
            env_path='/home/malaarabiou/Programming_Projects/Godot_Projects/PingPongGodot/builds/PingPong.x86_64',
            port=port + random.randint(1, 200),
            speedup=10,
            show_window=True,
        )
    )

    if not ray.is_initialized():
        ray.init(local_mode=True)

    path_checkpoint: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Godot_RL_Experimentation/ray_results/PPO_2023-11-27_17-05-54/PPO_Pong_d7d23_00000_0_2023-11-27_17-05-54/checkpoint_000000'
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config: AlgorithmConfig = Algorithm.get_config(algorithm).copy(copy_frozen=False)

    algorithm_config.evaluation(
        evaluation_duration=1,
        # evaluation_config={'render_env': True, },
    )
    algorithm_config.rollouts(
        num_rollout_workers=1,
        num_envs_per_worker=1,
    )

    # algorithm_config.environment(
    #     env=AntColonyEnvironment,
    #     env_config=ant_colony_environment_epic_configuration,
    # )
    # algorithm_config.env_config['graphic_interface_configuration']['render_environment'] = True

    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)
    number_iteration: int = 20
    total_foods_collected: float = 0
    for i in range(number_iteration):
        evaluation_result = algorithm.evaluate()
