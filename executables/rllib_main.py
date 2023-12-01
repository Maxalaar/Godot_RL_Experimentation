import ray
from ray.rllib.algorithms.ppo import PPOConfig

from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv, register_env, rllib_training
import random
import time
from ray import air, tune
import os

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
            speedup=20,
            show_window=False,
        )
    )

    config = (
        PPOConfig()
        .environment(env="Pong")
        # .environment("Taxi-v3")
        .rollouts(num_rollout_workers=10)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1, evaluation_interval=50)
    )

    stop = {
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
        'time_total_s': 60 * 60 * 24,
    }

    tuner = tune.Tuner(
        trainable="PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            verbose=1,
            storage_path=os.path.abspath(os.getcwd()) + '/ray_results',
        ),
    )
    results = tuner.fit()

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")
