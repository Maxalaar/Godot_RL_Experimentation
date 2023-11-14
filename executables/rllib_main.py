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
        name="RingPong",
        env_creator=lambda env_ctx: RayVectorGodotEnv(
            env_path='environments/RingPong/bin/RingPong.x86_64',
            port=port + random.randint(1, 100),
            speedup=10,
        )
    )

    config = (
        PPOConfig()
        .environment(env="RingPong")
        # .environment("Taxi-v3")
        .rollouts(num_rollout_workers=2)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    stop = {
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        'time_total_s': 60 * 1,
        # "episode_reward_mean": args.stop_reward,
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

    # algo = config.build(use_copy=False)
    #
    # for _ in range(5):
    #     print(algo.train())
    #
    # print(algo.evaluate())
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")
