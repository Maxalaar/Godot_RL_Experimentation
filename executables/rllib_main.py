from ray.rllib.algorithms.ppo import PPOConfig
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv

if __name__ == '__main__':
    environment: RayVectorGodotEnv = RayVectorGodotEnv(
        env_path='environments/RingPong/bin/RingPong.x86_64'
    )

    # Configure the algorithm
    configuration = (
        PPOConfig()
        .environment(env=environment)
        .rollouts(num_rollout_workers=2)
        .framework('torch')
        .training(model={'fcnet_hiddens': [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    # Build the algorithm
    algorithme = configuration.build()

    # Train the algorithm
    for _ in range(5):
        print(algorithme.train())

    # Evaluate the algorithm
    algorithme.evaluate()
