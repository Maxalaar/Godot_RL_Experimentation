from ray.rllib.algorithms.ppo import PPOConfig
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv, register_env, rllib_training
from godot_rl.main import get_args
from ray import tune


if __name__ == '__main__':
    environment: RayVectorGodotEnv = RayVectorGodotEnv(
        env_path='environments/RingPong/bin/RingPong.x86_64'
    )

    args, extras = get_args()
    args.config_file = 'configuration_rllib.yaml'
    args.env_path = 'environments/RingPong/bin/RingPong.x86_64'

    rllib_training(args, extras)


    # register_env()

    # Configure the algorithm
    configuration = (
        PPOConfig()
        # .environment(env=environment)
        .environment(
            env='godot',
            env_config={

            }
        )
        .rollouts(num_rollout_workers=2)
        .framework('torch')
        .training(model={'fcnet_hiddens': [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    # Build the algorithm
    algorithme = configuration.build(
        use_copy=False
    )

    # Train the algorithm
    for _ in range(5):
        print(algorithme.train())

    # Evaluate the algorithm
    algorithme.evaluate()
