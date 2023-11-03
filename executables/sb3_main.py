from stable_baselines3 import A2C

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

if __name__ == '__main__':
    env: StableBaselinesGodotEnv = StableBaselinesGodotEnv(
        env_path='environments/RingPong/bin/RingPong.x86_64',
        n_parallel=10,
    )

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    #     # VecEnv resets automatically
    #     # if done:
    #     #   obs = vec_env.reset()
