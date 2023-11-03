import os
import pathlib

from stable_baselines3 import A2C, PPO

from godot_rl.wrappers.onnx.stable_baselines_export import export_ppo_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

if __name__ == '__main__':
    onnx_export_path = 'models/model_v1'
    env: StableBaselinesGodotEnv = StableBaselinesGodotEnv(
        env_path='environments/RingPong/bin/RingPong.x86_64',
        n_parallel=10,
    )

    model: PPO = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    if onnx_export_path is not None:
        path_onnx = pathlib.Path(onnx_export_path).with_suffix(".onnx")
        print("Exporting onnx to: " + os.path.abspath(path_onnx))
        export_ppo_model_as_onnx(model, str(path_onnx))

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    #     # VecEnv resets automatically
    #     # if done:
    #     #   obs = vec_env.reset()
