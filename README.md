# Instructions for Running gdrl with RLLib and Stable-Baselines3

Project aimed at allowing easy testing of Godot RL Agent with small, simple usage cases.

## Creating and Activating a Conda Environment

The first step is to create the Conda environment by executing the following command (The command may take several minutes to finish executing):

```shell
python executables/conda_environment/load_conda_environment.py
```

Next, activate the Conda environment with the following command:

```shell
conda activate Godot_RL_Experimentation
```

We remove the base Godot-RL library.
```shell
pip uninstall godot-rl
```

Move to godot_rl_agents directory.

```shell
cd godot_rl_agents
```

Installation of the library locally (we install the library locally because there are modifications to make it compatible with RLlib).

```shell
pip install -e .
```

## Run gdrl with Stable-Baselines3

To run `gdrl` with the specified environment and experiment name using Stable-Baselines3, use the following command:

```shell
gdrl --env=gdrl --env_path=environments/RingPong/bin/RingPong.x86_64 --experiment_name=Experiment_01 --viz
```

Or you can use the following command to execute the Python file :

```shell
python executables/sb3_main.py
```
## Run gdrl with RLlib

You can use the following command to execute the Python file:

```shell
python executables/rllib_main.py
```