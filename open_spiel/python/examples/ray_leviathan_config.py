import tensorflow.compat.v1 as tf
from absl import app, flags, logging

from open_spiel.python import policy, rl_environment
from open_spiel.python.algorithms import exploitability, policy_gradient
from python.games.leviathan_game_v1 import _NUM_PLAYERS, Action, LevithanState

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv as OpenSpielEnvWrapper
from ray.rllib.algorithms.ppo import PPOConfig
import ray

# Game and environment setup
game = "python_leviathan"
num_players = _NUM_PLAYERS
env_configs = {}
env = rl_environment.Environment(game, **env_configs)

# Register the environment with Ray
register_env("leviathan", lambda config: OpenSpielEnvWrapper(env))

# Initialize Ray
ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure PPO algorithm
config = PPOConfig()
config['env'] = "leviathan"
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)
# config["log_level"] = "WARN"

# Build and train the algorithm
algo = config.build(env="leviathan")
algo.train()