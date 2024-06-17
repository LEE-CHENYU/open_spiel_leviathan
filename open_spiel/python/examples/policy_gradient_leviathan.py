import logging
from absl import app
from absl import flags

from open_spiel.python.games import leviathan_game  # Import the Leviathan game
from open_spiel.python.pytorch import policy_gradient

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e5), "Number of train episodes.")
flags.DEFINE_integer("eval_every", int(1e3), "How often to evaluate the policy.")
flags.DEFINE_enum("algorithm", "a2c", ["rpg", "qpg", "rm", "a2c"], "Algorithms to run.")

def _eval_agent(env, agent, num_episodes):
    """Evaluates `agent` for `num_episodes`."""
    rewards = 0.0
    for _ in range(num_episodes):
        time_step = env.new_initial_state()
        episode_reward = 0
        while not time_step.is_terminal():
            current_player = time_step.current_player()
            agent_output = agent.step(time_step, is_evaluation=True)
            action_list = [agent_output.action]
            time_step.apply_action(action_list[0])
            episode_reward += time_step.returns()[current_player]
        rewards += episode_reward
    return rewards / num_episodes

def main_loop(unused_arg):
    """Trains a Policy Gradient agent in the Leviathan environment."""
    env = pyspiel.load_game("python_leviathan")  # Initialize the Leviathan environment
    observation_tensor_size = env.observation_tensor_size()
    num_actions = env.num_distinct_actions()

    train_episodes = FLAGS.num_episodes

    agent = policy_gradient.PolicyGradient(
        player_id=0,
        info_state_size=observation_tensor_size,  # Use observation tensor size
        num_actions=num_actions,
        loss_str=FLAGS.algorithm,
        hidden_layers_sizes=[128, 128],
        batch_size=128,
        entropy_cost=0.01,
        critic_learning_rate=0.1,
        pi_learning_rate=0.1,
        num_critic_before_pi=3)

    # Train agent
    for ep in range(train_episodes):
        time_step = env.new_initial_state()
        while not time_step.is_terminal():
            current_player = time_step.current_player()
            agent_output = agent.step(time_step)
            action_list = [agent_output.action]
            time_step.apply_action(action_list[0])
        # Episode is over, step agent with final info state.
        agent.step(time_step)

        if ep and ep % FLAGS.eval_every == 0:
            logging.info("-" * 80)
            logging.info("Episode %s", ep)
            logging.info("Loss: %s", agent.loss)
            avg_return = _eval_agent(env, agent, 100)
            logging.info("Avg return: %s", avg_return)

if __name__ == "__main__":
    app.run(main_loop)