import json
import os
import torch

from sources.agent.dqn import DeepQNetworkAgent
from sources.gameplay.environment import Environment
from sources.gui.pygame import PyGameGUI
from train import DeepQNetwork, create_environment


def create_snake_environment(level_filename):
    """Create a new Snake environment from the config file."""

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, log_level=1)


def load_model(model_path, config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), "model_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    class MockEnv:
        def __init__(self, config):
            self.num_block_types = config["num_block_types"]
            self.observation_shape = tuple(config["observation_shape"])
            self.num_actions = config["num_actions"]

    env = MockEnv(config)
    model = DeepQNetwork(env, config["num_last_frames"])

    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, config


def play_gui(env, agent, num_episodes):
    """
    Play a set of episodes using the specified Snake agent.
    Use the interactive graphical interface.

    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """

    gui = PyGameGUI()
    gui.load_env(env)
    gui.load_agent(agent)
    return gui.run(num_episodes=num_episodes)


def main():
    env = create_environment("sources\\init_configs\\10x10_blank.json")

    model, config = load_model("model_2_re\\dqn_modal.pth")

    agent = DeepQNetworkAgent(model, env, config["num_last_frames"], memory_size=-1)

    mean_score = play_gui(env, agent, 50)
    print(mean_score)


if __name__ == "__main__":
    main()
