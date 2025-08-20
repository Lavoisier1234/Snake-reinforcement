import json
import os
# import sys
import time

from torch import nn
import torch
from sources.agent import DeepQNetworkAgent
from sources.gameplay.environment import Environment
# from sources.utils.cli import HelpOnFailArgumentParser


# def parse_command_line_args(args):
#     """Parse command-line arguments and organize them into a single structured object."""

#     parser = HelpOnFailArgumentParser(
#         description="Snake AI training client.",
#         epilog="Example: train.py --level 10x10.json --num-episodes 30000",
#     )

#     parser.add_argument(
#         "--level",
#         required=True,
#         type=str,
#         help="JSON file containing a level definition.",
#     )
#     parser.add_argument(
#         "--num-episodes",
#         required=True,
#         type=int,
#         default=30000,
#         help="The number of episodes to run consecutively.",
#     )

#     return parser.parse_args(args)


class DeepQNetwork(nn.Module):
    def __init__(
        self, env, num_last_frames: int, model_name: str | None = None
    ) -> None:
        super().__init__()
        self.model_name = model_name if model_name else time.strftime("%Y%m%d-%H%M%S")
        self.board_shape = env.observation_shape
        self.num_last_frames = num_last_frames
        self.in_chnnals = env.num_block_types * num_last_frames
        self.input_shape = (self.in_chnnals,) + env.observation_shape
        self.output_shape = (env.num_actions,)
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_chnnals,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_chnnals, *env.observation_shape)
            flattened_size = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(
                flattened_size,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, env.num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.fc(x)


def create_environment(config_filename) -> Environment:
    with open(config_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, log_level=1)


def main():
    # parsed_args = parse_command_line_args(sys.argv[1:])

    # num_episodes = parsed_args.num_episodes
    num_episodes = 30010
    num_last_frames = 4

    env = create_environment(
        "D:\\Home\\Works\\University\\Code practice\\Python practice\\Reinforcement Learning\\Snake-reinforcement\\sources\\init_configs\\10x10_blank.json"
    )
    # initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepQNetwork(env, num_last_frames, "2_re").to(device)

    os.makedirs(f"model_{model.model_name}", exist_ok=True)

    agent = DeepQNetworkAgent(
        model, env, num_last_frames=num_last_frames, memory_size=100000
    )
    agent.train(
        env,
        num_episodes=num_episodes,
        batch_size=64,
        checkpoint_freq=num_episodes // 20,
        exploration_range=(0.95, 0.1),
        discount_factor=0.95,
        checkpoint_file="model_2_re\\dqn_checkpoint_00030000.pth",
    )


if __name__ == "__main__":
    main()
