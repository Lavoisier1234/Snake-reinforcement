from __future__ import annotations


from collections import deque
import copy
import json
import random
import torch

from sources.agent import AgentBase
from sources.gameplay.environment import Environment
from sources.utils.memory import Memory, encode_state

# from train import DeepQNetwork


class DeepQNetworkAgent(AgentBase):
    def __init__(
        self,
        model: DeepQNetwork,  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
        env: Environment,
        num_last_frames: int = 4,
        memory_size: int = 1000,
    ) -> None:
        """

        Args:
            model (DeepQNetwork): input shape should be (num_block_types * num_last_frames, board_size, board_size)
                output shape should be (num_samples, num_actions)
            env(Environment):
            num_last_frames (int, optional):  Defaults to 4.
            memory_size (int, optional):  Defaults to 1000.
        """
        self.model = model
        self.env = env
        self.num_last_frames = num_last_frames
        self.memory = Memory(model.board_shape, memory_size)

        self.frames = deque(maxlen=num_last_frames)
        self.env.new_episode()
        init_board = env.get_observation()
        for _ in range(self.num_last_frames):
            self.frames.append(init_board)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def begin_episode(self) -> None:
        self.frames = deque(maxlen=self.num_last_frames)
        self.env.new_episode()
        init_board = self.env.get_observation()
        for _ in range(self.num_last_frames):
            self.frames.append(init_board)

    def end_episode(self):
        pass

    def act(self, observation: list[list[int]]) -> int:
        if random.random() < 0.005:
            return random.choice(range(self.env.num_actions))
        else:
            self.frames.append(observation)
            x = encode_state(
                list(self.frames), self.env, self.num_last_frames
            )  # (1, num_block_types * num_last_frames, h, w)
            q = self.model(x.unsqueeze(0).to(self.device))  # Tensor of (1, 3)
            return q.argmax().item()

    def load_checkpoint(self, checkpoint_file: str):
        checkpoint = torch.load(checkpoint_file)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        target_model = copy.deepcopy(self.model)
        target_model.load_state_dict(checkpoint["target_model_state_dict"])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_episode = checkpoint["episode"] + 1
        epsilon = checkpoint["epsilon"]

        return target_model, optimizer, start_episode, epsilon

    def train(
        self,
        env: Environment,
        num_episodes: int = 1000,
        batch_size: int = 50,
        discount_factor: float = 0.9,
        target_model_update_freq: int = 1000,
        checkpoint_freq: int | None = None,
        exploration_range: tuple[float, float] = (1.0, 0.1),
        exploration_phase_size: float = 0.3,
        checkpoint_file: str | None = None,
    ):
        """train

        Args:
            env (Environment):
            num_episodes (int, optional):
                the number of episodes to run. Defaults to 1000.
            batch_size (int, optional):
                size of sampling. Defaults to 50.
            discount_factor (float, optional):
                gamma. Defaults to 0.9.
            checkpoint_freq (int | None, optional):
                save the model into file every `checkpoint_freq` episode run. If `None`, do not save. Defaults to None.
            exploration_range (tuple[float, float], optional):
                range of probability of randomly exploration (epsilon). Defaults to (1.0, 0.1).
            exploration_phase_size(float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
            checkpoint_file(str | None, optional):
                restart with the checkpint file. `None` to start from zero. Defaults to None.
        """
        TAU = 0.001

        max_epsilon, min_epsilon = exploration_range
        epsilon_decay = (max_epsilon - min_epsilon) / (
            num_episodes * exploration_phase_size
        )

        if checkpoint_file is None:
            target_model = copy.deepcopy(self.model)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

            epsilon = max_epsilon
            start_episode = 0
        else:
            target_model, optimizer, start_episode, epsilon = self.load_checkpoint(
                checkpoint_file
            )

        loss = torch.Tensor([-1.0])

        for episode in range(start_episode, num_episodes):
            self.frames = deque(maxlen=self.num_last_frames)
            timestep = self.env.new_episode()
            init_board = timestep.observation
            for _ in range(self.num_last_frames):
                self.frames.append(init_board)

            game_over = False
            loss_lst = []

            while not game_over:
                current_state = encode_state(
                    list(self.frames), env, self.num_last_frames
                )
                if random.random() < epsilon:
                    action = random.choice(range(env.num_actions))
                else:
                    with torch.no_grad():
                        q_values = self.model.forward(
                            current_state.unsqueeze(0).to(self.device)
                        )
                        action = int(q_values.argmax().item())

                env.choose_action(action)
                timestep = env.time_step()

                self.frames.append(timestep.observation)
                reward = timestep.reward
                game_over = timestep.is_episode_end

                next_encoded = encode_state(
                    list(self.frames), env, self.num_last_frames
                )

                self.memory.remember(
                    current_state, action, reward, next_encoded, game_over
                )

                if len(self.memory) > batch_size:
                    # train_step
                    states, actions, rewards, next_states, episode_ends = (
                        self.memory.sample(batch_size)
                    )

                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    rewards = rewards.to(self.device)
                    next_states = next_states.to(self.device)
                    episode_ends = episode_ends.to(self.device)

                    q_values = self.model.forward(states)

                    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_actions = self.model.forward(next_states).argmax(1)
                        next_q = (
                            target_model.forward(next_states)
                            .gather(1, next_actions.unsqueeze(1))
                            .squeeze(1)
                        )

                        target_q: torch.Tensor = rewards + discount_factor * next_q * (
                            1 - episode_ends
                        )

                    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)

                    loss = loss_fn(current_q, target_q)
                    if episode % 10 == 0:
                        loss_lst.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()

                    for target_param, param in zip(
                        target_model.parameters(), self.model.parameters()
                    ):
                        target_param.data.copy_(
                            TAU * param.data + (1.0 - TAU) * target_param.data
                        )

                if epsilon > min_epsilon:
                    epsilon -= epsilon_decay / 10.0

            if checkpoint_freq is not None and episode % checkpoint_freq == 0:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "target_model_state_dict": target_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "episode": episode,
                        "epsilon": epsilon,
                    },
                    f"model_{self.model.model_name}\\dqn_checkpoint_{episode:08d}.pth",
                )

            if episode % 10 == 0:
                summary = (
                    f"Episode: {episode:5d}/{num_episodes:5d} | Loss mean: {sum(loss_lst) / len(loss_lst):8.4f} | Epsilon: {epsilon:.2f}\n"
                    f"Fruits: {env.stats.fruit_eaten:2d} | Timesteps: {env.stats.timestep_survived:4d} | Total reward: {env.stats.sum_episode_reward:.2f}\n"
                )

                print(summary)

            # if episode % 10 == 0:
            #     print(f"updating target model at episode: {episode}")
            #     target_model.load_state_dict(self.model.state_dict())

        torch.save(
            self.model.state_dict(), f"model_{self.model.model_name}\\dqn_modal.pth"
        )
        with open(f"model_{self.model.model_name}\\model_config.json", "w") as f:
            json.dump(
                {
                    "num_block_types": env.num_block_types,
                    "num_last_frames": self.num_last_frames,
                    "observation_shape": env.observation_shape,
                    "num_actions": env.num_actions,
                },
                f,
            )
