from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn.functional as F

from sources.gameplay.environment import Environment


def encode_state(
    state_queue: list, env: Environment, num_last_frames: int
) -> torch.Tensor:
    """Turn the state queue into the form of model input

    Args:
        state_queue (list): the last `num_last_frames` state list, each shape: (h, w)
        env (Environment):
        num_last_frames (int):

    Returns:
        Processed tensor with shape:(num_block_types * num_last_frames, h, w)
    """
    assert len(state_queue) == num_last_frames

    processed_frames = []
    for state in state_queue:
        state_tensor = torch.tensor(state, dtype=torch.long)
        one_hot = F.one_hot(state_tensor, num_classes=env.num_block_types)
        one_hot = one_hot.permute(2, 0, 1).float()
        processed_frames.append(one_hot)

    stacked = torch.cat(processed_frames, dim=0)
    return stacked


class Memory:
    def __init__(self, board_shape: tuple[int, int], memory_size: int = 100) -> None:
        """
        Args:
            board_shape (tuple[int, int]): the shape of the game board (h * w).
            memory_size (int, optional): memory size limit (-1 for unlimited). Defaults to 100.
        """
        if memory_size >= 0:
            self.memory = deque(maxlen=memory_size)
        else:
            self.memory = deque()
        self.board_shape = board_shape
        self.memory_size = memory_size

    def clear(self) -> None:
        if self.memory_size >= 0:
            self.memory = deque(maxlen=self.memory_size)
        else:
            self.memory = deque()

    def remember(
        self,
        state,
        action: int,
        reward: int,
        state_next,
        is_episode_end: bool,
    ) -> None:
        self.memory.append((state, action, reward, state_next, is_episode_end))

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = random.sample(self.memory, batch_size)

        states = torch.stack([b[0] for b in batch]).squeeze(1)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        states_next = torch.stack([b[3] for b in batch]).squeeze(1)
        episode_ends = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        return states, actions, rewards, states_next, episode_ends

    def __len__(self) -> int:
        return len(self.memory)

    # def get_batch(self, model, batch_size: int, discount_factor: float = 0.9):
    #     batch_size = min(len(self.memory), batch_size)
    #     experience = np.array(random.sample(self.memory, batch_size))
    #     board_size = np.prod(self.board_shape)

    #     states = experience[:, 0:board_size]
    #     actions = experience[:, board_size]
    #     rewards = experience[:, board_size + 1]
    #     states_naxt = experience[:, board_size + 2 : 2 * board_size + 2]
    #     episode_ends = experience[:, 2 * board_size + 2]

    #     states = encode_state(states)
