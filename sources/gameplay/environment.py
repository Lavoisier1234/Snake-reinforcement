import copy
import csv
import pprint
import random
import time
from typing import NoReturn

from sources.gameplay.elements import (
    ALL_BLOCK_TYPES,
    ALL_SNAKE_ACTIONS,
    BlockType,
    Board,
    Point,
    Snake,
    SnakeAction,
)


class TimestepResult:
    """Represents the information provided to the agent after each timestep."""

    def __init__(
        self, observation: list[list[int]], reward: int, is_episode_end: bool
    ):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        if self.observation is None:
            game_map = None
        else:
            game_map = "\n".join(
                ["".join(str(block) for block in row) for row in self.observation]
            )
        return f"{game_map}\nR = {self.reward}   end={self.is_episode_end}\n"


class Environment:
    """
    Represents the RL environment for the Snake game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def __init__(self, config: dict, log_level: int = 1) -> None:
        """create environment

        Args:
            config (dict): level configration from JSON
            log_level (int, optional):
                Defaults to 1.
                0 = do not write any debug information;
                1 = write a CSV file containing the statistics for every episode;
                2 = same as 1, but also write a full log file containing the state of each timestep.
        """
        self.board = Board(game_map=config["board"])
        self.snake = None
        self.fruit = None
        self.initial_snake_length = config["initial_snake_length"]
        self.rewards = config["rewards"]
        self.max_step_limit = config.get("max_step_limit", 1000)
        self.is_game_over = False

        self.timestep_idx = 0
        self.current_action = None
        self.stats = EpisodeStatistics()
        self.log_level = log_level
        self.debug_file = None
        self.stats_file = None

    def seed(self, value) -> None:
        random.seed(value)

    @property
    def observation_shape(self) -> tuple[int, int]:
        return self.board.size, self.board.size

    @property
    def num_actions(self) -> int:
        return len(ALL_SNAKE_ACTIONS)

    @property
    def num_block_types(self) -> int:
        return len(ALL_BLOCK_TYPES)

    def new_episode(self) -> TimestepResult | NoReturn:
        self.board.setup_blocks()
        self.stats.reset()
        self.timestep_idx = 0

        self.snake = Snake(
            self.board.get_snake_head(), length=self.initial_snake_length
        )
        self.board.place_snake(self.snake)
        self.fruit = self.board.get_fruit()
        if self.fruit is None:
            self.generate_fruit()
        self.current_action = None
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over,
        )

        self.record_timestep_stats(result)
        return result

    def record_timestep_stats(self, result: TimestepResult) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if self.log_level >= 1 and self.stats_file is None:
            self.stats_file = open(f"snake-env-{timestamp}.csv", "w", newline="")

        if self.log_level >= 2 and self.debug_file is None:
            self.debug_file = open(f"snake-env-{timestamp}.log", "w")

        self.stats.record_timestep(self.current_action, result)
        self.stats.timestep_survived = self.timestep_idx

        if self.log_level >= 2:
            print(result, file=self.debug_file)

        if result.is_episode_end:
            if self.log_level >= 1:
                writer = csv.writer(self.stats_file)  # pyright: ignore[reportArgumentType]
                writer.writerows(self.stats.to_list())
            if self.log_level >= 2:
                print(self.stats, file=self.debug_file)

    def get_observation(self) -> list[list[int]]:
        assert self.board._blocks is not None
        return copy.deepcopy(self.board._blocks)

    def choose_action(self, action) -> None:
        assert self.snake is not None

        self.current_action = action
        if self.current_action == SnakeAction.TURN_LEFT:
            self.snake.turn_left()
        elif self.current_action == SnakeAction.TURN_RIGHT:
            self.snake.turn_right()

    def time_step(self) -> TimestepResult | NoReturn:
        """Execute the timestep and return the new observable state."""
        assert self.snake is not None

        self.timestep_idx += 1
        reward = 0

        old_head = self.snake.head
        old_tail = self.snake.tail

        # about to eat fruit?
        if self.snake.peek_next_move() == self.fruit:
            self.snake.grow()
            self.generate_fruit()
            old_tail = None
            reward += self.rewards["ate_fruit"] * self.snake.length / 5.0
            self.stats.fruit_eaten += 1

        else:
            self.snake.move()
            reward += self.rewards["timestep"]

        self.board.update_snake(old_head, old_tail, self.snake.head)

        if not self.is_alive():
            if self.has_hit_boundary():
                self.stats.termination_reason = "hit_boundary"
            elif self.has_hit_self():
                self.stats.termination_reason = "hit_self"
                self.board[self.snake.head] = BlockType.SNAKE_HEAD

            self.is_game_over = True
            reward = self.rewards["died"]

        if self.timestep_idx >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = "timestep_limit_exceeded"

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over,
        )

        self.record_timestep_stats(result)
        return result

    def generate_fruit(self, position: Point | None = None) -> None:
        if position is None:
            position = self.board.get_rendom_empty_block()
        self.board[position] = BlockType.FRUIT
        self.fruit = position

    def has_hit_boundary(self) -> bool:
        assert self.snake is not None

        return not (
            0 <= self.snake.head.h < self.board.size
            and 0 <= self.snake.head.w < self.board.size
        )

    def has_hit_self(self) -> bool:
        assert self.snake is not None

        return self.board[self.snake.head] == BlockType.SNAKE_BODY

    def is_alive(self) -> bool:
        assert self.snake is not None

        if self.has_hit_boundary():
            return False
        return not self.has_hit_boundary() and not self.has_hit_self()


class EpisodeStatistics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.timestep_survived = 0
        self.sum_episode_reward = 0
        self.fruit_eaten = 0
        self.termination_reason: str | None = None
        self.action_counter = {action: 0 for action in ALL_SNAKE_ACTIONS}

    def record_timestep(self, action: int | None, result: TimestepResult) -> None:
        self.sum_episode_reward += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self) -> dict:
        """turn self into a dict"""

        flat_stats = {
            "timestep_survived": self.timestep_survived,
            "sum_episode_reward": self.sum_episode_reward,
            "mean_reward": self.sum_episode_reward / self.timestep_survived
            if self.timestep_survived != 0
            else None,
            "fruit_eaten": self.fruit_eaten,
            "termination_reason": self.termination_reason,
        }
        flat_stats.update(
            {
                f"action_counter_{action}": self.action_counter.get(action, 0)
                for action in ALL_SNAKE_ACTIONS
            }
        )

        return flat_stats

    def to_list(self) -> list:
        return [list(self.flatten().keys()), list(self.flatten().values())]

    def __str__(self) -> str:
        return pprint.pformat(self.flatten())
