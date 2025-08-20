import json
import unittest
from pathlib import Path

from sources.gameplay.elements import SnakeAction
from sources.gameplay.environment import Environment


def get_env_config_file(name: str) -> Path:
    config_dir = Path(__file__).parent.parent / "sources" / "init_configs"
    return config_dir / f"{name}.json"


def load_env(name: str) -> Environment:
    with open(get_env_config_file(name)) as cfg:
        env_cfg = json.load(cfg)
        return Environment(env_cfg, log_level=0)


class TestEnvironment(unittest.TestCase):
    def test_env_new_episode(self):
        env = load_env("10x10_blank")
        env.new_episode()

        self.assertEqual(env.board.size, 10)

        assert env.snake is not None
        self.assertEqual(env.snake.head, (4, 5))
        self.assertEqual(env.timestep_idx, 0)
        self.assertIsNotNone(env.fruit)

    def test_env_normal_step_reports_correctly(self):
        env = load_env("10x10_test1")
        env.seed(42)

        result = env.new_episode()

        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        env.choose_action(SnakeAction.MAINTAIN)
        result = env.time_step()
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        result = env.time_step()
        self.assertEqual(result.reward, 4)
        self.assertFalse(result.is_episode_end)

        result = env.time_step()
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        result = env.time_step()
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        result = env.time_step()
        self.assertEqual(result.reward, -1)
        self.assertTrue(result.is_episode_end)

        self.assertEqual(env.stats.sum_episode_reward, 3)
        self.assertEqual(env.stats.timestep_survived, 5)
        self.assertEqual(env.stats.termination_reason, "hit_boundary")
        self.assertEqual(
            env.stats.action_counter,
            {
                SnakeAction.MAINTAIN: 5,
                SnakeAction.TURN_LEFT: 0,
                SnakeAction.TURN_RIGHT: 0,
            },
        )

    def test_env_hit_self_reports_over(self):
        env = load_env("10x10_test2")
        env.seed(42)

        result = env.new_episode()

        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        env.choose_action(SnakeAction.MAINTAIN)
        result = env.time_step()
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        result = env.time_step()
        self.assertEqual(result.reward, 5)
        self.assertFalse(result.is_episode_end)

        env.choose_action(SnakeAction.TURN_RIGHT)
        result = env.time_step()
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        env.choose_action(SnakeAction.TURN_RIGHT)
        result = env.time_step()
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_episode_end)

        env.choose_action(SnakeAction.TURN_RIGHT)
        result = env.time_step()
        self.assertEqual(result.reward, -1)
        self.assertTrue(result.is_episode_end)

        self.assertEqual(env.stats.sum_episode_reward, 4)
        self.assertEqual(env.stats.timestep_survived, 5)
        self.assertEqual(env.stats.termination_reason, "hit_self")
        self.assertEqual(
            env.stats.action_counter,
            {
                SnakeAction.MAINTAIN: 2,
                SnakeAction.TURN_LEFT: 0,
                SnakeAction.TURN_RIGHT: 3,
            },
        )

    def test_env_timestep_limit_exceeded(self):
        env = load_env("10x10_blank")
        env.seed(42)
        env.new_episode()

        for i in range(env.max_step_limit - 1):
            env.choose_action(SnakeAction.TURN_RIGHT)
            result = env.time_step()
            self.assertFalse(result.is_episode_end)

        env.choose_action(SnakeAction.TURN_RIGHT)
        result = env.time_step()
        self.assertTrue(result.is_episode_end)

    def test_env_new_start_resets_state(self):
        env = load_env("10x10_blank")
        env.seed(42)
        env.new_episode()

        for i in range(env.max_step_limit - 1):
            env.choose_action(SnakeAction.TURN_RIGHT)
            env.time_step()

        result = env.time_step()
        self.assertTrue(result.is_episode_end)
        self.assertEqual(env.stats.timestep_survived, env.max_step_limit)
        self.assertEqual(env.stats.termination_reason, "timestep_limit_exceeded")

        env.new_episode()

        self.assertEqual(env.board.size, 10)

        assert env.snake is not None
        self.assertEqual(env.snake.head, (4, 5))
        self.assertEqual(env.timestep_idx, 0)
        self.assertIsNotNone(env.fruit)

        self.assertEqual(env.stats.sum_episode_reward, 0)
        self.assertEqual(env.stats.fruit_eaten, 0)
        self.assertEqual(env.stats.timestep_survived, 0)
        self.assertIsNone(env.stats.termination_reason)
        self.assertEqual(set(env.stats.action_counter.values()), {0})
