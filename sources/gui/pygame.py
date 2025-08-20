import pygame

from sources.agent import AgentBase
from sources.gameplay.elements import BlockType, Point, SnakeAction
from sources.gameplay.environment import Environment


class PyGameGUI:
    FPS_LIMIT = 60
    AI_STEP_DELAY = 15
    BLOCK_SIZE = 40

    def __init__(self) -> None:
        pygame.init()
        self.agent: AgentBase | None = None
        self.env = None
        self.screen = None
        self.fps_clock = None
        self.timestep_watch = Stopwatch()

    def load_env(self, env: Environment):
        self.env = env
        screen_size = (
            self.env.board.size * self.BLOCK_SIZE,
            self.env.board.size * self.BLOCK_SIZE,
        )
        self.screen = pygame.display.set_mode(screen_size)
        self.screen.fill(Colors.SCREEN_BACKGROUND)
        pygame.display.set_caption("Snake")

    def load_agent(self, agent):
        self.agent = agent

    def render_blocks(self, h, w):
        assert self.env is not None
        assert self.screen is not None

        block_coords = pygame.Rect(
            h * self.BLOCK_SIZE,
            w * self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            self.BLOCK_SIZE,
        )

        if self.env.board[Point(h, w)] == BlockType.EMPTY:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, block_coords)
        else:
            color = Colors.CELL_TYPE[self.env.board[Point(h, w)]]
            pygame.draw.rect(self.screen, color, block_coords)

            internal_padding = self.BLOCK_SIZE // 6 * 2
            internal_square_coords = block_coords.inflate(
                (-internal_padding, -internal_padding)
            )
            pygame.draw.rect(self.screen, color, internal_square_coords)

    def render(self):
        assert self.env is not None
        for h in range(self.env.board.size):
            for w in range(self.env.board.size):
                self.render_blocks(h, w)

    def run(self, num_episodes=1):
        """Run the GUI player for the specified number of episodes."""
        pygame.display.update()
        self.fps_clock = pygame.time.Clock()

        try:
            sum_score = 0
            for episode in range(num_episodes):
                score = self.run_episode()
                sum_score += score
                pygame.time.wait(1000)
            return sum_score / num_episodes
        except QuitRequestedError:
            pass

    def run_episode(self):
        assert self.env is not None
        assert self.agent is not None

        self.timestep_watch.reset()
        self.agent.begin_episode()
        timestep_result = self.env.time_step()

        timestep_delay = self.AI_STEP_DELAY

        running = True
        while running:
            action = SnakeAction.MAINTAIN

            timestep_timed_out = self.timestep_watch.time() >= timestep_delay

            if timestep_timed_out:
                self.timestep_watch.reset()

                action = self.agent.act(timestep_result.observation)

                self.env.choose_action(action)
                timestep_result = self.env.time_step()

                if timestep_result.is_episode_end:
                    self.agent.end_episode()
                    running = False

            self.render()
            score = self.env.snake.length - self.env.initial_snake_length  # pyright: ignore[reportOptionalMemberAccess]
            pygame.display.set_caption(f"Snake  [Score: {score:02d}]")
            pygame.display.update()
            self.fps_clock.tick(self.FPS_LIMIT)  # pyright: ignore[reportOptionalMemberAccess]

        return score


class Stopwatch:
    """Measures the time elapsed since the last checkpoint."""

    def __init__(self):
        self.start_time = pygame.time.get_ticks()

    def reset(self):
        """Set a new checkpoint."""
        self.start_time = pygame.time.get_ticks()

    def time(self):
        """Get time (in milliseconds) since the last checkpoint."""
        return pygame.time.get_ticks() - self.start_time


class Colors:
    SCREEN_BACKGROUND = (170, 204, 153)
    CELL_TYPE = {
        # BlockType.WALL: (56, 56, 56),
        BlockType.SNAKE_BODY: (105, 132, 164),
        BlockType.SNAKE_HEAD: (122, 154, 191),
        BlockType.FRUIT: (173, 52, 80),
    }


class QuitRequestedError(RuntimeError):
    pass
