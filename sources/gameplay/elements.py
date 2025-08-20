from collections import deque, namedtuple
import itertools
from random import choice
from typing import NoReturn, Self


class Point(namedtuple("Point", ["h", "w"])):
    """Represent a set of 2D coordinates"""

    def __add__(self, other: Self):
        return Point(self.h + other.h, self.w + other.w)

    def __sub__(self, other: Self):
        return Point(self.h - other.h, self.w - other.w)

    def get_coordinate(self) -> tuple[int, int]:
        return self.h, self.w


class BlockType:
    """Defines all types of blocks that can be found in the game."""

    EMPTY = 0
    FRUIT = 1
    SNAKE_HEAD = 2
    SNAKE_BODY = 3


ALL_BLOCK_TYPES = [
    BlockType.EMPTY,
    BlockType.FRUIT,
    BlockType.SNAKE_HEAD,
    BlockType.SNAKE_BODY,
]


class SnakeDirection:
    """Defines all possible directions the snake can take, as well as the corresponding offsets."""

    LEFT = Point(0, -1)
    RIGHT = Point(0, 1)
    UP = Point(-1, 0)
    DOWN = Point(1, 0)


ALL_SNAKE_DIRECTIONS = [
    SnakeDirection.LEFT,
    SnakeDirection.DOWN,
    SnakeDirection.RIGHT,
    SnakeDirection.UP,
]  # Arranged in a counterclockwise direction, providing easier way for turn left or right


class SnakeAction:
    """Defines all possible actions the agent can take in the environment."""

    MAINTAIN = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


ALL_SNAKE_ACTIONS = [
    SnakeAction.MAINTAIN,
    SnakeAction.TURN_LEFT,
    SnakeAction.TURN_RIGHT,
]


# class Block(Point):
#     """

#     Args:
#           h(int): the height coordinate of this block
#           w(int): the width coordinate of this block
#           blocktype: type of this block
#     """

#     def __init__(self, h: int, w: int, blocktype: int = BlockType.EMPTY) -> None:
#         super().__init__(h, w)
#         self.blocktype = blocktype
#         self._blocktype_to_str = {
#             BlockType.EMPTY: "Empty",
#             BlockType.FRUIT: "Fruit",
#             BlockType.SNAKE_HEAD: "Snake_head",
#             BlockType.SNAKE_BODY: "Snake_body",
#         }
#         self._blocktype_to_map = {
#             BlockType.EMPTY: ".",
#             BlockType.FRUIT: "O",
#             BlockType.SNAKE_HEAD: "S",
#             BlockType.SNAKE_BODY: "s",
#         }

#     def __str__(self) -> str:
#         return self._blocktype_to_map[self.blocktype]

#     def __repr__(self) -> str:
#         return f"This is the {self._blocktype_to_str[self.blocktype]} block with coordinate [{self.h}, {self.w}]"


class Snake:
    """The class of the snake

    Only implement the actual change of the snake's data. All the validity check is not included
    """

    def __init__(self, start_coord: Point, length: int = 3) -> None:
        # Place the snake vertically, heading up.
        self.body = deque(
            [Point(start_coord.h + i, start_coord.w) for i in range(length)]
        )
        self.direction = SnakeDirection.UP
        self.directions = ALL_SNAKE_DIRECTIONS

    @property
    def head(self) -> Point:
        """Get the position of the snake's head."""
        return self.body[0]

    @property
    def tail(self) -> Point:
        """Get the position of the snake's tail."""
        return self.body[-1]

    @property
    def length(self) -> int:
        """Get the current length of the snake."""
        return len(self.body)

    def peek_next_move(self) -> Point:
        """Get the point the snake will move to at its next step."""
        return self.head + self.direction

    def turn_left(self) -> None:
        direction_idx = self.directions.index(self.direction)
        self.direction = self.directions[
            (direction_idx + 1) % 4
        ]  # 4 dircetions in total

    def turn_right(self) -> None:
        direction_idx = self.directions.index(self.direction)
        self.direction = self.directions[direction_idx - 1]

    def grow(self):
        """Grow the snake by 1 block from the head."""
        self.body.appendleft(self.peek_next_move())

    def move(self):
        """Move the snake 1 step forward, taking the current direction into account."""
        self.body.appendleft(self.peek_next_move())
        self.body.pop()


class Board:
    """Define the game board"""

    def __init__(self, game_map: list[str] | None = None) -> None:
        """Create a new board

        Args:
            game_map(list[str]): Represent the game board by a list of str, each for a row
        """
        self.game_map = game_map
        self._blocks: list[list[int]] | None = None
        self._empty_blocks = set()
        self._game_map_to_block_type = {
            ".": BlockType.EMPTY,
            "O": BlockType.FRUIT,
            "S": BlockType.SNAKE_HEAD,
            "s": BlockType.SNAKE_BODY,
        }
        self._block_type_to_game_map = {
            block_type: game_map
            for game_map, block_type in self._game_map_to_block_type.items()
        }

    def __getitem__(self, point: Point) -> int:
        assert self._blocks is not None

        h, w = point
        return self._blocks[h][w]

    def __setitem__(self, point: Point, block_type: int) -> None:
        assert self._blocks is not None

        h, w = point
        self._blocks[h][w] = block_type

        if block_type == BlockType.EMPTY:
            self._empty_blocks.add(point)
        else:
            if point in self._empty_blocks:
                self._empty_blocks.remove(point)

    def __str__(self) -> str | None:
        return (
            None
            if self._blocks is None
            else "\n".join(
                ["".join(str(block) for block in row) for row in self._blocks]
            )
        )

    @property
    def size(self) -> int:
        """Get the size of the game map(size==height==width)"""
        return 0 if self.game_map is None else len(self.game_map)

    def setup_blocks(self) -> None | NoReturn:
        assert self.game_map is not None

        try:
            self._blocks = [
                [self._game_map_to_block_type[char] for char in row]
                for row in self.game_map
            ]
            self._empty_blocks = {
                Point(h, w)
                for h in range(self.size)
                for w in range(self.size)
                if self[Point(h, w)] == BlockType.EMPTY
            }

        except KeyError as e:
            raise ValueError(f'Unknown game map symbol: "{e.args[0]}"')

    def get_fruit(self) -> Point | None:
        for h in range(self.size):
            for w in range(self.size):
                if self[Point(h, w)] == BlockType.FRUIT:
                    return Point(h, w)

    def get_snake_head(self) -> Point | NoReturn:
        for h in range(self.size):
            for w in range(self.size):
                if self[Point(h, w)] == BlockType.SNAKE_HEAD:
                    return Point(h, w)
        raise ValueError("Fail to find any snake head block")

    def get_rendom_empty_block(self) -> Point:
        return choice(list(self._empty_blocks))

    def place_snake(self, snake: Snake) -> None:
        self[snake.head] = BlockType.SNAKE_HEAD
        for snake_block in itertools.islice(snake.body, 1, None):
            self[snake_block] = BlockType.SNAKE_BODY

    def update_snake(
        self, old_head: Point, old_tail: Point | None, new_head: Point
    ) -> None:
        self[old_head] = BlockType.SNAKE_BODY

        if old_tail is not None:
            self[old_tail] = BlockType.EMPTY

        if 0 <= new_head.h < self.size and 0 <= new_head.w < self.size:
            if self[new_head] != BlockType.SNAKE_BODY or new_head == old_tail:
                self[new_head] = BlockType.SNAKE_HEAD
