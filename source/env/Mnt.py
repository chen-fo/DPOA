import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces


EVENT_NONE = 0
EVENT_TARGET = 1
EVENT_OUT_OF_BOUNDS = -1
EVENT_MINE = -2
EVENT_COLLISION = -3
EVENT_CONTACT = -4
EVENT_TIMEOUT = -5


class Mnt(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        size=16,
        num_agents=2,
        num_mines=8,
        sound=8,
        max_steps=30,
        gamma=0.95,
        render_mode=None,
        terminate_on_target=True,
        contact_radius=1,
        target_reward=50.0,
        hazard_penalty=-10.0,
        step_penalty=0.0,
    ):
        super().__init__()

        if size < 5:
            raise ValueError("size must be at least 5")
        if num_agents < 1:
            raise ValueError("num_agents must be positive")
        if num_mines < 0:
            raise ValueError("num_mines must be non-negative")
        if sound < 1:
            raise ValueError("sound must be at least 1")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        self.size = int(size)
        self.num_agents = int(num_agents)
        self.agent_num = self.num_agents
        self.num_mines = int(num_mines)
        self.sound = int(sound)
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.render_mode = render_mode
        self.terminate_on_target = bool(terminate_on_target)
        self.contact_radius = int(contact_radius)
        self.target_reward = float(target_reward)
        self.hazard_penalty = float(hazard_penalty)
        self.step_penalty = float(step_penalty)

        # Relative actions around the current heading: left 90, left 45,
        # forward, right 45, right 90.
        self.relative_action_offsets = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
        self.action_size = len(self.relative_action_offsets)
        self.obs_size = 1 + 2 + 5 + 5 + 1

        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_agents, self.action_size, dtype=np.int64)
        )
        obs_low = np.array(
            [0.0, -float(self.size), -float(self.size)] + [0.0] * 10 + [0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [7.0, float(self.size), float(self.size)] + [1.0] * 10 + [float(self.max_steps)],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.tile(obs_low, (self.num_agents, 1)),
            high=np.tile(obs_high, (self.num_agents, 1)),
            shape=(self.num_agents, self.obs_size),
            dtype=np.float32,
        )

        self.map = np.zeros((self.size, self.size), dtype=np.int32)
        self.mines_map = np.zeros((self.size, self.size), dtype=np.int32)
        self.positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.prev_positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.headings = np.zeros(self.num_agents, dtype=np.int32)
        self.alive = np.ones(self.num_agents, dtype=bool)
        self.events = np.zeros(self.num_agents, dtype=np.int32)
        self.path = np.zeros((self.num_agents, self.max_steps + 1, 2), dtype=np.int32)
        self.target = np.zeros(2, dtype=np.int32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.map.fill(0)
        self.mines_map.fill(0)
        self.positions.fill(0)
        self.prev_positions.fill(0)
        self.headings.fill(0)
        self.alive[:] = True
        self.events.fill(EVENT_NONE)
        self.path.fill(0)

        options = options or {}
        self._reset_layout(options)
        self.path[:, 0, :] = self.positions

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_agents,):
            raise ValueError(
                f"actions must have shape ({self.num_agents},), got {actions.shape}"
            )
        if np.any(actions < 0) or np.any(actions >= self.action_size):
            raise ValueError(f"actions must be in [0, {self.action_size - 1}]")

        self.steps += 1
        self.events.fill(EVENT_NONE)
        rewards = np.full(self.num_agents, self.step_penalty, dtype=np.float32)
        self.prev_positions[:] = self.positions

        proposed = self.positions.copy()
        for agent_id in range(self.num_agents):
            if not self.alive[agent_id]:
                continue
            next_pos, next_heading, in_bounds = self._virtual_move(agent_id, actions[agent_id])
            self.headings[agent_id] = next_heading
            if not in_bounds:
                self.alive[agent_id] = False
                self.events[agent_id] = EVENT_OUT_OF_BOUNDS
                rewards[agent_id] = self.hazard_penalty
                continue
            proposed[agent_id] = next_pos

        self.positions[:] = proposed
        self._resolve_collisions(rewards)
        self._resolve_mines(rewards)
        self._resolve_contacts(rewards)
        self._resolve_target(rewards)

        if self.steps >= self.max_steps:
            timed_out = self.alive.copy()
            self.events[timed_out & (self.events == EVENT_NONE)] = EVENT_TIMEOUT
            rewards[timed_out] = np.where(
                rewards[timed_out] == self.step_penalty,
                self.hazard_penalty,
                rewards[timed_out],
            )
            self.alive[timed_out] = False

        self._rebuild_map()
        self.path[:, min(self.steps, self.max_steps), :] = self.positions

        terminated = bool(
            (self.terminate_on_target and np.any(self.events == EVENT_TARGET))
            or (not np.any(self.alive))
        )
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        return obs, rewards, terminated, truncated, info

    def render(self):
        lines = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if self.target[0] == x and self.target[1] == y:
                    row.append("T")
                elif self.mines_map[x, y] == 1:
                    row.append("*")
                elif self.map[x, y] > 0:
                    row.append(str(self.map[x, y] - 1))
                else:
                    row.append(".")
            lines.append(" ".join(row))
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self):
        return None

    def judge_move(self, agent_a, agent_b, action_a, action_b):
        pos_a, _, ok_a = self._virtual_move(agent_a, action_a)
        pos_b, _, ok_b = self._virtual_move(agent_b, action_b)
        if not ok_a or not ok_b:
            return False
        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1])) > self.contact_radius

    def virtual_move(self, agent_id, action):
        pos, _, in_bounds = self._virtual_move(agent_id, action)
        if not in_bounds:
            return False
        return pos.tolist()

    def check_live(self, agent_id):
        return bool(self.alive[agent_id])

    def check_done(self):
        return not np.any(self.alive)

    def get_info(self):
        return self._get_info()

    def who_around(self, agent_id, radius=3):
        around = []
        x, y = self.positions[agent_id]
        for other_id in range(self.num_agents):
            if other_id == agent_id or not self.alive[other_id]:
                continue
            ox, oy = self.positions[other_id]
            if abs(x - ox) <= radius and abs(y - oy) <= radius:
                around.append(other_id)
        return around

    def _reset_layout(self, options):
        custom_positions = options.get("positions")
        custom_headings = options.get("headings")
        custom_target = options.get("target")
        custom_mines = options.get("mines")

        if custom_positions is not None:
            positions = np.asarray(custom_positions, dtype=np.int32)
            if positions.shape != (self.num_agents, 2):
                raise ValueError(
                    f"positions must have shape ({self.num_agents}, 2), got {positions.shape}"
                )
            seen = set()
            for idx, (x, y) in enumerate(positions):
                self._validate_in_bounds((x, y), "agent position")
                key = (int(x), int(y))
                if key in seen:
                    raise ValueError("agent positions must be unique")
                seen.add(key)
                self.positions[idx] = positions[idx]
        else:
            self._sample_agent_positions()

        if custom_headings is not None:
            headings = np.asarray(custom_headings, dtype=np.int32)
            if headings.shape != (self.num_agents,):
                raise ValueError(
                    f"headings must have shape ({self.num_agents},), got {headings.shape}"
                )
            self.headings[:] = headings % 8
        else:
            self.headings[:] = np.random.randint(0, 8, size=self.num_agents)

        if custom_target is not None:
            target = np.asarray(custom_target, dtype=np.int32)
            if target.shape != (2,):
                raise ValueError("target must have shape (2,)")
            self._validate_in_bounds(target, "target")
            if any(np.array_equal(target, pos) for pos in self.positions):
                raise ValueError("target cannot overlap with an agent")
            self.target[:] = target
        else:
            self._sample_target()

        if custom_mines is not None:
            mines = np.asarray(custom_mines, dtype=np.int32)
            if mines.ndim != 2 or mines.shape[1] != 2:
                raise ValueError("mines must have shape (m, 2)")
            self._place_custom_mines(mines)
        else:
            self._sample_mines()

        self.prev_positions[:] = self.positions
        self._rebuild_map()

    def _sample_agent_positions(self):
        occupied = set()
        for agent_id in range(self.num_agents):
            while True:
                pos = (
                    random.randint(0, self.size - 1),
                    random.randint(0, self.size - 1),
                )
                if pos not in occupied:
                    occupied.add(pos)
                    self.positions[agent_id] = pos
                    break

    def _sample_target(self):
        occupied = {tuple(pos) for pos in self.positions}
        while True:
            target = (
                random.randint(2, self.size - 3),
                random.randint(2, self.size - 3),
            )
            if target not in occupied:
                self.target[:] = target
                return

    def _place_custom_mines(self, mines):
        self.mines_map.fill(0)
        occupied = {tuple(pos) for pos in self.positions}
        occupied.add(tuple(self.target))
        seen = set()
        for mine in mines:
            key = (int(mine[0]), int(mine[1]))
            self._validate_in_bounds(key, "mine")
            if key in occupied:
                raise ValueError("mine cannot overlap with an agent or target")
            if key in seen:
                raise ValueError("mine positions must be unique")
            seen.add(key)
            self.mines_map[key[0], key[1]] = 1

    def _sample_mines(self):
        self.mines_map.fill(0)
        occupied = {tuple(pos) for pos in self.positions}
        occupied.add(tuple(self.target))
        placed = 0
        max_mines = self.size * self.size - len(occupied)
        target_num_mines = min(self.num_mines, max_mines)
        while placed < target_num_mines:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            key = (x, y)
            if key in occupied or self.mines_map[x, y] == 1:
                continue
            self.mines_map[x, y] = 1
            placed += 1

    def _resolve_collisions(self, rewards):
        collisions = defaultdict(list)
        for agent_id in range(self.num_agents):
            if self.alive[agent_id]:
                collisions[tuple(self.positions[agent_id])].append(agent_id)

        for agent_ids in collisions.values():
            if len(agent_ids) < 2:
                continue
            for agent_id in agent_ids:
                self.alive[agent_id] = False
                self.events[agent_id] = EVENT_COLLISION
                rewards[agent_id] = self.hazard_penalty

    def _resolve_mines(self, rewards):
        for agent_id in range(self.num_agents):
            if not self.alive[agent_id]:
                continue
            x, y = self.positions[agent_id]
            if self.mines_map[x, y] == 1:
                self.alive[agent_id] = False
                self.events[agent_id] = EVENT_MINE
                rewards[agent_id] = self.hazard_penalty

    def _resolve_contacts(self, rewards):
        live_ids = np.flatnonzero(self.alive)
        for idx, agent_id in enumerate(live_ids):
            if not self.alive[agent_id]:
                continue
            ax, ay = self.positions[agent_id]
            for other_id in live_ids[idx + 1 :]:
                if not self.alive[other_id]:
                    continue
                bx, by = self.positions[other_id]
                if max(abs(ax - bx), abs(ay - by)) <= self.contact_radius:
                    self.alive[agent_id] = False
                    self.alive[other_id] = False
                    self.events[agent_id] = EVENT_CONTACT
                    self.events[other_id] = EVENT_CONTACT
                    rewards[agent_id] = self.hazard_penalty
                    rewards[other_id] = self.hazard_penalty

    def _resolve_target(self, rewards):
        for agent_id in range(self.num_agents):
            if not self.alive[agent_id]:
                continue
            if np.array_equal(self.positions[agent_id], self.target):
                self.alive[agent_id] = False
                self.events[agent_id] = EVENT_TARGET
                rewards[agent_id] = self.target_reward * (self.gamma ** self.steps)

    def _rebuild_map(self):
        self.map.fill(0)
        for agent_id in range(self.num_agents):
            if self.alive[agent_id]:
                x, y = self.positions[agent_id]
                self.map[x, y] = agent_id + 1

    def _get_obs(self):
        obs = np.zeros((self.num_agents, self.obs_size), dtype=np.float32)
        mine_sonar = np.zeros(5, dtype=np.float32)
        agent_sonar = np.zeros(5, dtype=np.float32)
        for agent_id in range(self.num_agents):
            if not self.alive[agent_id]:
                obs[agent_id, -1] = float(self.steps)
                continue
            target_delta = self.target - self.positions[agent_id]
            self._get_mine_sonar(agent_id, mine_sonar)
            self._get_agent_sonar(agent_id, agent_sonar)
            obs[agent_id, 0] = float(self.headings[agent_id])
            obs[agent_id, 1] = float(target_delta[0])
            obs[agent_id, 2] = float(target_delta[1])
            obs[agent_id, 3:8] = mine_sonar
            obs[agent_id, 8:13] = agent_sonar
            obs[agent_id, 13] = float(self.steps)
        return obs

    def _get_info(self):
        return {
            "alive": self.alive.copy(),
            "events": self.events.copy(),
            "positions": self.positions.copy(),
            "previous_positions": self.prev_positions.copy(),
            "headings": self.headings.copy(),
            "target": self.target.copy(),
            "mines": np.argwhere(self.mines_map == 1).astype(np.int32),
            "steps": self.steps,
        }

    def _virtual_move(self, agent_id, action):
        heading = int((self.headings[agent_id] + self.relative_action_offsets[int(action)]) % 8)
        dx, dy = self._bearing_to_delta(heading)
        next_pos = self.positions[agent_id] + np.array([dx, dy], dtype=np.int32)
        in_bounds = self._in_bounds(next_pos)
        return next_pos, heading, in_bounds

    @staticmethod
    def _bearing_to_delta(bearing):
        deltas = (
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        )
        return deltas[int(bearing)]

    def _get_mine_sonar(self, agent_id, out):
        x, y = self.positions[agent_id]
        sonar = np.zeros(8, dtype=np.float32)
        for bearing in range(8):
            dx, dy = self._bearing_to_delta(bearing)
            dist = 1
            while True:
                nx = x + dx * dist
                ny = y + dy * dist
                if not self._in_bounds((nx, ny)):
                    sonar[bearing] = 1.0 / dist
                    break
                if self.mines_map[nx, ny] == 1:
                    sonar[bearing] = 1.0 / dist
                    break
                dist += 1
        for idx in range(5):
            out[idx] = sonar[(self.headings[agent_id] + 6 + idx) % 8]

    def _get_agent_sonar(self, agent_id, out):
        x, y = self.positions[agent_id]
        sonar = np.zeros(8, dtype=np.float32)
        for bearing in range(8):
            dx, dy = self._bearing_to_delta(bearing)
            dist = 1
            while dist <= self.sound:
                nx = x + dx * dist
                ny = y + dy * dist
                if not self._in_bounds((nx, ny)):
                    sonar[bearing] = 1.0 / dist
                    break
                occupant = self.map[nx, ny]
                if occupant > 0 and occupant != agent_id + 1:
                    sonar[bearing] = 1.0 / dist
                    break
                dist += 1
            if sonar[bearing] == 0.0 and dist > self.sound:
                sonar[bearing] = 0.0
        for idx in range(5):
            out[idx] = sonar[(self.headings[agent_id] + 6 + idx) % 8]

    def _validate_in_bounds(self, pos, name):
        if not self._in_bounds(pos):
            raise ValueError(f"{name} {tuple(pos)} is out of bounds for size {self.size}")

    def _in_bounds(self, pos):
        x, y = int(pos[0]), int(pos[1])
        return 0 <= x < self.size and 0 <= y < self.size
