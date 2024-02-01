import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=8):
        super(GridWorldEnv, self).__init__()

        self.grid_size = grid_size
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right

        # Define the initial state (starting position)
        self.start_position = (0, 0)
        self.current_position = self.start_position

        # Define the goal state
        self.goal_position = (grid_size - 1, grid_size - 1)

        # Set maximum number of steps
        self.max_steps = grid_size * 2

        # Initialize step count
        self.current_step = 0

    def reset(self):
        self.current_position = self.start_position
        self.current_step = 0
        return self.position_to_state(self.current_position)

    def step(self, action):
        if action == 0:  # Up
            self.current_position = (max(0, self.current_position[0] - 1), self.current_position[1])
        elif action == 1:  # Down
            self.current_position = (min(self.grid_size - 1, self.current_position[0] + 1), self.current_position[1])
        elif action == 2:  # Left
            self.current_position = (self.current_position[0], max(0, self.current_position[1] - 1))
        elif action == 3:  # Right
            self.current_position = (self.current_position[0], min(self.grid_size - 1, self.current_position[1] + 1))

        # Calculate reward and check if the agent reached the goal
        reward = -1  # Constant negative reward for each step
        done = self.current_position == self.goal_position or self.current_step >= self.max_steps

        # Update step count
        self.current_step += 1

        return self.position_to_state(self.current_position), reward, done, {}

    def position_to_state(self, position):
        return position[0] * self.grid_size + position[1]

    def state_to_position(self, state):
        return divmod(state, self.grid_size)
