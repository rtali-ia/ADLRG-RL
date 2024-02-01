import numpy as np
import gym
from GridWorld import GridWorldEnv
import matplotlib.pyplot as plt

# Create the grid world environment
env = GridWorldEnv()

# Q-learning parameters
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration-exploitation trade-off
num_episodes = 10000


# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Lists to store the positions visited by the agent during testing
visited_positions = []

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, done, _ = env.step(action)

        # Q-value update
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state

# Testing the trained agent and plotting the path
state = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, _ = env.step(action)

    # Record the visited position for plotting
    visited_positions.append(env.state_to_position(state))

    state = next_state

# Plotting the path
grid_size = env.grid_size
grid = np.zeros((grid_size, grid_size))

for position in visited_positions:
    grid[position] = 1  # Mark visited positions

# Mark the start and goal positions
grid[env.start_position] = 0.5
grid[env.goal_position] = 0.5

# Plot the grid
plt.imshow(grid, cmap='Blues', interpolation='none', origin='lower', extent=[0, grid_size, 0, grid_size])
plt.title('Agent\'s Path in Grid World')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Visited')

plt.show()