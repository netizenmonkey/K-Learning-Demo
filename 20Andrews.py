import pygame
import random
import numpy as np

# Constants
GRID_SIZE = 8
TILE_SIZE = 64
SCREEN_SIZE = GRID_SIZE * TILE_SIZE
FPS = 10

# Q-learning settings
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_IDX = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

# Rewards
REWARDS = {
    "start": 0,
    "goal": 10,
    "elephant": -5,
    "fire": -10,
    "popcorn": 5,
    "empty": -1
}

def move(pos, action):
    x, y = pos
    if action == 'U': x = max(x - 1, 0)
    elif action == 'D': x = min(x + 1, GRID_SIZE - 1)
    elif action == 'L': y = max(y - 1, 0)
    elif action == 'R': y = min(y + 1, GRID_SIZE - 1)
    return x, y

def train_multiple_clowns(Q, grid, start, goal, max_episodes=1000, convergence_threshold=0.01):
    """
    Trains 20 clowns in parallel using shared Q-table until the values converge.
    """
    for episode in range(max_episodes):
        total_change = 0

        for _ in range(20):  # Simulate 20 clowns
            pos = start
            visited = set()
            while True:
                if pos not in Q:
                    Q[pos] = np.zeros(4)

                # Epsilon-greedy policy
                if random.random() < EPSILON:
                    action = random.choice(ACTIONS)
                else:
                    action = ACTIONS[np.argmax(Q[pos])]

                new_pos = move(pos, action)
                tile = grid[new_pos[0]][new_pos[1]]
                reward = REWARDS[tile]

                if new_pos not in Q:
                    Q[new_pos] = np.zeros(4)

                old_value = Q[pos][ACTION_IDX[action]]
                new_value = old_value + ALPHA * (reward + GAMMA * np.max(Q[new_pos]) - old_value)
                Q[pos][ACTION_IDX[action]] = new_value

                total_change += abs(new_value - old_value)
                pos = new_pos

                # Stop if we hit terminal tile
                if tile == "goal" or tile in ["fire", "elephant"]:
                    break

        # Check convergence
        avg_change = total_change / (GRID_SIZE * GRID_SIZE * 4)
        if avg_change < convergence_threshold:
            print(f"Converged after {episode} episodes")
            break

    return Q

def create_grid():
    grid = [["empty" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    start = (random.randint(0, 7), random.randint(0, 7))
    goal = (random.randint(0, 7), random.randint(0, 7))
    while goal == start:
        goal = (random.randint(0, 7), random.randint(0, 7))
    grid[start[0]][start[1]] = "start"
    grid[goal[0]][goal[1]] = "goal"
    for _ in range(10):
        x, y = random.randint(0, 7), random.randint(0, 7)
        if grid[x][y] == "empty":
            grid[x][y] = random.choice(["fire", "elephant", "popcorn"])
    return grid, start, goal

# Pygame integration and rendering should be added outside this segment in the main file.
# The above code is a simplified version of the Q-learning algorithm for training multiple clowns in a grid environment.