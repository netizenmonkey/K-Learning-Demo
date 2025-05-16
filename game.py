import pygame
import random
import numpy as np



# Constants
GRID_SIZE = 5
TILE_SIZE = 64
SCREEN_SIZE = GRID_SIZE * TILE_SIZE
FPS = 10
MAX_EPISODES = 100


speed = 1.0  # 1x speed


# Q-learning settings
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_IDX = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE+200, SCREEN_SIZE + 100))
pygame.display.set_caption("Andrew The Clown!")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 32)

# Load and resize images
tile_images = {
    "start": pygame.image.load("Star.png").convert_alpha(),
    "goal": pygame.image.load("Cannon.png").convert_alpha(),
    #"elephant": pygame.image.load("Elephant-stamp.png").convert_alpha(),
    "fire": pygame.image.load("Fire.png").convert_alpha(),
    "popcorn": pygame.image.load("Popcorn.png").convert_alpha(),
    "empty": pygame.image.load("Tile.png").convert_alpha(),
    "andrew": pygame.image.load("Andrew.png").convert_alpha()
}

for k in tile_images:
    tile_images[k] = pygame.transform.smoothscale(tile_images[k], (TILE_SIZE, TILE_SIZE))

# Rewards
REWARDS = {
    "start": 0,
    "goal": 10,
    "fire": -10,
    "popcorn": 5,
    "empty": -1
}

elephant_stand = pygame.image.load("Elephant-stand.png").convert_alpha()
elephant_stamp = pygame.image.load("Elephant-stamp.png").convert_alpha()
elephant_stand = pygame.transform.smoothscale(elephant_stand, (TILE_SIZE, TILE_SIZE))
elephant_stamp = pygame.transform.smoothscale(elephant_stamp, (TILE_SIZE, TILE_SIZE))
elephant_anim_frame = 0



def create_grid():
    grid = [["empty" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    start = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    goal = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    while goal == start:
        goal = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    grid[start[0]][start[1]] = "start"
    grid[goal[0]][goal[1]] = "goal"
    for _ in range(5):
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if grid[x][y] == "empty":
            grid[x][y] = random.choice(["fire", "elephant", "popcorn"])
    return grid, start, goal

def move(pos, action, grid):
    x, y = pos
    new_x, new_y = x, y
    if action == 'U': new_x = max(x - 1, 0)
    elif action == 'D': new_x = min(x + 1, GRID_SIZE - 1)
    elif action == 'L': new_y = max(y - 1, 0)
    elif action == 'R': new_y = min(y + 1, GRID_SIZE - 1)
    
    # Block if elephant
    if grid[new_x][new_y] == "elephant":
        return pos  # no movement
    return new_x, new_y


def draw_heatmap(Q):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pos = (row, col)
            if pos in Q:
                value = np.max(Q[pos])
                color = get_heat_color(value)
                overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
                overlay.set_alpha(100)
                overlay.fill(color)
                screen.blit(overlay, (col * TILE_SIZE, row * TILE_SIZE))

def get_heat_color(value):
    if value < -5:
        return (255, 0, 0)
    elif value < 0:
        return (255, 165, 0)
    elif value < 5:
        return (255, 255, 0)
    else:
        return (0, 255, 0)

def draw_grid(grid, pos, Q, show_path, episode, start, goal):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            tile = grid[row][col]
            if tile == "elephant":
                if elephant_anim_frame // 30 % 2 == 0:
                        screen.blit(elephant_stand, (col * TILE_SIZE, row * TILE_SIZE))
                else:
                        screen.blit(elephant_stamp, (col * TILE_SIZE, row * TILE_SIZE))
            else:
                    screen.blit(tile_images[tile], (col * TILE_SIZE, row * TILE_SIZE))

    if not show_path:
        draw_heatmap(Q)

    screen.blit(tile_images["andrew"], (pos[1] * TILE_SIZE, pos[0] * TILE_SIZE))

    if show_path and episode > 0:
        draw_learned_path(Q, start, goal, grid)

def draw_button(text, x, y, w, h, color):
    pygame.draw.rect(screen, color, (x, y, w, h))
    label = font.render(text, True, (0, 0, 0))
    screen.blit(label, (x + 10, y + 10))
    return pygame.Rect(x, y, w, h)

def draw_learned_path(Q, start, goal, grid):
    pos = start
    visited = set()
    path = []

    while pos != goal and pos in Q and pos not in visited:
        visited.add(pos)
        best_action = ACTIONS[np.argmax(Q[pos])]
        next_pos = move(pos, best_action, grid)

        if next_pos in visited:
            return
        path.append((pos, next_pos))
        pos = next_pos

    if pos != goal:
        return

    for start_pos, end_pos in path:
        start_pixel = (start_pos[1] * TILE_SIZE + TILE_SIZE // 2, start_pos[0] * TILE_SIZE + TILE_SIZE // 2)
        end_pixel = (end_pos[1] * TILE_SIZE + TILE_SIZE // 2, end_pos[0] * TILE_SIZE + TILE_SIZE // 2)
        pygame.draw.line(screen, (0, 0, 255), start_pixel, end_pixel, 4)

def main():
    
    global speed
    success_count = 0
    Q = {}
    grid, start, goal = create_grid()
    original_grid = [row[:] for row in grid]
    andrew_pos = start
    training = False
    episode = 0
    show_path = False
    score = 0
    high_score = float('-inf')

    while True:
        global elephant_anim_frame
        elephant_anim_frame += 1
        screen.fill((0, 0, 0))
        draw_grid(grid, andrew_pos, Q, show_path, episode, start, goal)

        start_btn = draw_button("Start Training", 10, SCREEN_SIZE + 10, 180, 50, (200, 200, 200))
        path_btn = draw_button("Show Learned Path", 210, SCREEN_SIZE + 10, 220, 50, (180, 255, 180))

        score_label = font.render(f"Score: {score}", True, (255, 255, 255))
        high_score_label = font.render(f"High: {high_score}", True, (255, 255, 255))
        screen.blit(score_label, (10, SCREEN_SIZE + 70))
        screen.blit(high_score_label, (220, SCREEN_SIZE + 70))


        slider_label = font.render("Speed:", True, (255, 255, 255))
        screen.blit(slider_label, (470, SCREEN_SIZE + 10))
        pygame.draw.rect(screen, (200, 200, 200), (540, SCREEN_SIZE + 25, 100, 10))  # track
        slider_pos = int(540 + (speed - 0.5) * 100)
        pygame.draw.circle(screen, (100, 100, 255), (slider_pos, SCREEN_SIZE + 30), 8)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Start button clicked
                if start_btn.collidepoint(event.pos):
                    Q = {}
                    grid, start, goal = create_grid()
                    original_grid = [row[:] for row in grid]
                    andrew_pos = start
                    training = True
                    episode = 0
                    score = 0
                    show_path = False
                    success_count = 0  # Reset success counter

                # Path button clicked (only after 10 successful reaches)
                elif success_count >= 10 and path_btn.collidepoint(event.pos):
                    show_path = True
                    training = False
                    andrew_pos = start

                    # Animate learned path
                    pos = start
                    visited = set()
                    while pos != goal and pos in Q and pos not in visited:
                        visited.add(pos)
                        best_action = ACTIONS[np.argmax(Q[pos])]
                        next_pos = move(pos, best_action, grid)
                        if next_pos in visited:
                            break
                        andrew_pos = next_pos
                        draw_grid(grid, andrew_pos, Q, show_path=True, episode=episode, start=start, goal=goal)
                        pygame.display.flip()
                        pygame.time.wait(int(500 / speed))  # Speed-adjusted animation delay
                        pos = next_pos

                # Speed slider clicked
                elif 540 <= event.pos[0] <= 640 and SCREEN_SIZE + 20 <= event.pos[1] <= SCREEN_SIZE + 40:
                    rel_x = event.pos[0] - 540
                    speed = 0.5 + rel_x / 100  # Range from 0.5x to 1.5x


        if training:
            
            if andrew_pos not in Q:
                Q[andrew_pos] = np.zeros(4)

            action = random.choice(ACTIONS) if random.random() < EPSILON else ACTIONS[np.argmax(Q[andrew_pos])]
            new_pos = move(andrew_pos, action, grid)

            tile = grid[new_pos[0]][new_pos[1]]

            if tile == "popcorn":
                reward = REWARDS[tile]
                score += reward
                grid[new_pos[0]][new_pos[1]] = "empty"
            elif tile == "fire":
                reward = REWARDS[tile]
                score = 0
                overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
                overlay.set_alpha(180)
                overlay.fill((255, 0, 0))
                screen.blit(overlay, (new_pos[1] * TILE_SIZE, new_pos[0] * TILE_SIZE))
                screen.blit(tile_images["andrew"], (new_pos[1] * TILE_SIZE, new_pos[0] * TILE_SIZE))
                pygame.display.flip()
                pygame.time.wait(500)
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        if original_grid[i][j] == "popcorn" and grid[i][j] == "empty":
                            grid[i][j] = "popcorn"
                episode += 1
                andrew_pos = start
                continue
            else:
                reward = REWARDS[tile]

            if new_pos not in Q:
                Q[new_pos] = np.zeros(4)

            Q[andrew_pos][ACTION_IDX[action]] += ALPHA * (
                reward + GAMMA * np.max(Q[new_pos]) - Q[andrew_pos][ACTION_IDX[action]]
            )

            andrew_pos = new_pos

            if tile == "goal":
                overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
                success_count += 1
                overlay.set_alpha(180)
                overlay.fill((0, 255, 0))
                screen.blit(overlay, (andrew_pos[1] * TILE_SIZE, andrew_pos[0] * TILE_SIZE))
                screen.blit(tile_images["andrew"], (andrew_pos[1] * TILE_SIZE, andrew_pos[0] * TILE_SIZE))
                pygame.display.flip()
                pygame.time.wait(500)
                high_score = max(high_score, score)
                episode += 1
                andrew_pos = start

        pygame.display.flip()
        clock.tick(FPS)

main()
