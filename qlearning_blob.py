import pygame
import random
import numpy as np 

height = 200
width = 200
window = pygame.display.set_mode((width, height))
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
pygame.display.set_caption("blob")
fps = 30
clock = pygame.time.Clock()
pygame.init()

class Blob:
    def __init__(self, color, x, y):
        self.x = x # random.randrange(0, width, 20)
        self.y = y # random.randrange(0, height, 20)
        self.color = color

    def draw(self):
        pygame.draw.rect(window, self.color, [self.x, self.y, 20, 20])

    def move(self, action):
        if action == 0: self.y -= 20 # move UP
        elif action == 1: self.y += 20 # move DOWN
        elif action == 2: self.x -= 20 # move LEFT
        elif action == 3: self.x += 20 # move RIGHT
        else: pass
        # collision detection
        if self.x <= 0: self.x = 0
        if self.x >= width: self.x = width-20
        if self.y <= 0: self.y = 0
        if self.y >= height: self.y = height-20

class Game:
    def __init__(self):
        self.agent = Blob(white, random.randrange(0, width, 20), random.randrange(0, height, 20))
        self.goal = Blob(green, 180, 180)
        self.enemy = Blob(red, random.randrange(0, width, 20), random.randrange(0, width, 20))
        self.action_space = [0, 1, 2, 3]
        self.done = False
        self.observation_space = []
        for x in range(0, width, 20):
            for y in range(0, height, 20):
                self.observation_space.append([x, y])
    
        self.state = [self.agent.x, self.agent.y] 

    def render(self):
        global run
        window.fill(black)
        self.agent.draw()
        self.goal.draw()
        self.enemy.draw()
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                return True 
        pygame.display.update()
        clock.tick(fps)

    def step(self, action):
        self.agent.move(action) 
        if self.agent.x == self.goal.x and self.agent.y == self.goal.y: 
            self.reward = 1
            self.done = True 
        elif self.agent.x == self.enemy.x and self.agent.y == self.enemy.y: 
            self.reward = -1
            self.done = True 
        else: 
            self.reward = 0
        self.next_step = [self.agent.x, self.agent.y]
        return self.next_step, self.reward, self.done

    def reset(self):
        self.agent = Blob(white, random.randrange(0, width, 20), random.randrange(0, height, 20))
        self.goal = Blob(green, 180, 180)
        self.action_space = [0, 1, 2, 3]
        self.done = False
        return [self.agent.x, self.agent.y] 


lr = 0.1
epsilon = 1
epsilon_decay_rate = 0.0001
epsilon_max = 1
epsilon_min = 0.01
discount = 0.99
max_steps = 100
episodes = 500000
all_rewards = []
env = Game()
q_table = np.zeros((len(env.observation_space), len(env.action_space)))

for episode in range(episodes):
    currrent_reward = 0
    state = env.reset()
    for step in range(max_steps):
        # env.render()
        if random.random() >= epsilon:
            action = np.argmax(q_table[env.observation_space.index(state), :])
        else:
            action = random.choice(env.action_space)

        new_state, reward, done = env.step(action)
        q_table[env.observation_space.index(state), action] = ((1 - lr) * q_table[env.observation_space.index(state), action]) + (lr * (reward + discount * np.max(q_table[env.observation_space.index(new_state), :])))
        state = new_state
        currrent_reward += reward
        if done: break

    # epsilon decay 
    epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-epsilon_decay_rate * episode)
    all_rewards.append(currrent_reward)

rewards = 0
for r in range(len(all_rewards)):
    rewards += all_rewards[r] 
    if r % 1000 == 0:
        avg_reward_per_1000_eps = rewards/1000 
        print(avg_reward_per_1000_eps)
        rewards = 0

for episode in range(10):
    state = env.reset()
    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[env.observation_space.index(state), :])
        new_state, _, done = env.step(action)
        state = new_state
        if done: break

pygame.quit()
quit()