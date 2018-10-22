import numpy as np
import gym
from gym.spaces import Discrete

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Env

import random
from tabulate import tabulate

# Actions: 0 = up, 1 = right, 2 = down, 3 = left
UP, RIGHT, DOWN, LEFT = range(4)
ENV_NAME = '2048-v0'

def n_ify(l):
    return [i.n for i in l]

class Tile:
    def __init__(self, n):
        self.n = n
        self.merged = False

    def __add__(self, other):
        if self.n == 0:
            return 0
        if other.n == 0:
            self.n, other.n = 0, self.n
            return -1
        if not self.merged and not other.merged and self.n == other.n:
            self.n, other.n = 0, self.n * 2
            other.merged = True
            return other.n * 2
        return 0

    def __repr__(self):
        return f"{self.n}:{self.merged}"
            

class Env2048(Env):
    def __init__(self, size, **kwargs):
        Env.__init__(self, **kwargs)
        self.size = size
        self.action_space = Discrete(4)
        self.observation_space = np.zeros((4, 4))
        self.done = False
        self.last_act = None
        self.score = 0
        self.steps = 0
        
    def reset(self):
        self.observation_space = np.zeros((4, 4))
        self.addn(2)
        self.done = False
        self.score = 0
        self.steps = 0
        return self.observation_space
            
    def add(self):
        empty = np.transpose(np.where(self.observation_space==0))
        if len(empty) == 0:
            self.done = True
            return "full"
        x, y = random.choice(empty)
        self.observation_space[x][y] = np.random.choice([2, 4])

    def addn(self, n):
        for _ in range(n):
            self.add()

    def fill(self):
        self.addn(self.size ** 2)
        
    def rot(self):
        self.observation_space = np.rot90(self.observation_space)

    def rotn(self, n):
        for _ in range(n):
            self.rot()
        
    def step(self, action):
        self.last_act = action
        if action == UP:
            self.rotn(3)
            score = self.right()
            self.rot()
        elif action == DOWN:
            self.rot()
            score = self.right()
            self.rotn(3)
        elif action == LEFT:
            self.rotn(2)
            score = self.right()
            self.rotn(2)
        else:
            score = self.right()
        # if score != 0:
        self.add()
        self.score += score
        self.steps += 1
        if self.steps > 1000:
            self.done = True
        else:
            self.done = not self.can_move()
        return self.observation_space, score, self.done, {'left':   action==LEFT,
                                                          'right':  action==RIGHT,
                                                          'down':   action==DOWN,
                                                          'up':     action==UP,
                                                          'failed': score==0,
                                                          'score':  self.score,
                                                          'steps':  self.steps}
    
    def _slide_idx(self, idx):
        gained = 0
        moved = False
        _row = self.observation_space[idx]
        row = list(map(Tile, _row))
        n = 0
        merged = []
        while n != self.size - 1:
            n = 0
            for i in range(0, self.size - 1):
                val = row[i] + row[i + 1]
                if val == 0:
                    n += 1
                elif val != -1:
                    gained += val
                    
        self.observation_space[idx] = n_ify(row)
        return gained

    def right(self):
        gained = 0
        for idx in range(self.size):
            gained += self._slide_idx(idx)
        return gained

    def can_move(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.observation_space[i][j] == 0:
                    return True
                if j < self.size - 1 and self.observation_space[i][j] == self.observation_space[i][j + 1]:
                    return True
                if i < self.size - 1 and self.observation_space[i][j] == self.observation_space[i + 1][j]:
                    return True
        return False

    def render(self, mode='human'):
        print("Action:", self.last_act, 'Score:', self.score)
        print(tabulate(self.observation_space, tablefmt="fancy_grid"))

    def close(self): ...
    
def test_env():
    env = Env2048(4)
    env.reset()
    env.addn(10)
    print("ORIGINAL")
    env.render()

    print("STEPPING LEFT")
    env.step(LEFT)
    env.render()
##    print("ADDING")
##    env.add()
##    env.render()

    print("STEPPING RIGHT")
    env.step(RIGHT)
    env.render()
##    print("ADDING")
##    env.add()
##    env.render()

    print("STEPPING UP")
    env.step(UP)
    env.render()
##    print("ADDING")
##    env.add()
##    env.render()

    print("STEPPING DOWN")
    env.step(DOWN)
    env.render()
##    print("ADDING")
##    env.add()
##    env.render()

def main():
    # Get the environment and extract the number of actions.
    env = Env2048(4) 
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, verbose=1) #, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    input("Done: ")
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)

main()
