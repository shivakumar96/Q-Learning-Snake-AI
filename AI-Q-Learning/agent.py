import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from NeuralNet import NNModel
from helper import plot
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


"""
State = [1,0,1,.1] all binary 
state[0] respresents 1 if going straight is danger 
state[1] respresents 1 if going right is danger 
state[2] respresents 1 if going left is danger // these are the border cases
state[3] respresents 1 if snake is going left direction
state[4] respresents 1 if snake is going right direction
state[5] respresents 1 if snake is going up direction
state[6] respresents 1 if snake is going down direction
state[7] respresents 1 if food is left side of the sankses direction 
state[8] respresents 1 if food is right side of the sankses direction 
state[9] respresents 1 if food is up side of the sankses direction 
state[10] respresents 1 if food is down side of the sankses direction 
"""

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = NNModel(lr=LR,action_size=3,input_dim=11)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) #returs a list of tuples
        else:
            mini_sample = self.memory
        
        states,actions,rewards,next_states, dones = zip(*mini_sample)
        states = np.array(states).reshape(len(states),11)
        next_states = np.array(next_states).reshape(len(next_states),11)
        actions =  np.array(actions).reshape(len(actions),3)
        rewards =  np.array(rewards).reshape(len(rewards),1)
        dones =  np.array(dones).reshape(len(dones),1)
        
        q_eval  =  self.model.model.predict(states)
        q_next =  self.model.model.predict(next_states)
        
        q_target = np.copy(q_eval)

        for idx in range(len(mini_sample)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * np.max(q_next[idx])

            q_target[idx][np.argmax(actions[idx]).item()] = Q_new
        # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        #q_target[batch_index][np.argmax(q_next[batch_index],axis=1)] =  rewards + DISCOUNT_FACT* np.max(q_next[batch_index],axis=1)*dones
        
        self.model.model.train_on_batch(states, q_target)
        #print(self.model.model.get_weights())
        
        #self.epsilion = self.epsilion - len(mini_sample) if self.epsilion - len(mini_sample)  > 0 else 0
        #train

    def train_short_memory(self, state, action, reward, next_state, done):
        state = np.array(state).reshape(1,11)
        next_state = np.array(next_state).reshape(1,11)
        action =  np.array(action).reshape(1,3)
        reward =  np.array(reward).reshape(1,1)
        done =  np.array(done).reshape(1,1)
        
        q_eval  =  self.model.model.predict(state)
        q_next =  self.model.model.predict(next_state)
        
        q_target = np.copy(q_eval)
        
        q_target[0][np.argmax(action[0]).item()] =  reward[0] + self.gamma* np.max(q_next[0])*done[0]

        self.model.model.train_on_batch(state, q_target)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 50 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            
            state0 = np.array(state).reshape(1,11)
            actions = self.model.model.predict(state0,verbose = 0)
            action_indx = np.argmax(actions).item()
            final_move[action_indx] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while agent.n_games <=130:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
    
    print('Game', agent.n_games, 'Score', score, 'Record:', record)

    plot_scores.append(score)
    total_score += score
    mean_score = total_score / agent.n_games
    plot_mean_scores.append(mean_score)
    plot(plot_scores, plot_mean_scores)   
    
    with open('Agent_TensorFlow_scores.csv', 'w') as f:
        for line in plot_scores:
            f.write(f"{line}\n")
    time.sleep(30)
    
    
    


if __name__ == '__main__':
    train()