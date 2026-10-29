import random
import numpy as np
import parameters as p
from wheatmill import WheatMill



class QLearningAgent:
    def __init__(self):
        self.rngesus = random.Random(p.SEED)
        self.history = []
        self.start_q = np.zeros((5, 5, 15)) #tank states * orders * options
        self.running_q = np.zeros((5,9,(p.WORKDAYS - 1)*2, 5)) #tank states * order states * steps * options
        self.final_q = np.zeros((5,5)) #tank states * options
        self.weekend_state = None #for final_q learning that carries over episode limits
        self.weekend_action = None
        self.weekend_reward = 0


    def choose(self, state, q):
        #EXPLORE
        if self.rngesus.random() < p.EPSILON:
            if q == self.start_q: #start of week choice
                return self.rngesus.randrange(9), True
            else:
                return self.rngesus.randrange(5), True

        #EXPLOIT
        values = q[state]
        if q == self.start_q: #start of week choice
            best_actions = [
                action for action in range(15)
                if values[action] == values.max()
            ]
        else:
            best_actions = [
                action for action in range(5)
                if values[action] == values.max()
            ]
        return self.rngesus.choice(best_actions), False


    def episode(self, env, show = False): #TODO IMPLEMENT SHOW
        start_state = env.generate_episode_start()
        startq_reward_tracker = 0
        start_action, explored = self.choose(start_state, self.start_q)

        plan_week_state = env.plan_week(start_action)
        stock, orders, time_left, weekend_reward = env.step(0) #first iteration - no decision is made here
        state = stock, orders, time_left

        self.weekend_reward += weekend_reward

        #FINAL Q LEARNING, Monte Carlo style
        if self.weekend_state is not None: #action and reward may be 0, returning False
            old = self.final_q[self.weekend_state][self.weekend_action]
            new = old + p.ALPHA * (
                self.weekend_reward - old
            )
            self.final_q[self.weekend_state][self.weekend_action] = new



        #MID WEEK LOOP
        while state[-1] > 0: #time_left
            action, explored = self.choose(state, self.running_q)

            stock, orders, time_left, reward = env.step(action)
            new_state = stock, orders, time_left
            startq_reward_tracker += reward

            #RUNNING Q LEARNING    
            old = self.running_q[state][action]
            best_next = self.running_q[new_state].max()
            new = old + p.ALPHA * (
                reward 
                + p.GAMMA * best_next
                - old
            )
            self.running_q[state][action] = new

            state = new_state


        self.weekend_state = state[0]
        #Weekend choice - whether to order emergency emptying and if so, how many. Max 4.
        self.weekend_action, explored = self.choose(self.weekend_state, self.final_q)
        reward = env.step(self.weekend_action)[-1]
        startq_reward_tracker += reward #Both start and final q learn from this
        self.weekend_reward = reward

        #START Q LEARNING, Monte Carlo style
        old = self.start_q[start_state][start_action]
        new = old + p.ALPHA * (
            startq_reward_tracker - old
        )
        self.start_q[start_state][start_action] = new



