import random
import numpy as np
import parameters as p
from wheatmill import WheatMill
from typing import NamedTuple

class EpisodeOutcome(NamedTuple):
    start_state: tuple #generated state
    plan_week_action: tuple 
    plan_week_state: tuple #state after planning the week
    startq_reward: int #all accumulated rewards for the week planner
    running_states: list #all the running states prior to action
    running_actions: list #all actions taken by running_q during the ep
    running_reward: int #accumulated running rewards
    endofweek_state: tuple #pre-weekend decision state
    endofweek_action: int #final_q action
    final_state: tuple
    weekend_prod_reward: int #Final reward for PREVIOUS WEEK's decision
    exploration_count: int #amount of explore actions taken






class QLearningAgent:
    def __init__(self):
        self.rngesus = random.Random(p.SEED)
        self.history = []
        self.start_q = np.zeros((5, 5, 15)) #tank states * orders * options
        self.running_q = np.zeros((5, 9, 10, 5)) #tank states * order states * steps * options
        self.final_q = np.zeros((5, 5)) #tank states * options
        self.weekend_state = None #for final_q learning that carries over episode limits
        self.weekend_action = None
        self.weekend_reward = 0


    def choose(self, state, q, flag = False, learn = True):
        #EXPLORE
        if self.rngesus.random() < p.EPSILON and learn:
            if flag: #start of week choice
                return self.rngesus.randrange(15), True
            else:
                return self.rngesus.randrange(5), True

        #EXPLOIT
        values = q[state]
        if flag: #start of week choice
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

    def decode_week_planning_action(self, action):
        outlook = action // 5
        timing = action % 5
        return outlook, timing

    def episode(self, env, learn = True):
        start_state = env.generate_episode_start()
        startq_reward_tracker = 0
        exploration_tracker = 0
        start_choice, explored = self.choose(start_state, self.start_q, flag = True, learn = learn)
        start_action = self.decode_week_planning_action(start_choice)
        if explored:
            exploration_tracker +=1
        
        plan_week_state = env.plan_week(start_action)
        stock, orders, time_left, weekend_prod_reward = env.step(0) #first iteration - no decision is made here
        state = stock, orders, time_left

        self.weekend_reward += weekend_prod_reward

        if learn:
            #FINAL Q LEARNING, Monte Carlo style
            if self.weekend_state is not None: #action and reward may be 0, returning False
                old = self.final_q[self.weekend_state][self.weekend_action]
                new = old + p.ALPHA * (
                    self.weekend_reward - old
                )
                self.final_q[self.weekend_state][self.weekend_action] = new


        running_actions = []
        running_states = []

        #MID WEEK LOOP
        while state[-1] > 0: #time_left
            running_states.append(state)
            action, explored = self.choose(state, self.running_q, learn = learn)
            running_actions.append(action)
            if explored:
                exploration_tracker +=1
            stock, orders, time_left, reward = env.step(action)
            new_state = stock, orders, time_left
            startq_reward_tracker += reward
            if learn:
            #RUNNING Q LEARNING    
                if time_left == 0:
                    best_next = self.final_q[new_state[0]].max()
                else:
                    best_next = self.running_q[new_state].max()

                old = self.running_q[state][action]
                new = old + p.ALPHA * (
                    reward 
                    + p.GAMMA * best_next
                    - old
                )
                self.running_q[state][action] = new

            state = new_state


        self.weekend_state = state[0]
        #Weekend choice - whether to order emergency emptying and if so, how many. Max 4.
        self.weekend_action, explored = self.choose(self.weekend_state, self.final_q, learn = learn)
        if explored:
            exploration_tracker +=1
        stock, orders, time, reward = env.step(self.weekend_action)
        final_state = stock, orders, time
        running_reward = startq_reward_tracker
        startq_reward_tracker += reward #Both start and final q learn from this
        self.weekend_reward = reward
        if learn:
        #START Q LEARNING, Monte Carlo style
            old = self.start_q[start_state][start_action]
            new = old + p.ALPHA * (
                startq_reward_tracker - old
            )
            self.start_q[start_state][start_action] = new
        outcome = EpisodeOutcome(start_state, start_action, plan_week_state, startq_reward_tracker, running_states, running_actions, running_reward, self.weekend_state, self.weekend_action, final_state, weekend_prod_reward, exploration_tracker)
        return outcome


    def train(self, episodes):
        training_environment = WheatMill()
        for _ in range(episodes):
            outcome = self.episode(training_environment)
            self.history.append(outcome)


    def greedy_path(self, demo_length = 5):
        demo_history =[]
        demo_environment = WheatMill()
        for _ in range(demo_length):
            outcome = self.episode(demo_environment, learn = False)
            demo_history.append(outcome)
        return demo_history

        
