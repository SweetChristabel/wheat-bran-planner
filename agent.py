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
    emergency_dispatches: int
    cancellations: int
    tank_full_events: int
    failed_dispatches: int





class RandomAgent:
    def __init__(self):
        self.rngesus = random.Random(p.DEFAULT_AGENT_SEED)


    def choose(self, flag = False):
        #EXPLORE
        if flag: #start of week choice
            return self.rngesus.randrange(15)
        else:
            return self.rngesus.randrange(5)


    def decode_week_planning_action(self, action):
        outlook = action // 5
        timing = action % 5
        return outlook, timing

    def episode(self, env):
        start_state = env.generate_episode_start()
        startq_reward_tracker = 0
        start_choice = self.choose(flag = True)
        start_action = self.decode_week_planning_action(start_choice)
        
        plan_week_state = env.plan_week(start_action)
        stock, orders, time_left, weekend_prod_reward = env.step(0) #first iteration - no decision is made here
        state = stock, orders, time_left


        running_actions = []
        running_states = []

        #MID WEEK LOOP
        while state[-1] > 0: #time_left
            running_states.append(state)
            action = self.choose()
            running_actions.append(action)
            stock, orders, time_left, reward = env.step(action)
            new_state = stock, orders, time_left
            startq_reward_tracker += reward

            state = new_state


        weekend_state = state[0]
        #Weekend choice - whether to order emergency emptying and if so, how many. Max 4.
        weekend_action = self.choose()
        stock, orders, time, reward = env.step(weekend_action)
        final_state = stock, orders, time
        running_reward = startq_reward_tracker
        startq_reward_tracker += reward
        weekend_prod_reward += reward
        outcome = EpisodeOutcome(
            start_state = start_state, plan_week_action = start_action, plan_week_state = plan_week_state, 
            startq_reward = startq_reward_tracker, running_states = running_states, running_actions = running_actions, 
            running_reward = running_reward, endofweek_state = weekend_state, endofweek_action = weekend_action,
            final_state = final_state, weekend_prod_reward = weekend_prod_reward, exploration_count = None,
            emergency_dispatches=env.emergency_dispatches,
            cancellations=env.cancellations,
            tank_full_events=env.tank_full_events,
            failed_dispatches=env.failed_dispatches
            )
        return outcome

    def run(self, env, episodes):
        return [self.episode(env) for _ in range(episodes)]


class RuleBasedAgent:

    def choose_week(self, state):
        stock, dist = state

        if stock <= 1: #low
            outlook = 2 #pessimistic
        elif stock == 2: #normal
            outlook = 0 #normal
        else:           #high
            outlook = 1 #optimistic

        if dist == 0:   #early
            timing = 2  #late
        elif dist == 1: #convex
            timing = 4  #concave
        elif dist == 2: #balanced
            timing = 0  #balanced
        elif dist == 3: #concave
            timing = 3  #convex
        else:           #late
            timing = 1  #early

        return outlook, timing


    def choose_daily(self, state):
        stock, orders, time_left = state

        # Critically high stock
        if stock == 4:
            return 3  # Emergency dispatch

        # Excess stock and insufficient demand
        elif stock >= 3 and orders <= 1:
            if time_left >= 5:
                return 1  # Plan ahead
            else:
                return 3  # Emergency dispatch

        # Low stock and high demand
        elif stock <= 1 and orders >= 4:
            if time_left >= 5:
                return 2  # Cancel later order
            else:
                return 4  # Cancel imminent order

        # No intervention necessary
        else:
            return 0
        
    def choose_weekend(self, state):
        if state <= 1:
            return 0
        return state #dispatch as many emergency orders as the tank state says


    def decode_week_planning_action(self, action):
        outlook = action // 5
        timing = action % 5
        return outlook, timing

    def episode(self, env):
        start_state = env.generate_episode_start()
        start_action = self.choose_week(start_state)
        startq_reward_tracker = 0
        plan_week_state = env.plan_week(start_action)
        stock, orders, time_left, weekend_prod_reward = env.step(0) #first iteration - no decision is made here
        state = stock, orders, time_left


        running_actions = []
        running_states = []

        #MID WEEK LOOP
        while state[-1] > 0: #time_left
            running_states.append(state)
            action = self.choose_daily(state)
            running_actions.append(action)
            stock, orders, time_left, reward = env.step(action)
            new_state = stock, orders, time_left
            startq_reward_tracker += reward
            state = new_state


        weekend_state = state[0]
        #Weekend choice - whether to order emergency emptying and if so, how many. Max 4.
        weekend_action = self.choose_weekend(weekend_state)
        stock, orders, time, reward = env.step(weekend_action)
        final_state = stock, orders, time
        running_reward = startq_reward_tracker
        startq_reward_tracker += reward
        weekend_prod_reward += reward
        outcome = EpisodeOutcome(
            start_state = start_state, plan_week_action = start_action, plan_week_state = plan_week_state, 
            startq_reward = startq_reward_tracker, running_states = running_states, running_actions = running_actions, 
            running_reward = running_reward, endofweek_state = weekend_state, endofweek_action = weekend_action,
            final_state = final_state, weekend_prod_reward = weekend_prod_reward, exploration_count = None,
            emergency_dispatches=env.emergency_dispatches,
            cancellations=env.cancellations,
            tank_full_events=env.tank_full_events,
            failed_dispatches=env.failed_dispatches
            )
        return outcome

    def run(self, env, episodes):
        return [self.episode(env) for _ in range(episodes)]

class QLearningAgent:
    def __init__(self):
            self.rngesus = random.Random(p.DEFAULT_AGENT_SEED)
            self.history = []
            self.start_q = np.zeros((5, 5, 15)) #tank states * orders * options
            self.running_q = np.zeros((5, 6, 10, 5)) #tank states * order states * steps * options
            self.final_q = np.zeros((5, 5)) #tank states * options
            self.weekend_state = None #for final_q learning that carries over episode limits
            self.weekend_action = None
            self.weekend_reward = 0
            self.epsilon = p.EPSILON_START
            self.epsilon_plot = []


    def choose(self, state, q, flag = False, learn = True):
        #EXPLORE
        if self.rngesus.random() < self.epsilon and learn:
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

        #START OF WEEK DECISION

        start_choice, explored = self.choose(start_state, self.start_q, flag = True, learn = learn)
        start_action = self.decode_week_planning_action(start_choice)
        if explored:
            exploration_tracker +=1
        
        plan_week_state = env.plan_week(start_action)

        #WEEKEND PRODUCTION

        stock, orders, time_left, weekend_prod_reward = env.step(0) #first iteration - no decision is made here so input is "do nothing"
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


        #WEEKEND CHOICE - whether to order emergency emptying and if so, how many. Max 4.

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
            old = self.start_q[start_state][start_choice]
            new = old + p.ALPHA * (
                startq_reward_tracker - old
            )
            self.start_q[start_state][start_choice] = new
        outcome = EpisodeOutcome(
            start_state = start_state, plan_week_action = start_action, plan_week_state = plan_week_state, 
            startq_reward = startq_reward_tracker, running_states = running_states, running_actions = running_actions, 
            running_reward = running_reward, endofweek_state = self.weekend_state, endofweek_action = self.weekend_action,
            final_state = final_state, weekend_prod_reward = weekend_prod_reward, exploration_count = exploration_tracker,
            emergency_dispatches=env.emergency_dispatches,
            cancellations=env.cancellations,
            tank_full_events=env.tank_full_events,
            failed_dispatches=env.failed_dispatches
            )
        return outcome

    def train(self, episodes):
        training_environment = WheatMill()
        epsilon_decay = (p.EPSILON_MIN / p.EPSILON_START) ** (1 / episodes)
        for i in range(episodes):
            outcome = self.episode(training_environment)
            self.epsilon = max(
                p.EPSILON_MIN,
                self.epsilon * epsilon_decay
            )

            if i % 1000 == 0:
                self.history.append(outcome)
                self.epsilon_plot.append(self.epsilon)

        return self.history

    def reset_episode_memory(self):
        self.weekend_state = None
        self.weekend_action = None
        self.weekend_reward = 0

    def greedy_path(self, env, episodes):
        demo_history =[]
        for _ in range(episodes):
            self.reset_episode_memory()
            outcome = self.episode(env, learn = False)
            demo_history.append(outcome)
        return demo_history