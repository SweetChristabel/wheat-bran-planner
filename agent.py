import random
import numpy as np
from parameters import SEED



class QLearningAgent:
    def __init__(self):
        self.rngesus = random.Random(SEED)
        self.history = []


    def choose(self, state):

    #plan-ahead - only option for initial decision

    #plan-emergency - only option for final decision

    #cancel-orders-ahead

    #cancel-orders-emergency

    #do-nothing