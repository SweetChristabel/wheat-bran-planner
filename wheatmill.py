import random
import parameters as p
from math import ceil

class TanksFullException(Exception): 
    pass

#TLDR
    # 5 total tank states
# 9 possible decisions at the beginning of the week * tank states
# 5 possible decisions mid-week * tank states * orders * step number
    # 9 possible orders for this+next iteration
    # (workdays - 1) * 2 mid-ep decision stages/steps, default 8
# 5 possible end of week decisions * tank states

#TODO: Consider how to communicate state with agent
#TODO: Consider how to communicate rewards with agent
#TODO: Consider how agent communicates action with env
#TODO: Look at unused functions and variables if any

class WheatMill():

    def __init__(self, seed= p.SEED, workdays=p.WORKDAYS, tank_capacity=p.TANK_CAPACITY):
        self.current_stock = 0
        self.tank_capacity = tank_capacity
        self.rngesus = random.Random(seed)
        self.orders = [[0, 0] for _ in range(workdays)] #pre-orders and overflow orders for each workday
        self.weekend = 7 - workdays
        self.workdays = workdays
        self.current_time = (0,-1) #(weekday, morning/evening): -1 signals episode start


    #region ORDER HANDLING

    def generate_episode_start(self):
        self.generate_preorders()
        stock = self.calculate_stock_level(foresight = False)
        return stock, self.orders

    def generate_preorders(self): #a workdays list with random amounts per day
        self.orders = [[self.rngesus.choices(list(p.ORDER_RATE.keys()), weights = p.ORDER_RATE.values())[0], 0] for _ in range(self.workdays)]
        for n in self.orders:
            daily_volume = 0
            for _ in range(n[0]):
                daily_volume += self.rngesus.choices(list(p.ORDER_SIZE.keys()), weights = p.ORDER_SIZE.values())[0]
            n[0] = daily_volume

                
   
    def dispatch(self): 
        today = self.current_time[0] #today's index, used a lot in this function
        expected_out = sum(self.orders[today])
        reality_lottery = self.rngesus.choices(list(p.DISPATCH_RANDOMNESS.keys()), weights = p.DISPATCH_RANDOMNESS.values())[0]


        if reality_lottery == "early pickup":
            if today < self.workdays - 1: #can't do early pickups if there's no week left
                day = self.rngesus.randint(today + 1, self.workdays - 1)
                slot = self.rngesus.choice([0,1])
                if self.orders[day][slot] >= 2 and self.current_stock >=2: #only delivers if there actually is stock and an order to deliver
                    self.orders[day][slot] -= 2 
                    self.current_stock -= 2
        elif reality_lottery == "surprise pickup":
            if self.current_stock >= 2:
                self.current_stock -= 2


        if self.rngesus.choice([0,1]) and self.current_stock >= 2: #50% chance to see if there's an outstanding order that can be picked up late
            for day in self.orders[:today]:
                for i in range(2):
                    if day[i] > 0:
                        volume_out = min(2, day[i])
                        day[i] -= volume_out
                        self.current_stock -= volume_out
                        break
                else:
                    continue
                break


        # Determine how much of today's volume actually leaves in this round
        if self.current_time[1] == 0:  # Morning - some orders
            actual_out = round(self.rngesus.triangular(0, expected_out))
        else:  # Evening - whatever remains
            actual_out = expected_out

        if reality_lottery == "no show": #one load remains undelivered
            actual_out = max(0, actual_out -2)

        # Deduct dispatched volume from today's outstanding orders
        remaining = actual_out
        for i in range(2):
            dispatched = min(self.orders[today][i], remaining)
            reward += dispatched * p.PRE_ORDER_REWARD if i == 0 else dispatched * p.RESPONSE_ORDER_REWARD
            self.orders[today][i] -= dispatched
            remaining -= dispatched
            if remaining <= 0:
                break
        actual_out -= remaining

        if actual_out > self.current_stock:
            unfulfilled = actual_out - self.current_stock
            self.current_stock = 0

        else:
            self.current_stock -= actual_out
            unfulfilled = 0

        reward += unfulfilled * p.UNFULFILLED_PENALTY

        return reward

    def get_imminent_orders(self) -> int: #gets today's and tomorrow's orders in full loads
        today = self.current_time[0]
        if today == self.workdays-1: #friday!
            orders = self.orders[today]
        else:    
            orders = self.orders[today] + self.orders[today+1]
        orders = ceil(orders/2) #half load orders do count
        return orders


    #region ACTIONS

    def plan_week(self, strategy: tuple) -> tuple: #initial, weekly planned response orders
        outlook, timing = strategy
        weekend_prod_estimate = 3 * self.weekend * 1.75 #a realistic model considering the probability tables
        monday_stock_estimate = min(
            self.tank_capacity,
            self.current_stock + weekend_prod_estimate
        )
        # Estimate production over the working week
        weekly_prod_estimate = self.workdays * 6

        if outlook == 1: #optimistic
            weekly_prod_estimate += 4
        elif outlook == 2: #pessimistic
            weekly_prod_estimate -= 4

        # Estimate total stock that must be accommodated
        customer_orders = sum(day[0] for day in self.orders)
        estimated_surplus = max(
            0,
            round(monday_stock_estimate + weekly_prod_estimate - customer_orders)
        )

        weights = self.get_planning_weights(timing)

        # Allocate whole half-loads proportionally
        total_weight = sum(weights)
        allocations = [
            int(estimated_surplus * weight / total_weight)
            for weight in weights
        ]

        # Distribute leftover half-loads caused by rounding
        remainder = estimated_surplus - sum(allocations)
        ranking = sorted(
            range(self.workdays),
            key=lambda i: (
                estimated_surplus * weights[i] / total_weight
                - allocations[i]
            ),
            reverse=True
        )
        for i in ranking[:remainder]:
            allocations[i] += 1

        for day_index in range(self.workdays):
            self.orders[day_index][1] = allocations[day_index]

        tanks = self.calculate_stock_level()
        orders = self.get_imminent_orders()
        time = self.calculate_time_left()

        return tanks, orders, time

    def get_planning_weights(self, strategy):
        env_weights = [
            1 / (1 + day_orders[0])
            for day_orders in self.orders
        ]
        env_total = sum(env_weights)
        env_weights = [w / env_total for w in env_weights]

        # Agent's preferred timing distribution
        middle = (self.workdays - 1) / 2
        agent_weights = []

        for day_index in range(self.workdays):
            if strategy == 1:       # early
                weight = self.workdays - day_index
            elif strategy == 2:     # late
                weight = day_index + 1
            elif strategy == 3:     # convex: peak in the middle
                weight = 1 + middle - abs(day_index - middle)
            elif strategy == 4:     # concave: peaks at both ends
                weight = 1 + abs(day_index - middle)
            else:                 # balanced (default)
                weight = 1

            agent_weights.append(weight)

        agent_total = sum(agent_weights)
        agent_weights = [w / agent_total for w in agent_weights]

        # Combine agent strategy and environmental preference
        weights = [
            p.AGENT_INFLUENCE * agent_weights[i] + (1-p.AGENT_INFLUENCE) * env_weights[i]
            for i in range(self.workdays)
        ]
        return weights


    def interpret_action(self, action):
        reward = 0
        if action == 0:
            pass #do nothing
        elif action == 1:
            self.plan_ahead()
            reward += p.LONG_NOTICE_CHANGE_COST
        elif action == 2:
            self.cancel_order_later()
            reward += p.LONG_NOTICE_CHANGE_COST
        elif action == 3:
            self.emergency_dispatch()
            reward += p.EMERGENCY_CHANGE_COST
        elif action == 4:
            self.cancel_order_now()
            reward += p.EMERGENCY_CHANGE_COST
        return reward

    def emergency_dispatch(self): #reliable but costly. also used for weekend.
        self.current_stock -= 2

    def plan_ahead(self,): #add overflow orders to a future day, which one? Should it have a penalty late in the week?
        pass #TODO

    def cancel_order_later(self): #cancels an order that's not today or tomorrow. prioritises overflow orders.
        pass #TODO

    def cancel_order_now(self): #cancels one of today's or if none left, tomorrow's orders. prioritises overflow orderss.
        pass #TODO

    #endregion


    #region PRODUCTION

    def produce(self): #single production cycle of 8 hours, sufficient for daytime
        production = self.rngesus.choices(list(p.PRODUCTION_RATE.keys()), weights=p.PRODUCTION_RATE.values())[0]
        if self.current_stock + production <= self.tank_capacity:
            self.current_stock += production
        else: 
            raise TanksFullException("Tanks are full, production stopped.")

    def overnight_produce(self):
        for _ in range(2):
            self.produce()


    def weekend_produce(self): #consider also adding 1 or 2, since friday evening and night are technically also included.
        for _ in range(3 * self.weekend):
            self.produce()

    def calculate_stock_level(self, foresight = True): #discretion for the agent with foresight
        if foresight:
            todays_orders = sum(self.orders[self.current_time[0]])
            situation = self.current_stock - todays_orders
        else:
            situation = self.current_stock
        if situation < 0:
            return 0 #empty
        elif situation <= 3:
            return 1 #low
        elif situation >= 14:
            return 4 #full
        elif situation >= 11:
            return 3 #high
        else:
            return 2 #normal
 
    
    #endregion


    #region TIME
    def increment_time(self):
        time = list(self.current_time)
        if time == [self.workdays - 1, 1]: #end of the workweek
            time[0] = 0
            time[1] = -1
        elif time[1] == 1: #regular evening
            time[1] = 0
            time[0] += 1
        else: #not an evening
            time[1] +=1
        self.current_time = tuple(time)

    def get_next_time(self, time):
        time = list(time)

        if time == [self.workdays - 1, 1]:
            time[0] = 0
            time[1] = -1
        elif time[1] == 1:
            time[1] = 0
            time[0] += 1
        else:
            time[1] += 1

        return tuple(time)

    def calculate_time_left(self): #calculates the amount of remaining decisions
        return self.workdays * 2 - (self.current_time[0] * 2 + self.current_time[1])

    #endregion




    def step(self, action): #episode starts with last workday evening, ends after last workday morning
        reward = 0
        try:
            if self.current_time == (0,-1): #Start of episode - we already have a long term plan
                reward += self.interpret_action(action)
                self.weekend_produce()
                self.increment_time()
                time_left = self.calculate_time_left()
                #first morning decision      

            elif self.current_time[1] == 0: #every morning including monday
                reward += self.interpret_action(action)
                reward += self.dispatch()
                self.produce()
                self.increment_time()
                time_left = self.calculate_time_left()
                #afternoon decision, whether it be end of ep or not

            elif self.current_time == (self.workdays - 1, 1): #gives final reward, ends the episode
                for _ in range(action): #emergency pre-weekend emptying
                    self.emergency_dispatch()
                    reward += p.EMERGENCY_CHANGE_COST
                reward += p.REMAINING_WEEKEND_STOCK_PENALTY * self.current_stock
                self.increment_time()
                time_left = 0 #calculate_time_left would return a negative value here

            else: #regular evening
                reward += self.interpret_action(action)
                if self.current_time[0]!= self.workdays-1: #overnight production, irrelevant on weekends
                    reward += self.dispatch()
                    self.overnight_produce()
                    self.increment_time()
                    time_left = self.calculate_time_left()
                    #morning decision

        except TanksFullException:
            self.increment_time()
            time_left = self.calculate_time_left()
            reward = p.FULL_PENALTY #ignore everything else if tanks go full, we REALLY don't want that

        stock_level = self.calculate_stock_level()
        orders = self.get_imminent_orders()

        return stock_level, orders, time_left, reward


    #endregion