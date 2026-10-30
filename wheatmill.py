import random
import parameters as p
from math import ceil

class TanksFullException(Exception): 
    pass


class WheatMill():

    """
    Initially, tank size and week length were configurable in parameters. That has now been removed.
    Currently, the only thing not supporting week length configuration is the week planning's order distribution
    and the agent's response to it, which are both hard coded to a 5 day week.

    """

    def __init__(self, seed= p.SEED): 
        self.current_stock = 0
        self.tank_capacity = 15
        self.rngesus = random.Random(seed)
        self.weekend = 2
        self.workdays = 5
        self.orders = [[0, 0] for _ in range(self.workdays)] #pre-orders and overflow orders for each workday
        self.current_time = (0,-1) #(weekday, morning/evening): -1 signals episode start
        self.emergency_dispatches = 0
        self.cancellations = 0
        self.tank_full_events = 0
        self.failed_dispatches = 0


    #region ORDER HANDLING

    def reset_episode_counters(self):
        self.emergency_dispatches = 0
        self.cancellations = 0
        self.tank_full_events = 0
        self.failed_dispatches = 0

    def generate_episode_start(self):
        self.generate_preorders()
        self.reset_episode_counters()
        stock = self.calculate_stock_level(foresight = False)
        distribution = self.classify_order_distribution()
        return stock, distribution

    def generate_preorders(self): #a workdays list with random amounts per day
        self.orders = [[self.rngesus.choices(list(p.ORDER_RATE.keys()), weights = p.ORDER_RATE.values())[0], 0] for _ in range(self.workdays)]
        for n in self.orders:
            daily_volume = 0
            for _ in range(n[0]):
                daily_volume += self.rngesus.choices(list(p.ORDER_SIZE.keys()), weights = p.ORDER_SIZE.values())[0]
            n[0] = daily_volume

     
    def classify_order_distribution(self):
        # Extract customer pre-orders, ignoring planned response orders
        daily_orders = [day[0] for day in self.orders]
        total_orders = sum(daily_orders)

        # No demand means there is no meaningful distribution pattern
        if total_orders == 0:
            return 2  # balanced

        # Normalise so the pattern is independent of total demand
        distribution = [volume / total_orders for volume in daily_orders]

        # Ideal patterns: early, convex, balanced, concave, late
        patterns = [
            [0.40, 0.30, 0.15, 0.10, 0.05],  # 0: early
            [0.10, 0.20, 0.40, 0.20, 0.10],  # 1: convex
            [0.20, 0.20, 0.20, 0.20, 0.20],  # 2: balanced
            [0.30, 0.15, 0.05, 0.15, 0.35],  # 3: concave
            [0.05, 0.10, 0.15, 0.30, 0.40],  # 4: late
        ]

        # Find the ideal pattern with the smallest squared distance
        distances = [
            sum((actual - ideal) ** 2 for actual, ideal in zip(distribution, pattern))
            for pattern in patterns
        ]

        return distances.index(min(distances))           
   
    def dispatch(self): 
        today = self.current_time[0] #today's index, used a lot in this function
        expected_out = sum(self.orders[today])
        reward = 0

        # Determine how much of today's volume actually leaves in this round
        if today == self.workdays - 1: #friday
            actual_out = expected_out
        elif self.current_time[1] == 0:  # Morning - some orders
            actual_out = round(self.rngesus.triangular(0, expected_out))
        else:  # Evening - whatever remains
            actual_out = expected_out

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
            self.failed_dispatches += 1

        else:
            self.current_stock -= actual_out
            unfulfilled = 0

        reward += unfulfilled * p.UNFULFILLED_PENALTY

        return reward

    def get_imminent_orders(self) -> int: #makes the order space more manageable for the agent
        today = self.current_time[0]
        if today == self.workdays-1: #friday!
            orders = sum(self.orders[today])
        else:    
            orders = sum(self.orders[today] + self.orders[today+1])
        orders = min(ceil(orders/4), 5) #half load orders do count
        return orders


    #region ACTIONS

    def plan_week(self, strategy: tuple) -> tuple: #initial, weekly planned response orders
        outlook, timing = strategy
        weekend_prod_estimate = 3 * self.weekend * 2 #a realistic model considering the probability tables
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

        tanks = self.calculate_stock_level(foresight=False)
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
            success = self.plan_ahead()
            if success:
                reward += p.LONG_NOTICE_CHANGE_COST
            else:
                reward += p.INVALID_ACTION_PENALTY
        elif action == 2:
            success = self.cancel_order_later()
            if success:
                reward += p.LONG_NOTICE_CHANGE_COST
            else:
                reward += p.INVALID_ACTION_PENALTY
        elif action == 3:
            success = self.emergency_dispatch()
            if success:
                reward += p.EMERGENCY_CHANGE_COST
            else:
                reward += p.INVALID_ACTION_PENALTY
        elif action == 4:
            success = self.cancel_order_now()
            if success:
                reward += p.EMERGENCY_CHANGE_COST
            else:
                reward += p.INVALID_ACTION_PENALTY
        return reward

    def emergency_dispatch(self): #reliable but costly. also used for weekend.
        if self.current_stock < 2:
            self.current_stock = 0
            return False
        else:
            self.emergency_dispatches += 1
            self.current_stock -= 2
            return True

    def plan_ahead(self,): #add an overflow order to day after tomorrow. Penalize if it's late in the week.
        current_day = self.current_time[0]

        # Eligible days are strictly later than tomorrow
        eligible_days = range(current_day + 2, self.workdays)

        # Choose the earliest eligible day
        target_day = next(iter(eligible_days), None)

        if target_day is None:
            return False  # No eligible day; caller applies failure penalty

        self.orders[target_day][1] += 2  # Add two half-loads (one full load)
        return True

    def cancel_order_later(self): #cancels an order that's not today or tomorrow. prioritises overflow orders.
        current_day = self.current_time[0]
        eligible_days = range(current_day + 2, self.workdays)

        # Prefer the earliest available overflow order
        for day in eligible_days:
            if self.orders[day][1] > 0:
                self.orders[day][1] -= min(2, self.orders[day][1])
                self.cancellations += 1
                return True

        # Otherwise, cancel the earliest available customer order
        for day in eligible_days:
            if self.orders[day][0] >= 2:
                self.orders[day][0] -= 2
                self.cancellations += 1
                return True

        return False

    def cancel_order_now(self): #cancels an imminent order
        current_day = self.current_time[0]
        tomorrow = current_day + 1

        # Priority: today's overflow, today's customers,
        # tomorrow's overflow, tomorrow's customers
        priorities = [
            (current_day, 1),
            (current_day, 0),
            (tomorrow, 1),
            (tomorrow, 0)
        ]

        for day, order_type in priorities:
            if day >= self.workdays:
                continue

            if self.orders[day][order_type] > 0:
                cancelled = min(2, self.orders[day][order_type])
                self.orders[day][order_type] -= cancelled
                self.cancellations += 1
                return True

        return False

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


    def weekend_produce(self):
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

    def calculate_time_left(self): #calculates the amount of remaining decisions
        return self.workdays * 2 - (self.current_time[0] * 2 + self.current_time[1]) - 1 

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
                    success = self.emergency_dispatch()
                    if success:
                        reward += p.EMERGENCY_CHANGE_COST
                    else:
                        reward += p.INVALID_ACTION_PENALTY
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
            self.tank_full_events += 1
            self.increment_time()
            time_left = self.calculate_time_left()
            reward = p.FULL_PENALTY #ignore everything else if tanks go full, we REALLY don't want that


        orders = self.get_imminent_orders()
        stock_level = self.calculate_stock_level(foresight = False)

        return stock_level, orders, time_left, reward


    #endregion