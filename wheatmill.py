import random
import parameters as p

class TanksFullException(Exception): 
    pass

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
        return self.orders

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

        return actual_out, unfulfilled

    def calculate_order_urgency(self, orders): #considers orders of next three iterations imminent #TODO Consider need
        imminent_orders = 0
        later_orders = 0
        time = self.current_time
        imminent_days = set()

        for _ in range(3):
            time = self.get_next_time(time)

            if time <= (0, 0):
                break
            imminent_days.add(time[0])

        for day_index, day_orders in enumerate(orders):
            if day_index in imminent_days:
                imminent_orders += sum(day_orders)
            elif day_index >= self.current_time[0]:
                later_orders += sum(day_orders)

        return imminent_orders, later_orders

    #region ACTIONS

    def plan_week(self, strategy: tuple): #initial, weekly planned response orders
        outlook, timing = strategy
        weekend_prod_estimate = 3 * self.weekend * 1.75 #a realistic model considering the probability tables
        monday_stock_estimate = min(
            self.tank_capacity,
            self.current_stock + weekend_prod_estimate
        )
        # Estimate production over the working week
        weekly_prod_estimate = self.workdays * 6

        if outlook == "optimistic":
            weekly_prod_estimate += 4
        elif outlook == "pessimistic":
            weekly_prod_estimate -= 4

        # Estimate total stock that must be accommodated
        customer_orders = sum(day[0] for day in self.orders)
        estimated_surplus = max(
            0,
            round(monday_stock_estimate + weekly_prod_estimate - customer_orders)
        )

        # Weigh days: fewer customer pre-orders means greater preference but strategy also counts
        weights = []
        for day_index, day_orders in enumerate(self.orders):
            weight = 1 / (1 + day_orders[0])

            if timing == "early":
                weight *= 1 + 0.1 * (self.workdays - 1 - day_index)
            elif timing == "late":
                weight *= 1 + 0.1 * day_index

            weights.append(weight)

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

    def emergency_dispatch(self): #reliable but costly. also used for weekend.
        self.current_stock -= 2

    def plan_additional_overflow(self, day): #add overflow orders to a future day, which one?
        pass

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

    def calculate_time_left(self): #calculates the amount of remaining decisions #TODO consider need
        return self.workdays * 2 - (self.current_time[0] * 2 + self.current_time[1])

    #endregion





    def step(self, action): #episode starts with last workday evening, ends after last workday morning
        reward = 0
        try:
            if self.current_time == (0,-1): #Start of episode - we already have a long term plan
                self.weekend_produce()
                self.increment_time()
                time_left = self.calculate_time_left()
                #first morning decision      

            elif self.current_time[1] == 0: #every morning including monday
                out, fail = self.dispatch()
                self.produce()
                self.increment_time()
                time_left = self.calculate_time_left()
                if self.current_time[0] == self.workdays - 1: #leads to last/weekend decision of ending episode
                    #weekend decision
                    pass
                else: #leads to a regular afternoon decision
                    #evening decision
                    pass

            elif self.current_time == (self.workdays - 1, 1): #gives final reward and orders for the new week, ends the episode
                #weekend pickups
                #tank assessment and weekly reward
                self.increment_time()
                time_left = 0 #calculate_time_left would return a negative value here

            else: #regular evening
                if self.current_time[0]!= self.workdays-1: #overnight production, irrelevant on weekends
                    out,fail = self.dispatch()
                    self.overnight_produce()
                    self.increment_time()
                    time_left = self.calculate_time_left()
                    #morning decision
        except TanksFullException:
            self.increment_time()
            time_left = self.calculate_time_left
            #TODO finish   

        stock_level = self.calculate_stock_level()

        return time_left, reward, self.orders, stock_level


    #endregion