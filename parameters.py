
SEED = 1337

#region REWARD TABLE

PRE_ORDER_REWARD = 10 #small, larger than response order
RESPONSE_ORDER_REWARD = 3
UNFULFILLED_PENALTY = -15
FULL_PENALTY = -30 #should be nasty
REMAINING_WEEKEND_STOCK_PENALTY = -2 #for each half load remaining in the tank during the weekend
LONG_NOTICE_CHANGE_COST = -2 #cancellations and extra dispatches planned beyond imminency
EMERGENCY_CHANGE_COST = -5 #cancellations and emergency fetchings within 3 day imminency or weekend

#endregion

#region ENVIRONMENT PARAMETERS
PRODUCTION_RATE = { #half-loads per 8 hours, probability 0-1
    0 : 0.15,
    1 : 0.1,
    2 : 0.7,
    3 : 0.05
}

ORDER_RATE = { #incoming orders per workday, probability 0-1
    0 : 0.1,
    1 : 0.25,
    2 : 0.35,
    3 : 0.23,
    4 : 0.06,
    5 : 0.01,
}

ORDER_SIZE = { #Usually a full load (two half-loads), but not always
    1 : 0.05,
    2 : 0.95,
}

DISPATCH_RANDOMNESS = {
    "surprise pickup": 0.01,
    "no show": 0.05,
    "early pickup": 0.04,
    "nothing unusual": 0.9
}

AGENT_INFLUENCE = 0.9 #agent vs environment weight in weekly planning

TANK_CAPACITY = 15 #Good for a two day weekend; if WORKDAYS is decreased, this should be increased by about 6 per day to prevent unavoidable overfilling.
WORKDAYS = 5 #Theoretical maximum 7, but weekend-less behaviour is not accounted for.

#endregion

#region AGENT PARAMETERS

ALPHA = 0
EPSILON = 0
GAMMA = 0


#endregion