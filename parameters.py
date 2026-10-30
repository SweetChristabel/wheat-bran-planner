
SEED = 1337

#region REWARD TABLE

PRE_ORDER_REWARD = 10 #small, larger than response order
RESPONSE_ORDER_REWARD = 3
UNFULFILLED_PENALTY = -15
FULL_PENALTY = -30 #should be nasty
REMAINING_WEEKEND_STOCK_PENALTY = -2 #for each half load remaining in the tank during the weekend
LONG_NOTICE_CHANGE_COST = -2 #cancellations and extra dispatches planned beyond imminency
INVALID_ACTION_PENALTY = -10 #w
EMERGENCY_CHANGE_COST = -5 #cancellations and emergency fetchings within 3 day imminency or weekend

#endregion

#region ENVIRONMENT PARAMETERS
PRODUCTION_RATE = { #half-loads per 8 hours, probability 0-1
    0 : 0.15,
    1 : 0.1,
    2 : 0.3,
    3 : 0.4,
    4 : 0.05
}

ORDER_RATE = { #incoming orders per workday, probability 0-1
    0 : 0.1,
    1 : 0.25,
    2 : 0.35,
    3 : 0.24,
    4 : 0.01,
}

ORDER_SIZE = { #Usually a full load (two half-loads), but not always
    1 : 0.05,
    2 : 0.95,
}

AGENT_INFLUENCE = 0.9 #agent vs environment weight in weekly planning

#endregion

#region AGENT PARAMETERS

ALPHA = 0.2 #learning rate



EPSILON_START = 0.9
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999999422



GAMMA = 0.5 #importance of future rewards, only used in running_q


#endregion