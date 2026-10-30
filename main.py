from agent import QLearningAgent, RuleBasedAgent, RandomAgent
from wheatmill import WheatMill
import time
from interpreters import outcome_interpreter, q_table_interpreter, epsilon_plotter, plot_comparison, compare
import os
import platform
import subprocess
import numpy as np
import parameters as p
import random

def notify(): #ONLY TESTED ON MAC
    system = platform.system()

    if system == "Windows":
        import winsound
        winsound.MessageBeep(winsound.MB_OK)

    elif system == "Darwin":  # macOS
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            check=False
        )

    elif system == "Linux":
        print("\a")  # Terminal bell (if supported)

    else:
        print("Notification sound unavailable on this OS.")

def evaluate(outcomes):
    episodes = len(outcomes)

    return {
        "mean_reward": np.mean([
            o.startq_reward + o.weekend_prod_reward
            for o in outcomes
        ]),

        "std_reward": np.std([
            o.startq_reward + o.weekend_prod_reward for o in outcomes
        ]),

        "emergency_dispatches_per_week":
            sum(o.emergency_dispatches for o in outcomes) / episodes,

        "cancellations_per_week":
            sum(o.cancellations for o in outcomes) / episodes,

        "tank_full_events_per_100_weeks":
            sum(o.tank_full_events for o in outcomes) / episodes * 100,

        "failed_dispatches_per_week":
            sum(o.failed_dispatches for o in outcomes) / episodes
        
    }

def train(agent, length, show_q = False):
    start_time = time.perf_counter()
    log = agent.train(length)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print("###FIRST###")
    outcome_interpreter(log[0])
    print("###HALFWAY###")
    outcome_interpreter(log[round(len(log)/2)])
    print("###LAST###")
    outcome_interpreter(log[-1])
    if show_q:
        q_table_interpreter(agent.start_q, agent)
        q_table_interpreter(agent.running_q)
        q_table_interpreter(agent.final_q)

    print(f"Training took {elapsed:.2f} seconds")
    print(f"Average per episode: {elapsed / length:.6f} seconds")




def main():
    rng = random.Random(p.ENVIRONMENT_SEED)
    os.makedirs("results", exist_ok=True)
    print(f"Starting training with {p.TRAINING_EPISODES} episodes. Please wait.")
    q_agent = QLearningAgent()
    train(q_agent, p.TRAINING_EPISODES, p.SHOW_Q_TABLES)
    epsilon_plotter(q_agent.epsilon_plot)
    notify()
    print("Ding dong, training is done")
    input("Press Enter to proceed with evaluations.")

    seeds = rng.sample(range(2**32), k=p.EVALUATION_RUNS)
    random_agent = RandomAgent()
    rule_agent = RuleBasedAgent()
    # Generate evaluation outcomes
    for i in range(p.EVALUATION_RUNS):
        q_env = WheatMill(seeds[i])
        ran_env = WheatMill(seeds[i])
        rule_env = WheatMill(seeds[i])

        q_outcomes = q_agent.greedy_path(q_env, p.EVALUATION_EPISODES)
        random_outcomes = random_agent.run(ran_env, p.EVALUATION_EPISODES)
        rule_outcomes = rule_agent.run(rule_env, p.EVALUATION_EPISODES)


        results = {
        "Q-learning": evaluate(q_outcomes),
        "Random": evaluate(random_outcomes),
        "Rule-based": evaluate(rule_outcomes),
        }
        compare(results, i)
        plot_comparison(results, i)






if __name__ == "__main__":
    main()