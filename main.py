from agent import QLearningAgent, RuleBasedAgent, RandomAgent
from wheatmill import WheatMill
import time
from interpreters import outcome_interpreter, q_table_interpreter, epsilon_plotter, plot_comparison
from evaluator import evaluate, compare
import os
import platform
import subprocess

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
    os.makedirs("results", exist_ok=True)
    print("How many training episodes should we run?")
    print("For quick but usable results, one million.")
    print("10 million and you can probably go eat dinner.")
    TRAIN_LENGTH = int(input("Insert number: "))
    SHOW_Q_TABLES = bool(input("Before we start - type 1 if you want to see the q-tables, 0 if you do not. :"))
    q_agent = QLearningAgent()
    train(q_agent, TRAIN_LENGTH, SHOW_Q_TABLES)
    epsilon_plotter(q_agent.epsilon_plot)
    notify()
    print("Ding dong, training is done")
    input("Press Enter to proceed with evaluations.")

    evaluations = 5
    seeds = [1337, 42, 58808, 6767, 9001]
    random_agent = RandomAgent()
    rule_agent = RuleBasedAgent()
    # Generate evaluation outcomes
    for i in range(evaluations):
        env = WheatMill(seeds[i])
        q_outcomes = q_agent.greedy_path(env, episodes=100)
        random_outcomes = random_agent.run(env, episodes=100)
        rule_outcomes = rule_agent.run(env, episodes=100)


        results = {
        "Q-learning": evaluate(q_outcomes),
        "Random": evaluate(random_outcomes),
        "Rule-based": evaluate(rule_outcomes),
        }
        compare(results, i)
        plot_comparison(results)






if __name__ == "__main__":
    main()