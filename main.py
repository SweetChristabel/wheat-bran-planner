from agent import QLearningAgent
import time
from interpreters import outcome_interpreter, q_table_interpreter, epsilon_plotter




def main():
    agent = QLearningAgent()

    TRAIN_LENGTH = 100000
    SHOW_Q_TABLES = True

    start_time = time.perf_counter()
    log = agent.train(TRAIN_LENGTH)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print("###FIRST###")
    outcome_interpreter(log[0])
    print("###HALFWAY###")
    outcome_interpreter(log[round(len(log)/2)])
    print("###LAST###")
    outcome_interpreter(log[-1])
    episode = agent.greedy_path(1)
    print("###GREEDY###")
    outcome_interpreter(episode[0])
    if SHOW_Q_TABLES:
        q_table_interpreter(agent.start_q, agent)
        q_table_interpreter(agent.running_q)
        q_table_interpreter(agent.final_q)

    print(f"Training took {elapsed:.2f} seconds")
    print(f"Average per episode: {elapsed / TRAIN_LENGTH:.6f} seconds")

    epsilon_plotter(agent.epsilon_plot)



if __name__ == "__main__":
    main()