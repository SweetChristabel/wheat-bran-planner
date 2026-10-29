from agent import QLearningAgent, EpisodeOutcome

def outcome_interpreter(outcome: EpisodeOutcome):
    for i in outcome:
        print(i)


def main():
    agent = QLearningAgent

    agent.train(10)
    episode = agent.greedy_path(1)

    outcome_interpreter(episode[0])
