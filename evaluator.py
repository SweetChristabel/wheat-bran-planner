import numpy as np

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


def compare(results, iteration):
    """Print a comparison of the agents."""

    print(f"\nAgent comparison, iteration {iteration}")
    print("-" * 100)

    metrics = [
        ("mean_reward", "Mean reward"),
        ("std_reward", "Std deviation"),
        ("emergency_dispatches_per_week", "Emergency dispatches"),
        ("cancellations_per_week", "Cancellations"),
        ("tank_full_events_per_100_weeks", "Tank full events / 100 weeks"),
        ("failed_dispatches_per_week", "Failed dispatches"),
    ]

    print(f"{'Metric':<30}", end="")
    for name in results:
        print(f"{name:>20}", end="")
    print()

    print("-" * 100)

    for key, label in metrics:
        print(f"{label:<30}", end="")
        for agent_metrics in results.values():
            print(f"{agent_metrics[key]:>20.2f}", end="")
        print()

    print("-" * 100)