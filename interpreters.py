from agent import EpisodeOutcome
import numpy as np
import pandas as pd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

#region DESCRIPTORS
# Shared state descriptions
stock_descriptions = [
    "Empty",
    "Low",
    "Normal",
    "High",
    "Full"
]

# start_q
order_distribution_descriptions = [
    "Early",
    "Convex",
    "Balanced",
    "Concave",
    "Late"
]

outlook_descriptions = [
    "Normal",
    "Optimistic",
    "Pessimistic"
]

timing_descriptions = [
    "Balanced",
    "Early",
    "Late",
    "Convex",
    "Concave"
]

# running_q
imminent_order_descriptions = [
    "0",
    "1-4",
    "5-8",
    "9-12",
    "13-16",
    "17+"
]

time_left_descriptions = [
    "Start",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

running_action_descriptions = [
    "Do nothing",
    "Plan ahead",
    "Cancel later order",
    "Emergency dispatch",
    "Cancel imminent order"
]

# final_q
final_action_descriptions = [
    "0 emergency dispatches",
    "1 emergency dispatch",
    "2 emergency dispatches",
    "3 emergency dispatches",
    "4 emergency dispatches"
]

TABLE_DESCRIPTORS = {
    (5, 5, 15): {
        "name": "Weekly planning Q-table",
        "dimension_names": ["Stock level", "Order dist."],
        "state_dimensions": [
            stock_descriptions,
            order_distribution_descriptions
        ],
        "action_decoder": "weekly",
        "legend": True
    },

    (5, 6, 10, 5): {
        "name": "Running Q-table",
        "dimension_names": [
            "Stock level",
            "Imminent orders",
            "Time remaining"
        ],
        "state_dimensions": [
            stock_descriptions,
            imminent_order_descriptions,
            time_left_descriptions
        ],
        "action_names": running_action_descriptions
    },

    (5, 5): {
        "name": "Weekend Q-table",
        "dimension_names": ["Stock level"],
        "state_dimensions": [
            stock_descriptions
        ],
        "action_names": final_action_descriptions
    }
}


def describe_state(state, descriptions):
    return tuple(
        descriptions[i][value]
        for i, value in enumerate(state)
    )
#endregion

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

def plot_comparison(results, iter):
    """Plot mean reward and standard deviation for each agent."""

    names = list(results.keys())
    means = [results[name]["mean_reward"] for name in names]
    deviations = [results[name]["std_reward"] for name in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, means, yerr=deviations, capsize=5)
    plt.ylabel("Mean weekly reward")
    plt.title("Agent performance comparison")
    plt.tight_layout()
    plt.savefig(f"results/agent_comparison{iter}.png")


def outcome_interpreter(outcome: EpisodeOutcome):

    print("\n" + "=" * 75)
    print("EPISODE REPORT")
    print("=" * 75)

    # WEEK PLANNING
    print("\nWEEK PLANNING")
    print("-" * 75)
    start_state = describe_state(
        outcome.start_state,
        [stock_descriptions, order_distribution_descriptions]
    )

    plan_state = describe_state(
        outcome.plan_week_state,
        [stock_descriptions, imminent_order_descriptions, time_left_descriptions]
    )

    print(f"Starting state:       {start_state}")
    outlook, timing = outcome.plan_week_action

    print(
        f"Planning decision:    "
        f"({outlook_descriptions[outlook]}, "
        f"{timing_descriptions[timing]})"
    )
    print(f"State after planning: {plan_state}")
    print(f"Accumulated reward:   {outcome.startq_reward}")

    # RUNNING PHASE
    print("\nRUNNING PHASE")
    print("-" * 75)
    print(f"{'Step':<6}{'State':<45}{'Action':<25}")
    print("-" * 75)

    for i, (state, action) in enumerate(
        zip(outcome.running_states, outcome.running_actions), 1
    ):
        state_description = describe_state(
            state,
            [
                stock_descriptions,
                imminent_order_descriptions,
                time_left_descriptions
            ]
        )

        action_description = running_action_descriptions[action]

        print(
            f"{i:<6}"
            f"{str(state_description):<45}"
            f"{action_description:<25}"
        )

    print("-" * 75)
    print(f"Running reward:       {outcome.running_reward}")

    # WEEKEND DECISION
    print("\nWEEKEND DECISION")
    print("-" * 75)

    end_state = stock_descriptions[outcome.endofweek_state]

    final_state = describe_state(
        outcome.final_state,
        [stock_descriptions, imminent_order_descriptions, time_left_descriptions]
    )

    print(f"End-of-week state:    {end_state}")
    print(f"Emergency dispatches: {outcome.endofweek_action}")
    print(f"Final state:          {final_state}")
    print(f"Weekend production:   {outcome.weekend_prod_reward}")

    # SUMMARY
    print("\nSUMMARY")
    print("-" * 75)
    print(f"Total accumulated reward: {outcome.startq_reward}")
    print(f"Exploration decisions:    {outcome.exploration_count}")
    print(f"Orders cancelled: {outcome.cancellations}")
    print(f"Orders failed to deliver: {outcome.failed_dispatches}")
    print(f"Total emergency dispatches: {outcome.emergency_dispatches}")
    print(f"Tanks overfilled {outcome.tank_full_events} times")

    print("=" * 75 + "\n")



def q_table_interpreter(q_table, agent=None):

    q = np.asarray(q_table)

    # Identify the table
    descriptor = TABLE_DESCRIPTORS.get(q.shape)

    state_dimensions = descriptor["state_dimensions"]
    dimension_names = descriptor["dimension_names"]


    # Generate state combinations
    states = list(product(
        *[range(size) for size in q.shape[:-1]]
    ))

    # Generate state labels
    state_labels = [
        tuple(
            descriptions[index]
            for descriptions, index in zip(state_dimensions, state)
        )
        for state in states
    ]

    # Generate action labels
    if "action_decoder" in descriptor:
        action_names = []

        for i in range(q.shape[-1]):
            outlook, timing = agent.decode_week_planning_action(i)

            action_names.append(f"({outlook}, {timing})")

    else:
        action_names = descriptor["action_names"]

    # Generate DataFrame
    values = q.reshape(-1, q.shape[-1])

    table = pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(
            state_labels,
            names=dimension_names
        ),
        columns=action_names
    )

    # Print table heading
    print(f"\n-------{descriptor['name'].upper()}-------")
    print(f"Q-table shape: {q.shape}\n")

    # Print legend if applicable
    if descriptor.get("legend", False):

        print("Action tuple legend:")

        print("  Outlook:")
        for i, description in enumerate(outlook_descriptions):
            print(f"    {i}: {description}")

        print("  Timing:")
        for i, description in enumerate(timing_descriptions):
            print(f"    {i}: {description}")

        print()

    print(table.to_string(float_format="%.3f"))

def epsilon_plotter(epsilon_history, interval=1000):
    episodes = [i * interval for i in range(len(epsilon_history))]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilon_history)

    plt.title("Epsilon Decay During Training")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/epsilonplot.png")