import math
import numpy as np
from matplotlib import pyplot as plt


def hone_graph(hone_base_rate,
               artisan_energy_base,
               attempt_cost,
               hone_boost: (int, float, int) = (0, 0, 0),
               silent=True):
    """
    Output information about honing success rates and costs distribution.

    :param hone_base_rate: The base chance of succeeding a hone attempt
    :param artisan_energy_base: The artisan energy gain per hone
    :param attempt_cost: The gold cost of honing
    :param hone_boost: [hone_attempt_number, chance_boost, cost]
    :param silent: prints outputs when False
    """
    max_attempts = math.ceil(100 / artisan_energy_base)
    success_dist = [min(hone_base_rate + hone_boost[1] if hone_boost[0] == 1 else hone_base_rate, 1)]
    chance_of_attempt = 1
    for i in range(1, max_attempts):
        chance_of_success = hone_base_rate + hone_base_rate * .1 * i
        if hone_boost[0] == i + 1:
            chance_of_success += hone_boost[1]
        chance_of_success = min(chance_of_success, 1)
        chance_of_attempt = chance_of_attempt - success_dist[i - 1]
        success_dist.append(chance_of_success * chance_of_attempt)

    success_dist[-1] = 1 - sum(success_dist[0:-1])
    cost_graph = np.array([[chance, attempt_cost * (i + 1)] for i, chance in enumerate(success_dist)])
    cost_graph[hone_boost[0] - 1][1] += hone_boost[2]
    average_cost = sum(x[0] * x[1] for x in cost_graph)

    if not silent:
        np.set_printoptions(suppress=True,
                            formatter={'float_kind': '{:f}'.format})

        print("success distribution with cost:\n", cost_graph)
        print("average cost = ", average_cost)
        plt.plot(cost_graph[:, 1], cost_graph[:, 0], marker='o', label='hone rate distribution')
        plt.plot(cost_graph[:, 1], np.cumsum(cost_graph[:, 0]), marker='o', label='cumulative distribution')

        plt.xlabel("gold cost")
        plt.ylabel("rate of success")
        plt.legend()
        plt.show()
    return average_cost


if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True)

    # Hone base chance, artisan energy gain, gold cost per attempt
    hone_stats = (0.3, 13.95, 1506)

    # Solar grace, blessing, protection, and book
    # (boost chance in percentage points, gold cost
    hone_boosters = ((.84, 66),
                     (1.67, 200),
                     (5, 310),
                     (10, 800))

    # Currently, the change in artisan energy gain as a result of using hone boosters is not considered

    for i in range(len(hone_boosters)):
        average_costs = [hone_graph(*hone_stats)]

        for j in range(5):
            hone_boost = (j + 1, hone_boosters[i][0] / 100, hone_boosters[i][1])
            average_costs.append(hone_graph(*hone_stats, hone_boost))

        print("costs when boosting for each hone attempt:", average_costs)
        default_cost = average_costs[0]
        print("% change on each hone attempt:",
              np.array([(x - default_cost) / default_cost for x in average_costs[1:]]),
              "\n")
