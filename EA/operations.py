import numpy as np
import random


def mating_selection(pop, p_fit, func):
    p_sum = np.sum(p_fit)  # Total fitness
    p_expect = [p / p_sum for p in p_fit]  # Fraction of total fitness
    p_expect = [
        p / (1 / len(p_expect)) for p in p_expect
    ]  # Calculate expected count from roullete wheel

    pop_with_fit = []
    for p, p_f in zip(pop, p_fit):
        pop_with_fit.append((p, p_f))

    parents = random.choices(
        pop_with_fit, weights=p_expect, k=len(pop)
    )  # Pick parents at random with expected counts as weights

    parents.sort(key=lambda x: x[1], reverse=True)
    parents = [p[0] for p in parents]

    return parents


def crossover(parents, crossover_rate, n_points):
    chromosome_length = len(parents[0])
    parents = parents[: int((len(parents) / 2))]  # Get top 50% parents
    parents_even, parents_odd = parents[::2], parents[1::2]
    new_pop = []
    for parent1, parent2 in zip(parents_even, parents_odd):
        child1, child2 = (
            parent1,
            parent2,
        )  # In case of no crossover, put parents back in population
        if random.random() < crossover_rate:  # Apply crossover
            Z = random.sample(
                range(chromosome_length), n_points
            )  # Gemerate n random crossover points
            Z.sort()
            for z in Z:  # n_point crossover
                child1 = np.append(child1[:z], child2[z:])
                child2 = np.append(child2[:z], child1[z:])

        new_pop.append(child1)
        new_pop.append(child2)
    new_pop = np.vstack(
        (new_pop, parents)
    )  # Append original top genomes back to population
    return new_pop


def mutate(pop, mutation_rate):
    chromosome_length = len(pop[0])

    for genome in pop:
        for bit in range(chromosome_length):
            if random.random() < mutation_rate:
                genome[bit] = 1 - genome[bit]

    return pop
