from ioh import get_problem
from ioh import logger
import sys
import numpy as np
import random

from operations import crossover
from operations import mutate
from operations import mating_selection

# Declaration of problems to be tested.
om = get_problem(1, dim=50, iid=1, problem_type = 'PBO')
lo = get_problem(2, dim=50, iid=1, problem_type = 'PBO')
labs = get_problem(18, dim=32, iid=1, problem_type = 'PBO')


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



def genetic_algorithm(
    func,
    population_size=4,
    crossover_rate=0.95,
    mutation_rate=None,
    n_points=2,
    seed=None,
    budget=None,
):
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.objective.y
    print(optimum)

    chromosome_length = func.meta_data.n_variables
    if mutation_rate is None:
        mutation_rate = 0.01

    if n_points == 0:
        n_points = func.meta_data.n_variables

    # 10 independent runs for each algorithm on each problem.
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None

        if seed is not None:
            np.random.seed(seed + r)
            random.seed(seed + r)

        pop = np.random.randint(
            2, size=(population_size, chromosome_length)
        )  # Generate population
        pop_fit = [func(x) for x in pop]  # Calculate fitness for population

        run_budget = budget
        while run_budget >= 1:
            parents = mating_selection(pop, pop_fit, func)
            pop = crossover(parents, crossover_rate=crossover_rate, n_points=n_points)
            pop = mutate(pop, mutation_rate=mutation_rate)

            pop_fit = [func(x) for x in pop]  # Calculate each fittnes
            f = pop_fit[np.argmax(pop_fit)]  # Get best fitness
            x = pop[np.argmax(pop_fit)]  # Get best fitness genome

            if f > f_opt:
                f_opt = f
                x_opt = x
                # print(f'updated... Newly found optimum: {f_opt}')
            if f_opt >= optimum:
                break
            run_budget -= population_size
            func.reset()
        print(f'run done with popsize {population_size} and crossover {crossover_rate}, budget left {run_budget} out of {budget}')
    return f_opt, x_opt

  
pop_size=[8,12,16,20,24]
cross_over = [0.60,0.70,0.80,0.90,1]

for c in cross_over:
	for p in pop_size:
		l = logger.Analyzer(root="OneMax", 
		folder_name="run_mut0.01_pop8-24_cross0.6-1", 
		algorithm_name=f"Genetic_algorithm_popsize{str(p)}_crossover{str(c)}", 
		algorithm_info="test of IOHexperimenter in python")


		om.attach_logger(l)
		f_opt, x_opt = genetic_algorithm(om, population_size=p, crossover_rate=c)
		print(f"Discovered optimum after budget / reaching actual optimum: {f_opt} \n With the bitstring: {x_opt}")
