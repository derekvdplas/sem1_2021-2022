from ioh import get_problem
from ioh import logger
import sys
import numpy as np
import random

from operations import crossover
from operations import mutate
from operations import mating_selection


def genetic_algorithm(
    func,
    population_size=8,
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
        mutation_rate = 1 / chromosome_length

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
    return f_opt, x_opt


# Declaration of problems to be tested.
om = get_problem(1, dim=50, iid=1, problem_type="PBO")
lo = get_problem(2, dim=50, iid=1, problem_type="PBO")
labs = get_problem(18, dim=32, iid=1, problem_type="PBO")


l = logger.Analyzer(
    root="data",
    folder_name="run",
    algorithm_name=f"Genetic_algorithm",
    algorithm_info="test of IOHexperimenter in python",
)

om.attach_logger(l)
f_opt, x_opt = genetic_algorithm(
    om,
    mutation_rate=None, # None is 1/n
    population_size=24,
    crossover_rate=0.95,
    n_points=1,
    seed=42)

lo.attach_logger(l)
f_opt, x_opt = genetic_algorithm(
    lo,
    mutation_rate=None, # None is 1/n
    population_size=20,
    crossover_rate=0.85,
    n_points=1,
    seed=42)

labs.attach_logger(l)
f_opt, x_opt = genetic_algorithm(
    labs,
    mutation_rate=None, # None is 1/n
    population_size=10, 
    crossover_rate=0.65,
    n_points=5,
    seed=42)
