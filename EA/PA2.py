from ioh import get_problem
from ioh import logger
import sys
import numpy as np
import random
from tqdm import tqdm

# Parameters
parent_size = 12
offspring_size = 8  # 8 times larger
# sigma = ...
# global global_lr
# global local_lr

default_budget = 50000
default_lower_bound = -5
default_upper_bound = 5


def Recombination(population, parents_fitness, type='weighted_global'):
    """
        Uses intermediary recombination (global or not global)
        TODO Maybe implement other methods as well for experimenting ?
    """
    offspring = []

    if type == 'global':
        # Only return 1 average offspring
        # ## weighted_global has no weights in first round, so it uses global
        parents = [parent for parent, sigma in population]
        sigmas = [sigma for parent, sigma in population]
        offspring.append((np.mean(parents, axis=0), np.mean(sigmas, axis=0)))

    if type == 'random':  # random sample 2 offsprings and return averages
        for _ in range(parent_size):
            sample_pop = random.sample(population, 2)
            parents = [parent for parent, sigma in sample_pop]
            sigmas = [sigma for parent, sigma in sample_pop]
            _offspring = (np.mean(parents, axis=0), np.mean(sigmas, axis=0))
            offspring.append(_offspring)

    if type == 'weighted_global': #Pretty much like global but with weights
        #minim = np.amin(parents_fitness) # find smallest
        #parents_fit = [p + abs(minim) for p in parents_fitness] #scale to start from zero by adding abs(smallest) to all
        #maxim = np.amax(parents_fit) # find biggest
        #weights = [ 1 - (p / maxim) for p in parents_fit]  # 1 - divide each by biggest, smallest will be 0 so weight = 1, highest will be 1 so weight = 0
        fitness_diff = [p - opt for p in parents_fitness]
        p_sum = np.sum(fitness_diff)
        weights = [p_sum / p for p in fitness_diff]
        parents = [parent for parent, sigma in population]
        sigmas = [sigma for parent, sigma in population]
        offspring.append(list((np.average(parents, axis=0, weights=weights), np.average(sigmas, axis=0, weights=weights))))
    return offspring

def Mutate(pop):
    """
        Check if pop size = 1, means global recombination
        Using individual step sizes
        TODO maybe include correlated mutations?
    """
    chromosome_length = len(pop[0])
    global_lr = 1 / np.sqrt(2 * chromosome_length)
    local_lr = round(1 / np.sqrt(2 * np.sqrt(chromosome_length)),3)

    offspring = []
    while len(offspring) < len(pop) * offspring_size:
        if len(pop) == 1:
            sigmas = pop[0][1]
            m_sigmas = np.asarray(
                [np.sign(s) * np.abs(s) ** (np.random.normal(0, global_lr) + np.random.normal(0, local_lr)) for s in sigmas])
            m_pop = np.asarray([p + np.random.normal(0, abs(s)) for p, s in zip(pop[0][0], m_sigmas)])
            if any(m_pop < default_lower_bound) or any(m_pop > default_upper_bound):  # useless values
                continue
            offspring.append((m_pop, m_sigmas))

    return offspring


def Select(offspring, offspring_fitness, parents, parents_fitness, plus=False):
    """
        plus or comma selection
        plus selection adds original parents to potential selection
    """

    population = offspring
    fitness = offspring_fitness

    if plus:
        population = offspring + parents
        fitness = offspring_fitness + parents_fitness

    pop_with_fit = []
    for p, p_f in zip(population, fitness):
        pop_with_fit.append((p, p_f))
    pop_with_fit.sort(key=lambda x: x[1], reverse=False)  # reverse=False -> keep lowest (minimisation)
    return [pop for pop, _ in pop_with_fit[:parent_size]], [fit for _, fit in pop_with_fit[:parent_size]]


def ES(
        func,
        budget=default_budget,
        lower_bound=default_lower_bound,
        upper_bound=default_upper_bound
):

    global opt
    opt = func.objective.y
    chromosome_length = func.meta_data.n_variables
    parents = np.random.uniform(lower_bound, upper_bound, size=(parent_size, chromosome_length))
    # sigmas = ...?
    # parents = [(pop, np.random.uniform(0,1, size=chromosome_length)) for pop in parents] # apend sigmas
    parents_fitness = [func(x) for x in parents]  # Evaluate parent
    parents = [(pop, [np.random.normal(0, 1/np.sqrt(2 * chromosome_length))] * chromosome_length) for i, pop in enumerate(parents)]
    budget = budget - parent_size

    # parent = Select(parent, parent_fitness)
    while budget > 0:
        # Generate offspring and evaluate
        # Note that this is just an example, and there may be difference among different evlution strategies.
        # You shall implement your own evolutin strategy.

        offspring = Recombination(parents, parents_fitness)
        offspring = Mutate(offspring)
        offspring_fitness = []
        for i in range(len(offspring)):
            offspring_fitness.append(func(offspring[i][0]))

        budget = budget - offspring_size
        parents, parents_fitness = Select(offspring, offspring_fitness, parents, parents_fitness)



# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress this folder and upload it to IOHanalyzer
l = logger.Analyzer(root="test_data",
                    folder_name="weighted_global-comma_select-fitness_diff",
                    algorithm_name="s1861581_s1697714",
                    algorithm_info="An Evolution Strategy on the 24 BBOB problems in python")

# Testing on 24 problems
for pid in tqdm(range(1, 25)):  # 1,25

    # Testing 10 instances for each problem
    for ins in tqdm(range(1, 11)):  # 1,11

        # Getting the problem with corresponding problem_id, instance_id, and dimension n = 5.
        problem = get_problem(pid, dim=5, iid=ins, problem_type='BBOB')

        # Attach the problem with the logger
        problem.attach_logger(l)

        # The assignment 2 requires only one run for each instance of the problem.
        ES(problem)
        # To reset evaluation information as default before the next independent run.
        # DO NOT forget this function before starting a new run. Otherwise, you may meet issues with uploading data to IOHanalyzer.
        problem.reset()
# This statemenet is necessary in case data is not flushed yet.
del l
