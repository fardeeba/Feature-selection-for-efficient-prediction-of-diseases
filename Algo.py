import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def calculate_fitness(features):

    X = dataset.values[:, features]
    Y = dataset.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))

    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    #crossover_point = np.round(np.uint8(offspring_size[1]/2));
    print("Size of offspring",offspring_size)
    crossover_point1 = random.randint(0,offspring_size[1]-44)
    print("First cross over point",crossover_point1)
    #crossover_point2 = random.randint(crossover_point1,offspring_size[1])
    crossover_point2 = crossover_point1 + 44
    print("Second cross over point", crossover_point2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
        offspring[k, (crossover_point1):(crossover_point2+1)] = parents[parent2_idx, (crossover_point1):(crossover_point2+1)]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, (crossover_point2+1):(offspring_size[1]+1)] = parents[parent1_idx, (crossover_point2+1):(offspring_size[1]+1)]

    return offspring

def mutation(offspring_crossover):
    #mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    mutation_counter = random.sample(range(offspring_crossover.shape[1]),4)
    for idx in range(offspring_crossover.shape[0]):
        for mutation_num in range(len(mutation_counter)):
            gene_idx = mutation_counter[mutation_num]
            # The random value to be added to the gene.
            if offspring_crossover[idx, gene_idx] == 1:
                offspring_crossover[idx, gene_idx] = 0
            elif offspring_crossover[idx, gene_idx] == 0:
                offspring_crossover[idx, gene_idx] = 1
    return offspring_crossover

num_weights = 132
dataset = pd.read_csv('Training.csv', sep=',', header=0)
feature_set = []
"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""

sol_per_pop = 100
# Creating the initial population.
#new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_si
num_parents_mating = 50


# Defining the population size.
pop_size = (sol_per_pop, num_weights)  # The population will have sol_per_pop chromosome where each chromosome has
# num_weights genes.ze)
new_population = np.random.choice([0, 1], size= pop_size)
#print(new_population)

diverse_pop = np.unique(new_population, axis= 0)

best_outputs = []
num_generations = 50
mutation_rate = 0.1
mutation_generation = mutation_rate*num_generations

for generation in range(num_generations):
    print("Generation : ", generation)
    fitNess = []
    rows = len(diverse_pop)
    cols = len(diverse_pop[0])
    # Measuring the fitness of each chromosome in the population.
    for chromosome in range(0,rows):
        feature_set = []
        for gene in range(0,cols):
            if diverse_pop[chromosome][gene] == 1:
                feature_set.append(gene)
      #  print("length of feature set: ",len(feature_set))
       # print("Elements of feature set: ", feature_set)
        fitness = calculate_fitness(feature_set)
       # print("Fitness value",fitness)
        fitNess.append(fitness)
        if(fitness == np.max(fitNess)):
            fit_chromo = chromosome

    #print("Fitness")
    #print(fitness)

    best_outputs.append(np.max(fitNess))

    best_chromosome = diverse_pop[fit_chromo]
    # The best result in the current iteration.
    print("Best result : ", np.max(fitNess))
    print("Best chromosome : ",best_chromosome)

    # Selecting the best parents in the population for mating.
    #print("Fitness list",fitNess)
    parents = select_mating_pool(diverse_pop, fitNess, num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
    print("Crossover")

    offspring_mutation = offspring_crossover

    if(generation == mutation_generation):
        offspring_mutation = mutation(offspring_crossover)
        mutation_generation = generation+(mutation_rate*num_generations)
        print("Mutation")
        print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    diverse_pop = np.unique(new_population, axis= 0)

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.

"""
fitNess = cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
"""
print("Best output list: ",best_outputs)
import matplotlib.pyplot as plt1

plt1.plot(best_outputs)
plt1.xlabel("Iteration")
plt1.ylabel("Fitness")
plt1.show()

