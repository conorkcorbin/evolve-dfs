# -*- coding: utf-8 -*-
"""
    pyeasyga module

"""
import random
import numpy as np
import pandas as pd
import copy
from operator import attrgetter

from six.moves import range


class GeneticAlgorithm(object):
    """Genetic Algorithm class.

    This is the main class that controls the functionality of the Genetic
    Algorithm.

    """

    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 elitism=True,
                 maximise_fitness=True,
                 oneToZeroRatio=None,
                 categories='Baseball'):
        """Instantiate the Genetic Algorithm.

        :param seed_data: input data to the Genetic Algorithm 
        :type seed_data: pandas dataframe
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of mutation operation
        :param list categories: 'Baseball' or 'Football'. Tells us what our genes (positions)
        are composed of. 
        """

        self.seed_data = seed_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness
        self.oneToZeroRatio = oneToZeroRatio
        if categories == 'Baseball': 
            self.categories = ['OF1', 'OF2', 'OF3','SP1', 'SP2', '1B','2B','3B','SS','C']
        elif categories == 'Football':
            self.categories = ['WR1', 'WR2', 'WR3', 'QB','RB1','RB2','TE','FLEX','D']

        else: 
            self.categories = None
        self.current_generation = []

        def create_individual(seed_data):
            """Create a candidate solution representation.

            :param seed_data: input data to the Genetic Algorithm
            :type seed_data: pandas dataframe
            :returns: candidate solution representation as a dictionary mapping
                position to player. Genes are the players, lineups are the individuals.

            """

            previously_chosen = []
            individuals = {gene: randomlySelect(gene,seed_data,previously_chosen) for gene in self.categories}
            return individuals

        def randomlySelect(key,seed_data,previously_chosen):
            """ This function randomly selects players from each
            category to construct an initial linup and does a check to make sure
            that we aren't picking the same player twice in the case where a player
            can play more than one position """

            # Remove the index of categories that have multiple slots
            already_chosen = True
            while already_chosen:   
                key = ''.join([i for i in key if not i.isdigit()])
                if key == 'FLEX':
                    seed_data = seed_data[(seed_data['Pos'] == 'WR') | (seed_data['Pos'] == 'RB') | (seed_data['Pos'] == 'TE')].reset_index(drop=True)
                else:
                    seed_data = seed_data[seed_data['Pos'].str.contains(key)].reset_index(drop=True)
                index = random.randrange(len(seed_data))
                value = (seed_data.get_value(index,'Player Name'),seed_data.get_value(index,'Proj FP'),seed_data.get_value(index,'Salary'))
                if value[0] not in previously_chosen:
                    already_chosen = False 
                    previously_chosen.append(value[0])
            # print(previously_chosen)
            return value

        def crossover(parent_1, parent_2):
            """Crossover (mate) two parents to produce two children.

            :param parent_1: candidate solution representation (list)
            :param parent_2: candidate solution representation (list)
            :returns: tuple containing two children

            """
            def exist_duplicates_values(values):
                return len(set(values)) != len(values)

            exist_duplicates = True
            while exist_duplicates:
                index = random.randrange(1, len(parent_1))
                keys = list(parent_1.keys())
                random.shuffle(keys)
                child_1 = { key:(parent_1[key] if i <= index else parent_2[key]) for key,i in zip(keys,range(len(self.categories))) }
                exist_duplicates = exist_duplicates_values(child_1.values())

            exist_duplicates = True
            while exist_duplicates:
                index = random.randrange(1, len(parent_2))
                keys = list(parent_2.keys())
                random.shuffle(keys)
                child_2 = {key:(parent_2[key] if i <= index else parent_1[key]) for key,i in zip(keys,range(len(self.categories)))}
                exist_duplicates = exist_duplicates_values(child_2.values())
    
            return child_1, child_2

        def mutate(individual,seed_data):
            """Reverse the bit of a random index in an individual."""
            mutate_index = random.randrange(len(individual))
            keys = list(individual.keys())
            k = keys[mutate_index]
            previously_chosen = [individual[key][0] for key in keys if key != k]
            individual[k] = randomlySelect(k,seed_data,previously_chosen)
            # print([individual[key][0] for key in individual])
            # individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = random.sample(population, self.tournament_size)
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.fitness_function = None
        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size // 10
        self.random_selection = random_selection
        self.create_individual = create_individual
        self.crossover_function = crossover
        self.mutate_function = mutate
        self.selection_function = self.tournament_selection

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual(self.seed_data)
            individual = Chromosome(genes)
            initial_population.append(individual)
        self.current_generation = initial_population

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in self.current_generation:
            individual.fitness = self.fitness_function(
                individual.genes, self.seed_data)
        # print(individual.fitness)
    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = random.random() < self.crossover_probability
            can_mutate = random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes,self.seed_data)
                self.mutate_function(child_2.genes,self.seed_data)

            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    def create_first_generation(self):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness()
        self.rank_population()
    
    

    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population()
        self.calculate_population_fitness()
        self.rank_population()

    def run(self):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation()

        for _ in range(1, self.generations):
            self.create_next_generation()

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return (best.fitness, best.genes)

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        return ((member.fitness, member.genes) for member
                in self.current_generation)


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes):
        """Initialise the Chromosome."""
        self.genes = genes
        self.fitness = 0

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))
