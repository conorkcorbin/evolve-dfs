import genalg
import pandas as pd
import numpy as np 

# Draft Kings Salary Cap
salarycap = 50000

# Define a fitness function
def fitness(individual, data):
 
    currentSolutionCost = 0
    currentSolutionPoints = 0
    possibleTeam = False
    for value in individual.values():
        currentSolutionCost += value[2]
        currentSolutionPoints += value[1]
    if (currentSolutionCost <= salarycap):
        return currentSolutionPoints
    return 0

def getUniqueLineups(lineups,predicted_points):
    unique_lists = []
    unique_points = []
    for l,points in zip(lineups,predicted_points): 
        l_unique = True
        for u in unique_lists:
            if set(l) == set(u):
                l_unique = False

        if (l_unique):
            unique_lists.append(np.array(l))
            unique_points.append(points)
    return unique_lists, unique_points

""" FOR FOOTBALL COMMENT TO HIDE """

# Load Data
offense = pd.read_csv('./NFL/OffenseWeek1.csv')
defense = pd.read_csv('./NFL/DefenseWeek1.csv')
df = pd.concat([offense,defense])

# Run the genetic algorithm
ga = genalg.GeneticAlgorithm(df,population_size=500,generations=100,mutation_probability=0.2,categories='Football') 
ga.fitness_function = fitness              
ga.run()                                 

# Get List of lineups from last generation
chosen_lineups = [[g[1][key][0] for key in g[1]] for g in ga.last_generation() if g[0] > 0 ]
predicted_points = [g[0] for g in ga.last_generation() if g[0] > 0]

# Filter For Unique Lineups
unique_lineups, unique_points = getUniqueLineups(chosen_lineups,predicted_points)
unique_lineups =  pd.DataFrame(unique_lineups).transpose()

# Print best lineup and  save last generation to CSV
print(ga.best_individual())
unique_lineups.to_csv('./lineupsNFL.csv',index=False)

""" FOR BASEBALL UNCOMMENT TO USE """ 

# # Load Data
# df = pd.read_csv('./MLB/9-11-2017proj.csv')

# # Run the genetic algorithm
# ga = genalg.GeneticAlgorithm(df,population_size=500,generations=100,mutation_probability=0.2,categories='Baseball') 
# ga.fitness_function = fitness              
# ga.run()   

# # Get List of lineups from last generation
# chosen_lineups = [[g[1][key][0] for key in g[1]] for g in ga.last_generation() if g[0] > 0 ]
# predicted_points = [g[0] for g in ga.last_generation() if g[0] > 0]

# # Filter For Unique Lineups
# unique_lineups, unique_points = getUniqueLineups(chosen_lineups,predicted_points)
# unique_lineups =  pd.DataFrame(unique_lineups).transpose()

# # Print best lineup and  save last generation to CSV
# print(ga.best_individual())
# unique_lineups.to_csv('./lineupsMLB.csv',index=False)



