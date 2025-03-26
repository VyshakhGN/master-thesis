import random
import string
TARGET = "to be or not to be"
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
MAX_GENERATIONS = 250

CHARACTERS = string.ascii_lowercase + " "

def random_char():
    return random.choice(CHARACTERS)

def create_individual():
    return [random_char() for i in range(18)]

def create_population(size):
    return [create_individual() for i in range(size)]

def fitness(individual):
    count = 0
    for i in range(len(individual)):
        if individual[i] == TARGET[i]:
            count += 1
    return count

def selection(population, fitnesses):
    tournament_size = 3
    def tournament():
        competitors = random.sample(list(zip(population, fitnesses)), tournament_size)
        return max(competitors, key=lambda item: item[1])[0]
    parent1 = tournament()
    parent2 = tournament()
    return parent1, parent2

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutation(individual, rate):
    for i in range(len(individual)):
        if random.random() < rate:
            individual[i] = random_char()
    return individual

#Main EA
def genetic_algorithm():
    population = create_population(POPULATION_SIZE)
    generation = 0
    best_individual = None
    best_fitness = -1
    while generation < MAX_GENERATIONS:
        fitnesses = [fitness(i) for i in population]
        if max(fitnesses) == 18:
            best_index = fitnesses.index(max(fitnesses))
            best_individual = population[best_index]
            print(f"Target achieved in generation {generation}: {''.join(best_individual)}")
            break

        current_best = max(fitnesses)
        if current_best > best_fitness:
            best_fitness = current_best
            best_index = fitnesses.index(current_best)
            best_individual = population[best_index]

        print(f"Generation {generation}, Best Fitness: {best_fitness}, Best Individual: {''.join(best_individual)}")

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, MUTATION_RATE)
            child2 = mutation(child2, MUTATION_RATE)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]
        generation += 1
    best_fitness = fitness(best_individual)
    return "".join(best_individual), best_fitness

if __name__ == "__main__":
    best, fit = genetic_algorithm()
    print(f"Final Phrase: {best}, with fitness {fit}")
