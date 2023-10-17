import math
import random
import matplotlib.pyplot as plt



class Chromosome:

    def __init__(self, value, genes, selection_prob, length, fitness):
        self.value = value  # decimal value
        self.genes = genes  # encoded chromosome -> binary value
        self.selection_prob = selection_prob
        self.length = length
        self.fitness = fitness
        self.cumulative_prob = None


class Population:
    print_flag = True # class variable used for printing data on the first step

    def __init__(self, n, domain, parameters, precision, crossover_prob, mutation_prob, steps):
        self.n = n
        self.domain = domain
        self.parameters = parameters
        self.precision = precision
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.steps = steps
        self.generation = [] # list of chromosomes

    def generate_initial_population(self):
        for index in range(self.n):
            random_number = random.uniform(self.domain[0], self.domain[1])

            chromosome = Chromosome(value=round(random_number, self.precision), genes=None, length=None, selection_prob=None, fitness=None)
            chromosome.genes = self.decode(chromosome)
            chromosome.length = self.calculate_length(chromosome)
            chromosome.fitness = self.calculate_fitness(chromosome)

            self.generation.append(chromosome)

        self.set_selection_probabilities()
        self.calculate_cumulative_probabilities()

    def calculate_fitness(self, chromosome):
        chromosome.fitness = self.parameters[0] * chromosome.value * chromosome.value + \
                             self.parameters[1] * chromosome.value + \
                             self.parameters[2]

        return chromosome.fitness

    def calculate_length(self, chromosome):
        chromosome.length = math.ceil(math.log2((self.domain[1] - self.domain[0]) * pow(10, self.precision)))
        return chromosome.length

    def elite_chromosome(self):
        elite = max(self.generation, key=lambda x: x.fitness)

        return elite

    #determines maximum fitness value among the chromosomes
    def calculate_max_fitness(self):
        return self.elite_chromosome().fitness

    #calculates average fitness value of the chromosomes in the population
    def calculate_mean_fitness(self):
        mean_performance = sum(chromosome.fitness for chromosome in self.generation)

        return mean_performance / self.n

    # converts the binary representation of a chromosome into a decimal value
    def encode(self, chromosome):
        decimal_value = int(chromosome.genes, 2)
        d = (self.domain[1] - self.domain[0]) / pow(2, chromosome.length) # discretization increment
        left_value = self.domain[0]  # lower bound of the dicretization interval
        interval_index = 0

        while interval_index != decimal_value:
            left_value += d
            interval_index += 1

        # updating chromosome's properties
        chromosome.value = round(left_value, self.precision)
        chromosome.fitness = self.calculate_fitness(chromosome)
        chromosome.selection_prob = self.calculate_selection_probability(chromosome)
        self.calculate_cumulative_probabilities()

        return chromosome

    # converts the decimal value of a chromosome into a binary representation
    def decode(self, chromosome):
        chromosome.length = self.calculate_length(chromosome)
        d = (self.domain[1] - self.domain[0]) / pow(2, chromosome.length) # discretization increment
        left_value = self.domain[0]  # lower bound of the discretization interval
        interval_index = 0

        while chromosome.value >= left_value:
            left_value += d
            interval_index += 1

        genes = format(interval_index, f'0{chromosome.length}b')
        chromosome.genes = ''.join(str(b) for b in genes)

        return chromosome.genes

    # computes the selection probability for a chromosome
    def calculate_selection_probability(self, chromosome):
        chromosome.selection_prob = chromosome.fitness / self.calculate_total_performance()
        return chromosome.selection_prob

    # determines the total performance of the population
    def calculate_total_performance(self):
        total_performance = 0

        for chromosome in self.generation:
            total_performance += chromosome.fitness

        return total_performance

    # sets the selection probability for each chromosome
    def set_selection_probabilities(self):
        for index_chromosome in range(self.n):
            chromosome = self.generation[index_chromosome]
            chromosome.selection_prob = self.calculate_selection_probability(chromosome)
            self.generation[index_chromosome] = chromosome

    #computes the cumulative probability for each chromosome
    def calculate_cumulative_probabilities(self):
        cumulative_prob = 0

        for index_chromosome in range(self.n):
            chromosome = self.generation[index_chromosome]
            chromosome.cumulative_prob = cumulative_prob + chromosome.selection_prob
            cumulative_prob = chromosome.cumulative_prob

    #returns the index of the chromosome in the population
    #relative to the selection intervales defined by the cumulative probabilities
    def binary_search(self, value):
        left, right = 0, self.n - 1

        while left < right:
            middle = (left + right) // 2

            if value < self.generation[middle].cumulative_prob:
                right = middle
            else:
                left = middle + 1

        return left

    # performs the selection operation
    # creates a new generation of chromosomes based on the cumulative probabilities
    def selection(self, filePath):
        outputFile = openFile(filePath)

        new_generation = []

        for index_chromosome in range(self.n):
            random_u = random.random()
            chromosome_selected = self.binary_search(random_u)

            if Population.print_flag:
                outputFile.write("u = " + str(random_u) + " selected chromosome " + str(chromosome_selected) + "\n")

            new_generation.append(self.generation[chromosome_selected - 1])

        # elitist selection
        random_index = random.randint(0, self.n - 1)
        new_generation[random_index] = self.elite_chromosome()

        self.generation = new_generation
        outputFile.close()

    #crossover operation between two parent chromosomes
    def crossover(self, chromosome1, chromosome2, crossover_point):
        child1 = chromosome1
        child2 = chromosome2

        child1 = chromosome1.genes[:crossover_point] + chromosome2.genes[crossover_point:]
        child2 = chromosome2.genes[:crossover_point] + chromosome1.genes[crossover_point:]

        return child1, child2

    #performs crossover on the current generation of chromosomes
    def crossed_population(self, filePath):
        outputFile = openFile(filePath)

        crossover_participants = []

        for index_chromosome, chromosome in enumerate(self.generation):
            crossover_probability = random.uniform(0, 1)

            if Population.print_flag:
                if index_chromosome < 9:
                    output_string = " "
                else:
                    output_string = ""

                output_string += str(index_chromosome + 1) + ": " + str(chromosome.genes) + "   u = " + str(crossover_probability)

                if crossover_probability < self.crossover_prob:
                    output_string += " < " + str(self.crossover_prob) + " participant" + "\n"
                    crossover_participants.append(index_chromosome)
                else:
                    output_string += "\n"

                outputFile.write(output_string)

        outputFile.write("\n")

        while len(crossover_participants) >= 2:
            index1, index2 = random.sample(crossover_participants, 2)
            crossover_point = random.randint(0, self.generation[0].length - 1)
            child1, child2 = self.crossover(self.generation[index1], self.generation[index2], crossover_point)

            if Population.print_flag:
                output_string = "Crossing chromosomes: " + str(index1 + 1) + " and " + str(index2 + 1) + "\n" + \
                                "Chromosome " + str(index1 + 1) + "   " + self.generation[index1].genes + "\n" + \
                                "Chromosome " + str(index2 + 1) + "   " + self.generation[index2].genes + "\n" + \
                                "Crossover point: " + str(crossover_point) + "\n" + \
                                "Crossover result: " + str(child1) + "  " + str(child2) + "\n\n"

                outputFile.write(output_string)

            self.generation[index1].genes = child1
            self.encode(self.generation[index1])

            self.generation[index2].genes = child2
            self.encode(self.generation[index2])

            crossover_participants.remove(index1)
            crossover_participants.remove(index2)

        outputFile.close()

    #mutates the chromosome by flipping the bit at the specified position
    def mutation(self, chromosome, index):
        mutated_chromosome = [int(genome) for genome in chromosome.genes]
        mutated_chromosome[index] = 1 - mutated_chromosome[index]
        mutated_chromosome = ''.join(str(genome) for genome in mutated_chromosome)

        return mutated_chromosome

    #performs mutation on the current generation of chromosomes
    def mutated_population(self, filePath):
        outputFile = openFile(filePath)
        output_string = ""

        for index_chromosome, chromosome in enumerate(self.generation):
            mutation_probability = random.uniform(0, 1)

            if mutation_probability < self.mutation_prob:
                output_string += str(index_chromosome + 1) + "\n"

                random_index = random.randint(0, self.generation[0].length - 1)
                chromosome.genes = self.mutation(chromosome, random_index)
                chromosome = self.encode(chromosome) # update on the current chromosome after mutation

        self.calculate_cumulative_probabilities()

        if Population.print_flag:
            outputFile.write(output_string)

        outputFile.close()

    def print_chromosomes(self, filePath):
        outputFile = openFile(filePath)

        for index_chromosome, chromosome in enumerate(self.generation):
            if index_chromosome < 9:
                output_string = " "
            else:
                output_string = ""

            outputFile.write(output_string + str(index_chromosome + 1) + ": " + str(chromosome.genes) + \
                         "   x = " + str(chromosome.value) + "    f = " + str(chromosome.fitness) + "\n")

        outputFile.close()

    def print_population(self, filePath):
        outputFile = open(filePath, "w")
        outputFile.write("Initial Population" + "\n \n")

        for index_chromosome, chromosome in enumerate(self.generation):
            if index_chromosome < 9:
                output_string = " "
            else:
                output_string = ""
            outputFile.write(output_string + str(index_chromosome + 1) + ": " + str(chromosome.genes) + \
                         "   x = " + str(chromosome.value) + "    f = " + str(chromosome.fitness) + "\n")

        outputFile.write("\nSelection probabilities:" + "\n \n")

        for index_chromosome, chromosome in enumerate(self.generation):
            outputFile.write("Chromosome " + str(index_chromosome + 1) + " with selection probability " + str(
                chromosome.selection_prob) + "\n")

        outputFile.write("\nSelection intervals: " + "\n \n")
        outputFile.write("Interval 0 = " + "( 0, " + str(self.generation[0].cumulative_prob) + " )" + "\n")

        for index_chromosome in range(self.n - 1):
            next_chromosome = self.generation[index_chromosome + 1]
            output_string = "Interval " + str(index_chromosome + 1) + " = ( " + str(
                self.generation[index_chromosome].cumulative_prob) + ", " + str(next_chromosome.cumulative_prob) + " )"
            outputFile.write(output_string + "\n")

        # selection
        outputFile.write("\nSelection process:" + "\n \n")
        outputFile.close()
        self.selection(filePath)

        outputFile = openFile(filePath)
        outputFile.write("\nAfter selection:" + "\n \n")
        outputFile.close()
        self.print_chromosomes(filePath)

        # crossing
        outputFile = openFile(filePath)
        outputFile.write("\nCrossover probability: " + str(self.crossover_prob) + "\n \n")
        outputFile.close()

        self.crossed_population(filePath)

        outputFile = openFile(filePath)
        outputFile.write("\nAfter crossing:" + "\n \n")
        outputFile.close()

        self.print_chromosomes(filePath)

        # mutation
        outputFile = openFile(filePath)
        outputFile.write("\nMutation probability for each genome: " + str(self.mutation_prob) + "\n \n")
        outputFile.write("The following chromosomes have been modified: " + "\n")
        outputFile.close()

        self.mutated_population(filePath)

        outputFile = openFile(filePath)
        outputFile.write("\nAfter mutation: " + "\n \n")
        outputFile.close()
        self.print_chromosomes(filePath)

        # evolution of the maximum and average fitness value
        outputFile = openFile(filePath)
        outputFile.write("\nMax and average performance evolution:" + "\n \n")
        outputFile.write("Max fitness = " + str(self.calculate_max_fitness()) + "     " + \
                         "Mean fitness = " + str(self.calculate_mean_fitness()))

        outputFile.close()



def openFile(filePath):
    return open(filePath, "a")

def main():
    # reading the input data
    with open("date.in") as inputFile:
        data = inputFile.read().splitlines()
        n = int(data[0])  # size of the initial population
        domain = eval(data[1])
        parameters = eval(data[2])
        precision = int(data[3])
        crossover_prob = float(data[4])
        mutation_prob = float(data[5])
        steps = int(data[6])

    filePath = "date.out"
    max_fitness = []
    mean_fitness = []

    #generationg the initial population
    generation = Population(n, domain, parameters, precision, crossover_prob, mutation_prob, steps)
    generation.generate_initial_population()
    generation.print_population(filePath)

    max_fitness.append(generation.calculate_max_fitness())
    mean_fitness.append(generation.calculate_mean_fitness())

    Population.print_flag = False

    for step in range(1, generation.steps):
        generation.selection(filePath) # selection
        generation.crossed_population(filePath) # crossing
        generation.mutated_population(filePath) # mutation

        max_fitness_value = generation.calculate_max_fitness()
        mean_fitness_value = generation.calculate_mean_fitness()

        max_fitness.append(max_fitness_value)
        mean_fitness.append(mean_fitness_value)

        outputFile = openFile(filePath)
        outputFile.write("Max fitness = " + str(max_fitness_value) + "     " + \
                         "Mean fitness = " + str(mean_fitness_value))

        outputFile.close()

    # graph

    generation_numbers = range(1, len(max_fitness) + 1)
    plt.plot(generation_numbers, max_fitness, label='Maximum Fitness Value', color='black')
    plt.plot(generation_numbers, mean_fitness, label='Mean Fitness Value', color='red')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of maximum and average performance')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
