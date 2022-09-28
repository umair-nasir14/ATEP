from .activations import modified_sigmoid,tanh

'''All common hyperparameters'''

class Hyperparameters:

    def __init__(self,args):

        self.args = args
        self.delta_threshold = self.args.delta_threshold
        self.population = self.args.neat_population
        
        self.delta_coefficients = {
            'c1' : self.args.c1,
            'c2' : self.args.c2,
            'c3' : self.args.c3
        }
        
        self.activation = tanh

        self.max_fitness = float('inf')
        self.max_generations = float('inf')
        self.max_fitness_history = self.args.max_stagnation

        self.crossover_probability = {
            'do' : self.args.crossover_probability,
            'skip' : 1 - self.args.crossover_probability
        }
        self.connection_weight_probability={

            'do': self.args.connection_weight_probability,
            'skip': 1 - self.args.connection_weight_probability
        }
        

        self.mutation_probability = {
            'node' : self.args.mutation_probability_node,
            'connection' : 1 - self.args.mutation_probability_node,
            
        }

        self.weight_mutation_probability = {

            'weight_perturbation' : self.args.weight_mutate_large_probability,
            'weight_random' : 1 - self.args.weight_mutate_large_probability,
            
        }

        self.bias_mutation_probability = {

            'bias_perturbation' : self.args.bias_mutate_large_probability,
            'bias_random' : 1 - self.args.bias_mutate_large_probability
        

        }




