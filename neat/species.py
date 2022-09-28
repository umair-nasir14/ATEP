# Copyright (c) 2020.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import random
import copy
import math

from .config import Hyperparameters



class Species:

    '''
    
    Represents a specie object.
    
    '''
    def __init__(self, max_fitness_history, *members) -> None:
        
        self.members = list(members) # Members are individuals in specie
        self.fitness_sum = 0
        self.fitness_history = []
        self.max_fitness_history = max_fitness_history

    def crossover(self, mutation_probability, crossover_probability):

        '''Implements crossover with probabilities described in the Paper'''

        type = list(crossover_probability.keys())
        probs = [crossover_probability[i] for i in type]
        choosen = random.choices(type, weights = probs)[0]

        if choosen == 'skip' or len(self.members) == 1:
            '''If it chooses skip or there is only one member, we mutate.'''

            hyperparams = Hyperparameters()
            child = random.choice(self.members).clone()
            child.mutate(mutation_probability,hyperparams.connection_weight_probability)

        elif choosen == 'do':
    
            (parent_1, parent_2) = random.sample(self.members, 2) 
            child = self.do_crossover(parent_1, parent_2)
        return child
            

    def do_crossover(self,parent_1,parent_2):

        '''Implements Reproduction.'''

        from neat_distributed.genome import Genome

        child = Genome(parent_1.inputs, parent_1.outputs, parent_1.activation)
            
        parent_1_in = set(parent_1.connections)
        parent_2_in = set(parent_2.connections)


        
        # Get similar genes from random parent.
        for g in parent_1_in & parent_2_in:

            parent = random.choice([parent_1, parent_2])
            child.connections[g] = copy.deepcopy(parent.connections[g])

        # Get excess and disjoint genes from fitter parent

        if parent_1.fitness > parent_2.fitness:
            for i in parent_1_in - parent_2_in:
                child.connections[i] = copy.deepcopy(parent_1.connections[i])
        else:
            for i in parent_2_in - parent_1_in:
                child.connections[i] = copy.deepcopy(parent_2.connections[i])

            # Number of nodes in child

        child.max_node_count = 0
        for (i, j) in child.connections:
            current_max = max(i, j)
            child.max_node_count = max(child.max_node_count, current_max)
        child.max_node_count += 1

        # Get nodes

        for i in range(child.max_node_count):
                
            inherit_nodes_from = list()
            if i in parent_1.nodes:
                inherit_nodes_from.append(parent_1)
            if i in parent_2.nodes:
                inherit_nodes_from.append(parent_2)
                
            random.shuffle(inherit_nodes_from)
            parent = max(inherit_nodes_from, key=lambda parent: parent.fitness)
            child.nodes[i] = copy.deepcopy(parent.nodes[i])

        child.reset()
        
        return child
    def kill_genomes(self, fittest=True):

        '''Kill the weakest genomes.'''

        self.members.sort(key=lambda genome:genome.fitness, reverse=True)

        if fittest:
            remaining=4
        else:
            remaining = int(math.ceil(1.0*len(self.members)))
        
        self.members = self.members[:remaining]

    def update_fitness(self):
        '''finds adjusted fitness and updates fitness history.'''
        for g in self.members:
            g.adjusted_fitness = g.fitness/len(self.members)
        
        self.fitness_sum = sum([g.adjusted_fitness for g in self.members])
        self.fitness_history.append(self.fitness_sum)
        if len(self.fitness_history) > self.max_fitness_history:
            self.fitness_history.pop(0)

        
    def can_progress(self):
        '''Finds if the specie should progress'''
        n = len(self.fitness_history)
        avg = sum(self.fitness_history)/n
        return (avg > self.fitness_history[0] and len(self.members) > 175) or n < self.max_fitness_history
    
    def add_genome(self, genome):
        '''Adds Genome in the specie.'''
        self.members.append(genome)
    
    def _get_best_genome(self):
        '''Gets the fittest Genome in the specie.'''#Should do for novelty search as well 
        return max(self.members, key=lambda genome: genome.fitness)
    
