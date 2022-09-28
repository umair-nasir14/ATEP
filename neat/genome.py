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


import copy
import random
import itertools

from .config import Hyperparameters

from atep.niches.box2d.env import make_env, bipedhard_custom, Env_config




HISTORICAL_MARKER = 0 

DEFAULT_ENV = Env_config(
        name='default_env',
        ground_roughness=0,
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[])

class Node:
    '''Object representation of a node'''
    def __init__(self, activation, bias_node=False):
        self.output = 0
        self.bias = 0#random.uniform(-1,1)#0 # This is jsut a weight of node. Bias written to distinguish between weight of connection and node
        self.activation = activation

        if bias_node == True:
            self.bias = 1
            self.activation = activation
            #self.output = 0

class Connection:
    '''Object representation of a connection'''
    def __init__(self, weight, historical_marker):

        self.innovation_number  = historical_marker
        self.weight = weight
        self.enabled = True



class Genome:

    '''
    
    A Genome will take inputs, outputs and activation function. Represents a genome and applies mutation on it.
    
    '''

    def __init__(self, inputs,outputs,activation):
        
        # NN structure

        self.connections = {}
        self.nodes = {}

        self.bias_node = []

        

        # Nodes

        self.inputs = inputs
        self.outputs = outputs
        self.max_node_count = inputs + outputs
        self.unhidden_nodes = inputs + outputs

        self.activation = activation

        # Performance 

        self.fitness = 0
        self.adjusted_fitness = 0

        self.env_name = bipedhard_custom.env_name
        #self.make_env(seed=24582922, env_config=DEFAULT_ENV)
    

    def generate_init_nn(self):

        '''
        Generates initial Neural Network for minimal structure and no hidden nodes.
        '''
        # Nodes
        self.bias_node = Node(self.activation,bias_node=True)
        self.nodes[0] = self.bias_node 

        for i in range(1,self.max_node_count):
            self.nodes[i] = Node(self.activation)

        # Connections
        for i in range(self.inputs):
            for j in range(self.inputs,self.unhidden_nodes):
                self.add_connection(i, j, random.uniform(-1,1))

    def add_connection(self, i, j, weight, init_generation=False):

        '''

        Add connections and enables it.

        '''

        global HISTORICAL_MARKER #For innovation number

        if (i, j) in self.connections:
            self.connections[(i,j)].enabled = True
        else:
            HISTORICAL_MARKER +=1
            self.connections[(i,j)] = Connection(weight, HISTORICAL_MARKER)
            #print(HISTORICAL_MARKER)
            

    def add_node(self):

        '''
        
        Adds node. Disables the connection first and enables both 
        of the new connections around.
        
        '''

        enabled = [i for i in self.connections if self.connections[i].enabled] # Choose connection which is true
        if enabled == []:
            #genome = Genome(self.inputs, self.outputs, self.hyperparams.activation)
            #self.generate_init_nn()
            (i,j) = self.random_connections()
            self.add_connection(i,j, random.uniform(-1,1))
            #n = Node(self.activation)
            enabled = [i for i in self.connections if self.connections[i].enabled]
            #self.add_connection(n, n, random.uniform(-1,1))
        (i,j) = random.choice(enabled)
        connection  = self.connections[(i,j)]
        connection.enabled = False # False to insert node
        # Defining node
        node_to_add = self.max_node_count
        self.max_node_count += 1
        self.nodes[node_to_add] = Node(self.activation)
        # Adding the node
        self.add_connection(i, node_to_add, 1.0)
        self.add_connection(node_to_add, j, connection.weight)

    def mutate(self, connection_node_probabilities, connection_w_probabilities):

        '''
        
        Mutates with all aspects.

        '''
        
        '''

        Mutation for nodes and connections.
        
        '''

        if self._is_disabled():
            self.add_enabled

        to_be_mutated = list(connection_node_probabilities.keys())

        probs = [connection_node_probabilities[i] for i in to_be_mutated] 
        choosen = random.choices(to_be_mutated, weights = probs)[0]

        if choosen == 'connection':
            (i,j) = self.random_connections()
            self.add_connection(i,j, random.uniform(-1,1))
        elif choosen == 'node':
            self.add_node()
      
        '''
        Mutation for node and connection weights.
        '''
        connection_mutate = list(connection_w_probabilities.keys())
        probs_weight = [connection_w_probabilities[i] for i in connection_mutate]
        choosen_weight = random.choices(connection_mutate, weights = probs_weight)[0]

        if choosen_weight == 'do':
            
            hyperparams = Hyperparameters()
            con_m_prob = list(hyperparams.weight_mutation_probability.keys())
            probs_weight = [hyperparams.weight_mutation_probability[i] for i in con_m_prob]
            choosen_weight_type = random.choices(con_m_prob, weights = probs_weight)[0]

            con_b_prob = list(hyperparams.bias_mutation_probability.keys())
            probs_bias = [hyperparams.bias_mutation_probability[i] for i in con_b_prob]
            choosen_bias_type = random.choices(con_b_prob, weights = probs_bias)[0]

            self.weight_perturb(choosen_weight_type)
            self.bias_perturb(choosen_bias_type)



        # resetting genome's internal state

        
        self.reset()
    

    def weight_perturb(self, choosen):

        '''
        
        Mutates weights of connections.

        '''
        
        conn = random.choice(list(self.connections.keys()))
        if choosen == 'weight_perturbation':
            self.connections[conn].weight += random.uniform(-1,1)
        elif choosen == 'weight_random':
            self.connections[conn].weight = random.uniform(-1,1)

    def bias_perturb(self, choosen):

        '''
        
        Mutates weights of nodes.

        '''
        
        node = random.choice(range(self.inputs,self.max_node_count))

        if choosen == 'bias_perturbation':
            self.nodes[node].bias += random.uniform(-1,1)
        elif choosen == 'bias_random':
            self.nodes[node].bias = random.uniform(-1,1)

    
    def forward_propagate(self, inputs):

        '''
        
        Forward propagation through evaluation of inputs and calculating 
        the outputs 
        
        '''

        if len(inputs) != self.inputs:

            raise ValueError('length of inputs do not match')

        for i in range(len(inputs)):
            self.nodes[i].output = inputs[i]
        
        adj_node_list = dict()
        for i in range(self.max_node_count):
            adj_node_list[i] = list()

        for (i,j) in self.connections:
            if not self.connections[(i,j)].enabled:
                continue
            adj_node_list[j].append(i)

        all_nodes = itertools.chain(range(self.unhidden_nodes,self.max_node_count),
                                    range(self.inputs,self.unhidden_nodes))

        for n in all_nodes:
            idx = 0
            for c in adj_node_list[n]:
                idx += self.connections[(c,n)].weight * self.nodes[c].output
            node = self.nodes[n]
            node.output = node.activation(idx + node.bias)

        return [self.nodes[i].output for i in range(self.inputs,self.unhidden_nodes)]
    
    def random_connections(self):

        '''
        
        Returns random connections. i != j while i is not an output and j is not an input.

        '''

        i = random.choice([n for n in range(self.max_node_count) if not self._is_output_node(n) ])
        j_all = [n for n in range(self.max_node_count) if not self._is_input_nodes(n) and n != i]

        if not j_all:
            j = self.max_node_count
            self.add_node()
        else:
            j = random.choice(j_all)
        
        return (i,j)
    
    def make_env(self, seed, render_mode=False, env_config=None):
        self.render_mode = render_mode
        self.env = make_env(self.env_name, seed=seed,
                            render_mode=render_mode, env_config=env_config)

    def add_enabled(self):
        '''Enables the disabled connection'''
        disabled = [c for c in self.connections if not self.connections[c].enabled]
        if len(disabled) > 0:
            self.connections[random.choice(disabled)].enabled = True

    def reset(self):
        '''Reset the internal state of genome by setting node outputs to 0'''
        for n in range(self.max_node_count):
            self.nodes[n].output = 0
        self.fitness = 0
    
    def clone(self):
        '''To copy genomes'''
        return copy.deepcopy(self)

    def _is_disabled(self):
        '''Check if it is disabled'''
        return all(self.connections[i].enabled == False for i in self.connections)
    

    def _is_output_node(self, node):
        '''To check if it is output node?'''
        return self.inputs <= node < self.unhidden_nodes

    def _is_input_nodes(self, node):
        '''To check if it is input node?'''
        return 0 <= node < self.inputs

    def _get_nodes(self):
        '''Get node object'''
        return self.nodes.copy()
    
    def _get_quantity_nodes(self):
        '''Get len of nodes'''
        return self.max_node_count
    
    def _get_connections(self):
        '''Get connection object'''
        return self.connections.copy()

    def _get_fitness(self):
        '''Get fitness'''
        return self.fitness

    def _set_fitness(self, fitness_score):
        '''Set fitness'''
        self.fitness = fitness_score

    
