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



import multiprocessing
from os import remove
from pickletools import optimize

from .species import Species

import ray





import numpy as np
import random
import json
import time
import logging
import functools
from collections import namedtuple
logger = logging.getLogger(__name__)

from atep.stats import compute_centered_ranks, batched_weighted_sum
from atep.niches.box2d.env import make_env, bipedhard_custom, Env_config

from atep.logger import CSVLogger


class NEAT:

    '''
    Base class for NEAT algorithm.
    '''

    def __init__(self,
                optim_id, 
                inputs, 
                outputs, 
                population, 
                hyperparams,
                args,
                make_niche,
                genomes=None, 
                created_at = 0,
                log_file='unname.log',
                is_candidate=False,
                env=None,
                env_params=None):
        
        self.optim_id = optim_id
        self.inputs = inputs
        self.outputs = outputs

        self.species = []
        self.population = population
        self.generation = 0
        self.current_species = 0
        self.current_genome = 0
        self.global_best = None

        self.hyperparams = hyperparams

        self.genomes = genomes
        self.created_at = created_at

        self.args = args

        #self.fiber_shared = fiber_shared
        #niches = fiber_shared["niches"]
        #niches[self.optim_id] = make_niche
        self.make_niche = make_niche
        
        self.checkpoint_thetas = None
        self.checkpoint_scores = None

        self.self_evals = None   # Score of current parent theta
        self.proposal = None   # Score of best transfer
        self.proposal_theta = None # Theta of best transfer
        self.proposal_source = None # Source of best transfer

        self.created_at = created_at
        self.start_score = None

        self.best_score = None
        self.best_theta = None
        self.recent_scores = []
        self.transfer_target = None
        self.pata_ec = None

        self.iteration = 0

        self.func_evals = 0

        if is_candidate == False:
            log_fields = [
                'fitness_{}'.format(optim_id),
                'accept_theta_in_{}'.format(optim_id),
                'eval_returns_mean_from_others_in_{}'.format(self.optim_id)    
            ]
            log_path = log_file + '/' + log_file.split('/')[-1] + '.' + optim_id + '.csv'
            self.data_logger = CSVLogger(log_path, log_fields + [
                'time_elapsed_so_far',
                'iteration',
                'ANNECS',
                'FUNCTION_EVAL_COUNT'
            ])
            logger.info('Optimizer {} created!'.format(optim_id))

        self.filename_best = log_file + '/' + log_file.split('/')[-1] + '.' + optim_id + '.best.json'
        self.log_data = {}
        self.t_start = time.time()

        self.env_configs = env 
        self.env_params = env_params


    def delta(self, genome_a,genome_b, delta_coefficients):
        
        '''Implements Delta formula'''

        a_genes = set(genome_a.connections)
        b_genes = set(genome_b.connections)

        similar_genes = a_genes & b_genes
        disjoint_and_excess = (a_genes - b_genes) | (b_genes - a_genes)

        N_genes = len(max(a_genes, b_genes, key=len))
        if N_genes < 40: #Tunable parameter. Suggested 20 but 10 works fine for xor
            N_genes = 1
        N_nodes = min(genome_a.max_node_count,genome_b.max_node_count)

        weight_difference = 0
        bias_difference = 0


        for i in similar_genes:

            weight_difference += abs(genome_a.connections[i].weight - 
                                    genome_b.connections[i].weight)

        for i in range(N_nodes):

            bias_difference += abs(genome_a.nodes[i].bias - 
                                    genome_b.nodes[i].bias)
        
        # We do for node and connection weights seperatedly, and disjoint
        # and excess togather which makes it easier to implement.

        s1 = delta_coefficients['c3'] * bias_difference/N_nodes
        s2 = delta_coefficients['c2'] * len(disjoint_and_excess)/N_genes
        s3 = delta_coefficients['c3'] * weight_difference/len(similar_genes)

        #print(s1 + s2 + s3)
        
        return s1 + s2 + s3
        
    
    def classify_genome(self, genome):

        '''Classifying Genomes into the species'''

        if not self.species: # Check if there is no Specie then make one.
            self.species.append(Species(self.hyperparams.max_fitness_history, genome))

        else:

            for s in self.species:
                representative = s.members[0]
                distance = self.delta(genome, representative, self.hyperparams.delta_coefficients)

                if distance <= self.hyperparams.delta_threshold:
                    s.members.append(genome)
                    return
            # New specie for different genome
            self.species.append(Species(self.hyperparams.max_fitness_history, genome))

    def initial_population(self):

        '''Initial population of defined number of individuals.'''

        
        from .genome import Genome
        genomes = []
        for i in range(self.population):
            genome = Genome(self.inputs, self.outputs, self.hyperparams.activation)
            genome.generate_init_nn()
            self.classify_genome(genome)
            genomes.append(genome)

    
        self.global_best = self.species[0].members[0]

        #return genomes
    
    def update_fittest(self):

        '''Check the fittest and update it.'''
        #print('updating fittest')
        best_genomes = [s._get_best_genome() for s in self.species]
        current_best = max(best_genomes, key= lambda g:g.fitness)

        if current_best.fitness > self.global_best.fitness:
            self.global_best = current_best.clone() 

        self.self_evals = self.global_best.fitness
        self.genome = self.global_best
    def evolution(self):

        '''A process of one generaation.'''
        #print('evolution() check 1')
        global_fitness_sum = 0
        for s in self.species:
            s.update_fitness()
            global_fitness_sum +=s.fitness_sum

        if global_fitness_sum == 0: #Check progress
            for s in self.species:
                for g in s.members:
                    g.mutate(self.hyperparams.mutation_probability,self.hyperparams.connection_weight_probability)
        
        else:
            # keeping those that can improve
            survived = []
            for s in self.species:
                if s.can_progress():
                    survived.append(s)
            #print('evolution() check 2')    
            self.species = survived
            #print(f'len(survived) = {len(self.species)}')
            #eliminate weakest genomes
            for s in self.species:
                s.kill_genomes()
            #print(f'after killing genomes: {len(self.species)}')
            # Repopulate with a ratio w.r.t fitness here.
            #print('evolution() check 3')
            #difference = self.population - self._get_population()
            #ratio = difference/len(self.species)
            for i, s in enumerate(self.species):

                #ratio = s.fitness_sum/global_fitness_sum
                children = int(round(self.population/4)) + random.randint(-100,100) #int(round(ratio*difference))
                #print(f'children : {children}')
                for k in range(children):
                    self.classify_genome(s.crossover(self.hyperparams.mutation_probability,
                                         self.hyperparams.crossover_probability))

            # if no specie survives we mutate through minimal structure and the best individual

            if not self.species:
                #print(f'We are in repopulation part, species are: {self.species} and population is {self.population}')
                counter = 0
                for i in range(self.population):
                    #print(f'i = {i}')
                    if self._get_population() == 3:# Atleast 3 individuals are present
                        #print(f'We are in i%3 == 0 part, i is: {i}')
                        g = self.global_best.clone()
                    else:
                        from .genome import Genome
                        g = Genome(self.inputs,self.outputs,self.hyperparams.activation)
                        g.generate_init_nn()

                    counter += 1
                    
                    g.mutate(self.hyperparams.mutation_probability,self.hyperparams.connection_weight_probability)
                    self.classify_genome(g)
                    if counter == int(self.population/5):
                        break
            
            # cap the population
            '''if self._get_population() > self.population:
                print(f'self._get_population check1: {self._get_population()}')
                genomes = [s.members for s in self.species]
                #genomes.sort(key=lambda genome:genome.fitness, reverse=True)
                genomes = genomes[:self.population]

                for genome in genomes:
                    self.classify_genome(genome)
                print(f'self._get_population check1: {self._get_population()}')'''
        self.generation += 1



    def evolution_process(self):

        '''Termination conditions.'''

        self.update_fittest()
        fitness_based = self.global_best.fitness <= self.hyperparams.max_fitness
        generation_based = self.generation != self.hyperparams.max_generations

        return fitness_based and generation_based

    def increment_iteration(self):

        '''Evolves and increments to the next generation.'''

        s = self.species[self.current_species]
        if self.current_genome < len(s.members) + 1:
            self.current_genome += 1
        else:
            if self.current_species < len(self.species)-1:
                self.current_species +=1
                self.current_genome = 0
            else:
                # evolution
                self.evolution()
                self.current_genome = 0
                self.current_species = 0
    
    def evaluate_genome(self, genome,env_config,env_params,make_niche):

        from atep.poet_algo import eval_a_genome
        #best_genome = self._get_fittest_genome()
        #print(f'genome:{genome}')
        result = dict()
        result[(0,1)] = eval_a_genome.remote(genome,self.args.master_seed,env_config,env_params,make_niche)
        self.func_evals +=1
        #genome = self.species[result[0]].members[result[1]]
        for res in result:
            genome._set_fitness(ray.get(result[res]))
        return genome.fitness


        
    def parallel_eval_processes(self,eval,env_config,env_params, model, *args, **kwargs):

        '''Does parallel evalutaion on CPU.'''

        results = dict()
        #print(f'species : {self.species}')

        for i in range(len(self.species)):
            for j in range(len(self.species[i].members)):
                results[(i,j)] = eval.remote(self.species[i].members[j],self.args.master_seed,env_config,env_params,model)
                self.func_evals +=1
                #print(f'genome in evaluation : {self.species[i].members[j]}')
                #print(f'results[(i,j)] i.e fitness : {results[(i,j)]}')
                #results[(i,j)] = eval(self.species[i].members[j])
                                        
     
        for result in results:
            #print('result check')
            genome = self.species[result[0]].members[result[1]]
            #print(f'result for genome: {genome}')
            genome._set_fitness(ray.get(results[result]))
            #print(f'fitness of {genome} is {ftns} ')
            #genome._set_fitness(results[result])
        #print('evaluation check 1')
        self.evolution()

    
    def update_dicts_after_transfer(self, source_optim_id, source_optim_theta, stats, keyword, source_optim=None):
        eval_key = 'eval_returns_mean_from_others_in_{}'.format(self.optim_id)
        self.log_data.update({
            eval_key : source_optim_id + '_' + str(stats)
        })


        if keyword == 'proposal' and stats > self.transfer_target:
            if  stats > self.proposal:
                self.proposal = stats
                self.proposal_source = source_optim_id + ('' if keyword=='theta' else "_proposal")
                self.proposal_theta = np.array(source_optim_theta)
                
                # Replacing whole population
                print(f'tranferring from {source_optim_id} to {self.optim_id}')
                pop_bef = self._get_population()
                print(f'Population of {pop_bef} being deleted')
                self.species = []
                
                print('Copying whole population...')
                for s in source_optim.species:
                    for genome in s.members:
                        self.classify_genome(genome)
                pop_now = self._get_population()
                print(f'Population after copying is {pop_now}')
                self.update_fittest()
        
        if keyword == 'proposal_n%' and stats > self.transfer_target:
            if  stats > self.proposal:
                self.proposal = stats
                self.proposal_source = source_optim_id + ('' if keyword=='theta' else "_proposal")
                self.proposal_theta = np.array(source_optim_theta)
                
                # Replacing whole population
                print(f'tranferring from {source_optim_id} to {self.optim_id}')
                pop_bef = self._get_population()
                print(f'Population of Target optim: {pop_bef}')
                individuals = []
                for s in self.species:
                    for genome in s.members:
                        individuals.append(genome)
                self.species = []
                individuals.sort(key=lambda genome:genome.fitness, reverse=True)
                r = int(len(individuals)*(0.6))
                print(f'Deleting {int(len(individuals)-r)} individuals')
                individuals = individuals[:r]

                for individual in individuals:
                    self.classify_genome(individual)

                source_inds = []
                for s in source_optim.species:
                    for genome in s.members:
                        source_inds.append(genome) 
                source_inds.sort(key=lambda genome:genome.fitness, reverse=True)
                rs = int(len(source_inds)*(0.4))
                source_inds = source_inds[:rs]

                
                print(f'Copying {rs} individuals...')
                for genome in source_inds:
                    self.classify_genome(genome)
                pop_now = self._get_population()
                print(f'Population after copying is {pop_now}')
                self.update_fittest()
            

            elif keyword == 'specie_and_fitness':

            

                if stats > self.transfer_target:

                    if source_optim_theta not in [s.members for s in source_optim.species]:
                            #print(f'source_optim_genome is not in specie')
                        source_optim.classify_genome(source_optim_theta)
                    for s in source_optim.species:
                        #print(source_optim_theta)
                        #print(s.members)
                        if source_optim_theta in s.members:
                            inds = 0
                            s.members.sort(key=lambda genome:genome.fitness, reverse=True)
                            logger.info(f'Population before transfer: {self._get_population()}')
                            logger.info(f'Population of specie being transferred: {len(s.members[:100])}')
                            if len(s.members) < 100:
                                inds = len(s.members)
                            else:
                                inds = 100
                            for genome in s.members:#[:inds] for max 100 individuals to transfer
                                self.classify_genome(genome)
                            logger.info(f'Population after transfer: {self._get_population()}')
                            self.update_fittest()
                            #s.members.pop(s.members.index(source_optim_theta))
                else:
                    logger.info('No transfer possible')
            
            elif keyword == 'random':

                self.proposal = stats
                self.proposal_source = source_optim_id + ('' if keyword=='theta' else "_proposal")
                #self.proposal_theta = np.array(source_optim_theta)
                
                # Replacing whole population
                print(f'tranferring from {source_optim_id} to {self.optim_id}')
                pop_bef = self._get_population()
                print(f'Population of {pop_bef} being deleted')
                self.species = []
                
                print('Copying whole population...')
                for s in source_optim.species:
                    for genome in s.members:
                        self.classify_genome(genome)
                pop_now = self._get_population()
                print(f'Population after copying is {pop_now}')
                self.update_fittest()

                

        return stats > self.transfer_target
    
    def update_dicts_after_es(self, self_eval_stats):

        self.self_evals = self_eval_stats
        if self.start_score is None:
            self.start_score = self.self_evals
        self.proposal = self_eval_stats
        self.proposal_source = self.optim_id
        self.proposal_theta = np.array(self._get_fittest_genome())

        if self.checkpoint_scores is None:
            self.checkpoint_thetas = np.array(self._get_fittest_genome())
            self.checkpoint_scores = self_eval_stats


        if self.best_score is None or self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = np.array(self._get_fittest_genome())

        assert len(self.recent_scores) <= 5
        if len(self.recent_scores) == 5:
            self.recent_scores.pop(0)
        self.recent_scores.append(self.self_evals)
        self.transfer_target = max(self.recent_scores)

        self.log_data.update({
            'fitness_{}'.format(self.optim_id):
                self.self_evals,
            'accept_theta_in_{}'.format(self.optim_id): 'self'
        })

       
    
    def pick_proposal(self, checkpointing, reset_optimizer):

        accept_key = 'accept_theta_in_{}'.format(
                self.optim_id)
        if checkpointing and self.checkpoint_scores > self.proposal:
            self.log_data[accept_key] = 'do_not_consider_CP'
        else:
            self.log_data[accept_key] = '{}'.format(
                self.proposal_source)
            if self.optim_id != self.proposal_source:
                self.genome=self.proposal_theta
                self.self_evals = self.proposal

        self.checkpoint_thetas = np.array(self.genome)
        self.checkpoint_scores = self.self_evals

        if self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = np.array(self.genome)
        


    def update_pata_ec(self, archived_optimizers, optimizers, lower_bound, upper_bound):
        def cap_score(score, lower, upper):
            if score < lower:
                score = lower
            elif score > upper:
                score = upper

            return score

        raw_scores = []
        for source_optim in archived_optimizers.values():
            raw_scores.append(cap_score(source_optim._get_fittest_genome().fitness,lower_bound,upper_bound))

        for source_optim in optimizers.values():
            raw_scores.append(cap_score(source_optim._get_fittest_genome().fitness,lower_bound,upper_bound))

        self.pata_ec = compute_centered_ranks(np.array(raw_scores))

    
    def save_to_logger(self, iteration,ANNECS,FUNCTION_EVAL_COUNT):
        self.log_data['time_elapsed_so_far'] = time.time() - self.t_start
        self.log_data['iteration'] = iteration
        self.log_data['ANNECS'] = ANNECS
        self.log_data['FUNCTION_EVAL_COUNT'] = FUNCTION_EVAL_COUNT
        self.data_logger.log(**self.log_data)

        logger.debug('iter={} Optimizer {} best score {}'.format(
            iteration, self.optim_id, self.best_score))

        #if iteration % 100 == 0:
        #    self.save_policy(self.filename_best+'.arxiv.'+str(iteration))

        #self.save_policy(self.filename_best)
    
    def evaluate_transfer(self, optimizers, evaluate_proposal=True, propose_with_adam=False):

        best_init_score = None
        best_init_genome = None

        for source_optim in optimizers.values():
            score = self.evaluate_genome(source_optim._get_fittest_genome(),None,None,self.make_niche)#Change to genome
            if best_init_score == None or score > best_init_score:
                best_init_score = score
                best_init_genome = np.array(source_optim._get_fittest_genome())

            if evaluate_proposal:
                '''task = self.start_step(source_optim.theta)
                proposed_theta, _ = self.get_step(
                    task, propose_with_adam=propose_with_adam, propose_only=True)
                score = self.evaluate_theta(proposed_theta)'''
                if score > best_init_score:
                    best_init_score = score
                    best_init_genome = np.array(source_optim._get_fittest_genome())

        return best_init_score, best_init_genome



    def _del_population(self):

        for s in self.species:
            for genome in s.members:
                del genome

    def _get_population(self):
        '''Gets the number of individuals'''
        return sum([len(s.members) for s in self.species])
    
    def get_population(self):
        '''Gets the population'''
        return [s.members for s in self.species]

    def copy_population(self):
        '''Gets population in the form of species'''
        return [s for s in self.species]

    def _get_fittest_genome(self):
        '''Gets the fittest genome.'''
        return self.global_best

    def _get_current_genome(self):
        '''Gets the current genome'''
        specie = self.species[self.current_species]
        return specie.members[self.current_genome]
    
    def _get_current_species(self):
        '''Gets the current specie'''
        return self.current_species
    
    def _get_generation(self):
        '''Gets the current generation'''
        return self.generation
    
    def _get_species(self):
        '''Gets all species'''
        return self.species
    
    def _get_species_len(self):
        '''Gets number of species.'''
        return len(self.species)









