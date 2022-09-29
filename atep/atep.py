# Code is modified from:
# https://github.com/uber-research/poet/blob/master/poet_distributed/
# under Apache license 



from copy import copy
from atep.niches.box2d import model

from atep.niches.box2d.model import Model
from .logger import CSVLogger
import logging
logger = logging.getLogger(__name__)
import numpy as np
from atep.es import ESOptimizer
from atep.es import initialize_worker_fiber
from collections import OrderedDict
from atep.niches.box2d.env import Env_config, make_env, bipedhard_custom
from atep.niches.box2d.cppn import CppnEnvParams
from atep.reproduce_ops import Reproducer
from atep.novelty import compute_novelty_vs_archive
import json

from neat.neat import NEAT
from neat.config import Hyperparameters

import ray
import os
import random

import sys

import pickle

import time

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


import sys
import inspect
import logging

logger = logging.getLogger(__name__)



@ray.remote
def eval_a_genome(genome,seed,env_config,env_params,model):
    
    fitnesses = []
    runs_per_net=1
    max_episode_length = 2000
    if env_config:
        model.model.env.set_env_config(env_config)

    if env_params:
        model.model.env.augment(env_params)
    
    for _ in range(runs_per_net):

        

        observation = model.model.env.reset()
        fitness = 0.0
        done = False
        
        for t in range(max_episode_length):
            if observation is None:
                observation = np.zeros(model.model.inputs)

            action = genome.forward_propagate(observation)
            observation, reward, done, _ = model.model.env.step(action)
            fitness += reward
            

            if done:
                break

        fitnesses.append(fitness)
    
    
    return np.mean(fitnesses)



@ray.remote
def construct_niche_fns_from_env(args, env, env_params, seed):
    def niche_wrapper(configs, env_params, seed):  # force python to make a new lexical scope
        def make_niche():
            from atep.niches import Box2DNiche
            return Box2DNiche(env_configs=configs,
                            env_params=env_params,
                            seed=seed,
                            init=args.init,
                            stochastic=args.stochastic)

        return make_niche

    niche_name = env.name
    configs = (env,)

    return niche_name, niche_wrapper(list(configs), env_params, seed)


class ATEP:
    def __init__(self, args):

        self.args = args


        self.ANNECS = 0
        self.env_registry = OrderedDict()
        self.env_archive = OrderedDict()
        self.env_reproducer = Reproducer(args)
        self.optimizers = OrderedDict()
        self.archived_optimizers = OrderedDict()
        self.neat_params = Hyperparameters(args)
       
        env = Env_config(
            name='flat',
            ground_roughness=0,
            pit_gap=[],
            stump_width=[],
            stump_height=[],
            stump_float=[],
            stair_height=[],
            stair_width=[],
            stair_steps=[])

        params = CppnEnvParams()
        self.add_optimizer(env=env, cppn_params=params, initial_pop=True, seed=args.master_seed)

    def create_optimizer(self, env, cppn_params, seed, created_at=0, initial_pop = False, model_params=None, is_candidate=False):

        assert env != None
        assert cppn_params != None

        optim_id, niche_fn = ray.get(construct_niche_fns_from_env.remote(args=self.args, env=env, env_params=cppn_params, seed=seed))
        
        niche = niche_fn()
            
        assert optim_id not in self.optimizers.keys()


        return NEAT(
            optim_id=optim_id,
            inputs=24,
            outputs=4, 
            population=self.neat_params.population, 
            hyperparams=self.neat_params,
            args=self.args,
            make_niche=niche,
            genomes=None,
            created_at=created_at,
            log_file=self.args.log_file,
            is_candidate=is_candidate,
            env = env,
            env_params = cppn_params
        )

    


    def add_optimizer(self, env, cppn_params, seed, created_at=0, initial_pop=True, model_params=None):
        '''
            creat a new optimizer/niche
            created_at: the iteration when this niche is created
        '''
        o = self.create_optimizer(env, cppn_params, seed, created_at,initial_pop, model_params)
        optim_id = o.optim_id
        self.optimizers[optim_id] = o


        assert optim_id not in self.env_registry.keys()
        assert optim_id not in self.env_archive.keys()
        self.env_registry[optim_id] = (env, cppn_params)
        self.env_archive[optim_id] = (env, cppn_params)

    def archive_optimizer(self, optim_id):
        assert optim_id in self.optimizers.keys()
        #assume optim_id == env_id for single_env niches
        o = self.optimizers.pop(optim_id)
        assert optim_id in self.env_registry.keys()
        self.env_registry.pop(optim_id)
        logger.info('Archived {} '.format(optim_id))
        self.archived_optimizers[optim_id] = o

    def ind_neat_step(self, iteration,func_evals):
    

        results = dict()
        x = 0
        y = 0
        for optimizer in self.optimizers.values():
            optimizer.update_fittest()
            
            for i in range(len(optimizer.species)):
                for j in range(len(optimizer.species[i].members)):
                    results[(x+i,y+j)] = eval_a_genome.remote(optimizer.species[i].members[j],optimizer.args.master_seed,optimizer.env_configs,optimizer.env_params,optimizer.make_niche)
                    optimizer.func_evals +=1
                y += (len(optimizer.species[i].members) - 1)
            x += (len(optimizer.species)-1)
            

        a=0
        b=0
        
        for optimizer in self.optimizers.values():

            for i in range(len(optimizer.species)):
                for j in range(len(optimizer.species[i].members)):
                    genome = optimizer.species[i].members[j]
                    genome._set_fitness(ray.get(results[(a+i,b+j)]))
                b += (len(optimizer.species[i].members) - 1)
            a += (len(optimizer.species)-1)

            optimizer.evolution()
            
        for optimizer in self.optimizers.values():
            
            current_best = optimizer._get_fittest_genome()
            species = optimizer._get_species_len()
            logger.info(" Iter={} | Optimizer {}  | Fitness Score: {} | Population: {} | Species: {} | iteration spent {}".format(
                iteration,
                optimizer.optim_id,
                current_best._get_fitness(), 
                optimizer._get_population(),
                species,
                iteration - optimizer.created_at
            ))
            for optimizer in self.optimizers.values():
                optimizer.update_dicts_after_es(self_eval_stats=current_best._get_fitness())



    def transfer(self, propose_with_adam, checkpointing, reset_optimizer, transfer_type):
        
        if transfer_type == 'FBT':
            logger.info('Computing direct transfers...')
            proposal_targets = {}
            for source_optim in self.optimizers.values():
                

                source_tasks = []
                proposal_targets[source_optim] = []
                source_genome = source_optim._get_fittest_genome()
                for target_optim in [o for o in self.optimizers.values()
                                        if o is not source_optim]:
                    
                    task = target_optim.evaluate_genome(source_genome,None,None,target_optim.make_niche)
                    source_tasks.append((task, target_optim))
                
                for task, target_optim in source_tasks:

                    try_proposal = target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                        source_optim_theta=source_genome,
                                        stats=task, keyword='theta')
                    if try_proposal:
                        proposal_targets[source_optim].append(target_optim)

            logger.info('Computing proposal transfers...')
            for source_optim in self.optimizers.values():
                source_tasks = []
                genome = source_optim._get_fittest_genome()
                for target_optim in [o for o in self.optimizers.values()
                                        if o is not source_optim]:
                    if target_optim in proposal_targets[source_optim]:
                        task = target_optim.evaluate_genome(genome,None,None,target_optim.make_niche)
                        source_tasks.append((task, target_optim))

                for task, target_optim in source_tasks:

                    target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                        source_optim_theta=genome,
                        stats=task, keyword='proposal',
                        source_optim=source_optim)

        elif transfer_type == "SBT":

            logger.info('Computing specie transfers...')
            for source_optim in self.optimizers.values():
                for target_optim in [o for o in self.optimizers.values()
                                        if o is not source_optim]:
                                        
                    target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                        source_optim_theta=source_optim._get_fittest_genome(),
                        stats=0, keyword=transfer_type,
                        source_optim=source_optim)
            logger.info('Considering transfers...')
            for o in self.optimizers.values():
                o.pick_proposal(checkpointing, reset_optimizer)


        elif transfer_type == 'specie_and_fitness':

            '''
            For future works.
            '''

            specie_targets = {}
            source_tasks = []
            for source_optim in self.optimizers.values():
                
                specie_targets[source_optim] = []
                best_genomes = [s._get_best_genome() for s in source_optim.species]

                for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:

                
                    target_delta = target_optim.delta(target_optim._get_fittest_genome(), source_optim_genome, target_optim.hyperparams.delta_coefficients)

                    if target_delta <= target_optim.hyperparams.delta_threshold:
                            specie_targets[source_optim].append(target_optim)

                    if target_optim in specie_targets:
                        task = target_optim.evaluate_genome(source_optim_genome,None,None,target_optim.make_niche)
                        source_tasks.append((task, target_optim))

                for task, target_optim in source_tasks:
                    logger.info('Transfering by specie and fitness')
                    target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                            source_optim_theta=source_optim_genome,
                            stats=task, keyword='specie_and_fitness',
                            source_optim=source_optim)
        
        elif transfer_type == 'Random':

            source_optim = {}

            K=5

            if len(list(self.optimizers.values())) < K:
                
                K = 2

            random_optims = random.choices(list(self.optimizers.values()),k=K)

            for source_optim in random_optims:
                source_optim_genome = []#max(best_genomes, key= lambda g:g.fitness)

                for target_optim in [o for o in random_optims
                                    if o is not source_optim]:
                    logger.info('Random transfer initiated')
                    target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                            source_optim_theta=source_optim_genome,
                            stats=0, keyword=transfer_type,
                            source_optim=source_optim)







        logger.info('Considering transfers...')
        for o in self.optimizers.values():
            o.pick_proposal(checkpointing, reset_optimizer)

    def check_optimizer_status(self, iteration):
        '''
            return two lists
        '''
        logger.info("health_check")
        repro_candidates, delete_candidates = [], []
        for optim_id in self.env_registry.keys():
            o = self.optimizers[optim_id]
            logger.info("niche {} created at {} start_score {} current_self_evals {}".format(
                optim_id, o.created_at, o.start_score, o.self_evals))
            if o.self_evals >= self.args.repro_threshold:
                repro_candidates.append(optim_id)

        logger.debug("candidates to reproduce")
        logger.debug(repro_candidates)
        logger.debug("candidates to delete")
        logger.debug(delete_candidates)

        return repro_candidates, delete_candidates


    def pass_dedup(self, env_config):
        if env_config.name in self.env_registry.keys():
            logger.debug("active env already. reject!")
            return False
        else:
            return True

    def pass_mc(self, score):
        if score == None:
            return False
        else:
            if score < self.args.mc_lower or score > self.args.mc_upper:
                return False
            else:
                return True

    def get_new_env(self, list_repro):

        optim_id = self.env_reproducer.pick(list_repro)
        assert optim_id in self.optimizers.keys()
        assert optim_id in self.env_registry.keys()
        parent_env_config, parent_cppn_params = self.env_registry[optim_id]
        child_env_config = self.env_reproducer.mutate(parent_env_config, no_mutate=True)
        child_cppn_params = parent_cppn_params.get_mutated_params()

        logger.info("we pick to mutate: {} and we got {} back".format(optim_id, child_env_config.name))
        logger.debug("parent")
        logger.debug(parent_env_config)
        logger.debug("child")
        logger.debug(child_env_config)

        seed = np.random.randint(1000000)
        return child_env_config, child_cppn_params, seed, optim_id

    def get_child_list(self, parent_list, max_children):
        child_list = []

        mutation_trial = 0
        while mutation_trial < max_children:
            new_env_config, new_cppn_params, seed, parent_optim_id = self.get_new_env(parent_list)
            mutation_trial += 1
            if self.pass_dedup(new_env_config):
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                score = o.evaluate_genome(self.optimizers[parent_optim_id]._get_fittest_genome(),None,None,o.make_niche)
                if self.pass_mc(score):
                    novelty_score = compute_novelty_vs_archive(self.archived_optimizers, self.optimizers, o, k=5,
                                        low=self.args.mc_lower, high=self.args.mc_upper)
                    logger.info("{} passed mc, novelty score {}".format(score, novelty_score))
                    child_list.append((new_env_config, new_cppn_params, seed, parent_optim_id, novelty_score))
                del o

        #sort child list according to novelty for high to low
        child_list = sorted(child_list,key=lambda x: x[4], reverse=True)
        return child_list

    def adjust_envs_niches(self, iteration, steps_before_adjust, max_num_envs=None, max_children=8, max_admitted=1):

        if iteration > 0 and iteration % steps_before_adjust == 0:
            
            list_repro, list_delete = self.check_optimizer_status(iteration)

            if len(list_repro) == 0:
                return

            logger.info("list of niches to reproduce")
            logger.info(list_repro)
            logger.info("list of niches to delete")
            logger.info(list_delete)

            for optim in self.optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper)

            for optim in self.archived_optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper)

            child_list = self.get_child_list(list_repro, max_children)
            
            if child_list == None or len(child_list) == 0:
                logger.info("mutation to reproduce env FAILED!!!")
                return
            
            admitted = 0
            
            for child in child_list:
                
                new_env_config, new_cppn_params, seed, _, _ = child
                
                # targeted transfer
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                score_child, theta_child = o.evaluate_transfer(self.optimizers)
                score_archive, _ = o.evaluate_transfer(self.archived_optimizers, evaluate_proposal=False)
                del o
                if self.pass_mc(score_child):  # check mc
                    
                    self.add_optimizer(env=new_env_config, cppn_params=new_cppn_params, seed=seed, created_at=iteration, model_params=np.array(theta_child))
                    admitted += 1
                    if self.pass_mc(score_archive):
                        self.ANNECS += 1
                    if admitted >= max_admitted:
                        break
            
            
            if max_num_envs and len(self.optimizers) > max_num_envs:
                num_removals = len(self.optimizers) - max_num_envs
                self.remove_oldest(num_removals)
            
            

    def remove_oldest(self, num_removals):
        list_delete = []
        for optim_id in self.env_registry.keys():
            if len(list_delete) < num_removals:
                list_delete.append(optim_id)
            else:
                break

        for optim_id in list_delete:
            self.archive_optimizer(optim_id)           

    def optimize(self, iterations=200,
                 steps_before_transfer=25,
                 propose_with_adam=False,
                 checkpointing=False,
                 starting_from_checkpoint=False,
                 check_pointing_file='checkpoint_400',
                 reset_optimizer=True):

        
        func_evals = 0
        if starting_from_checkpoint:
            random.seed(self.args.master_seed)
            with open('/mnt/lustre/users/mnasir/neat-poet-master/poet_distributed/'+check_pointing_file, 'rb') as f:
                func_evals, self.ANNECS,self.optimizers, self.archived_optimizers,self.env_registry,self.env_archive, iter = pickle.load(f)
            
        else:
            iter = 0

        for iteration in range(iter, iterations):
            
            self.adjust_envs_niches(iteration, self.args.adjust_interval * steps_before_transfer,
                                    max_num_envs=self.args.max_num_envs)
            

            self.ind_neat_step(iteration=iteration,func_evals=func_evals)


            if len(self.optimizers) > 1 and iteration % steps_before_transfer == 0:

                if self.args.transfer_type == "no_transfer":
                    pass
                else:
                    self.transfer(propose_with_adam=propose_with_adam,
                            checkpointing=checkpointing,
                            reset_optimizer=reset_optimizer,
                            transfer_type=self.args.transfer_type)
                                
                
            if iteration % steps_before_transfer == 0:
                for o in self.optimizers.values():
                    func_evals += o.func_evals
                    o.save_to_logger(iteration,self.ANNECS,func_evals)

            if checkpointing and iteration % 51 == 0:
                random.seed(self.args.master_seed)
                optim_pack = (func_evals, self.ANNECS, self.optimizers,self.archived_optimizers,self.env_registry,self.env_archive,iteration)
                with open('checkpoint_'+self.args.transfer_type+'_' + str(iteration), 'wb') as f:
                    pickle.dump(optim_pack, f)

