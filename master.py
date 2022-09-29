# Code is modified from:
# https://github.com/uber-research/poet/blob/master/poet_distributed/
# under Apache license


from argparse import ArgumentParser
from ast import parse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
from atep.atep import ATEP

import ray
import os

def run_main(args):


    ray.init(address=os.environ["thishostNport"])
    
    np.random.seed(args.master_seed)


    optimizer_zoo = ATEP(args=args)
    
    optimizer_zoo.optimize(iterations=args.n_iterations,
                       propose_with_adam=args.propose_with_adam,
                       reset_optimizer=True,
                       checkpointing=args.checkpointing,
                       starting_from_checkpoint=True,
                       check_pointing_file=args.start_from,
                       steps_before_transfer=args.steps_before_transfer)


def main():
    parser = ArgumentParser()
    parser.add_argument('log_file')
    ### NEAT params
    parser.add_argument('--delta_threshold',type=float, default=3.0)
    parser.add_argument('--neat_population',type=int,default=1000)
    parser.add_argument('--c1', type=float, default=1.0)
    parser.add_argument('--c2', type=float, default=1.0)
    parser.add_argument('--c3', type=float, default=3.7)
    parser.add_argument('--max_stagnation', type=int, default=60)
    parser.add_argument('--crossover_probability', type=float, default=0.3)
    parser.add_argument('--connection_weight_probability',type=float, default=0.95)
    parser.add_argument('--mutation_probability_node', type=float, default=0.15)
    parser.add_argument('--weight_mutate_large_probability', type=float, default=0.85)
    parser.add_argument('--bias_mutation_large_probability', type=float, default=0.85)
    
    ### General params
    parser.add_argument('--transfer_type',type=str,default='SBT')
    parser.add_argument('--n_iterations', type=int, default=200)
    parser.add_argument('--steps_before_transfer', type=int, default=25)
    parser.add_argument('--master_seed', type=int, default=111)
    parser.add_argument('--mc_lower', type=int, default=25)
    parser.add_argument('--mc_upper', type=int, default=340)
    parser.add_argument('--repro_threshold', type=int, default=200)
    parser.add_argument('--max_num_envs', type=int, default=20)
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--adjust_interval', type=int, default=4)
    parser.add_argument('--envs', nargs='+')
    parser.add_argument('--start_from_checkpointing', type=bool, default=False)
    parser.add_argument('--start_from', default=None)  # Json file to start from

    args = parser.parse_args()
    logger.info(args)

    run_main(args)

if __name__ == "__main__":
    main()
