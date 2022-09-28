# Code is modified from:
# https://github.com/uber-research/poet/blob/master/poet_distributed/
# under Apache license 



class Niche:
    def rollout_batch(self, genomes, batch_size, random_state, eval=False):
        import numpy as np
        fitnesses = np.zeros(batch_size)
        

        for i, genome in enumerate(genomes):
            fitnesses[i] = self.rollout(
                genome, random_state=random_state, eval=eval)

        return fitnesses
