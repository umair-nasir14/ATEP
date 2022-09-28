# Code is modified from:
# https://github.com/uber-research/poet/blob/master/poet_distributed/
# under Apache license 


import csv
from pprint import pformat
import logging

logger = logging.getLogger(__name__)


class CSVLogger:
    def __init__(self, fnm, col_names):
        logger.info('Creating data logger at {}'.format(fnm))
        self.fnm = fnm
        self.col_names = col_names
        #print('self.col_names',self.col_names)
        with open(fnm, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(col_names)
        # hold over previous values if empty
        self.vals = {name: None for name in col_names}

    def log(self, **cols):
        self.vals.update(cols)
        logger.info(pformat(self.vals))
        #print('self.vals:', self.vals)
        if any(key not in self.col_names for key in self.vals):
            raise Exception('CSVLogger given invalid key')
        with open(self.fnm, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([self.vals[name] for name in self.col_names])
