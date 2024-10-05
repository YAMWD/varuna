# online batch selection sampler
import torch
import numpy as np
import math
import random
from bisect import bisect_right
from torch.utils.data.sampler import Sampler
import copy
from torchvision import transforms

class OBS_Sampler(Sampler):

    def __init__(self, model, losses_fn, data_source, train_data, train_target, batch_size, fac, pp1, pp2, epoch, device = 'cpu', drop_last = True):
        self.sampler = RandomSampler(model, losses_fn, data_source, train_data, train_target, batch_size, fac, pp1, pp2, epoch, device)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def set_epoch(self, epoch):
        self.sampler.epoch = epoch

    def __iter__(self):
        batch = []
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size
    
class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, model, losses_fn, data_source, train_data, train_target, batch_size, fac, pp1, pp2, epoch, device = 'cpu'):
        self.model = model
        self.losses_fn = losses_fn
        self.data_source = data_source
        self.batch_size = batch_size
        self.data = train_data
        self.target = train_target        
        self.device = device
	
        self.sorting_evaluations_period = 100   # increase it if sorting is too expensive
        self.sorting_evaluations_ago = 2 * self.sorting_evaluations_period

        self.fac = fac
        self.pp1 = pp1
        self.pp2 = pp2

        self.curii = 0
        self.lastii = 0
        self.ntraining = len(train_data)

        self.bfs = np.ndarray((self.ntraining, 2))
        for idx in range(0, self.ntraining):
            self.bfs[idx][0] = 1e+10
            self.bfs[idx][1] = int(idx)

        self.prob = [None] * self.ntraining     # to store probabilies of selection, prob[i] for i-th ranked datapoint
        self.sumprob = [None] * self.ntraining  # to store sum of probabilies from 0 to i-th ranked point

        self.epoch = epoch

        self.stop = 0
        self.iter = 0

        self.indexes = []

    def init_bfs(self, bfs):
        self.bfs = copy.deepcopy(bfs)

    def init_prob(self):
        mult = math.exp(math.log(self.fac) / self.ntraining)

        for i in range(0, self.ntraining):
            if (i == 0):    self.prob[i] = 1.0
            else:           self.prob[i] = self.prob[i - 1] / mult

        psum = sum(self.prob)

        self.prob = [v / psum for v in self.prob]
        for i in range(0, self.ntraining):
            if (i == 0):    self.sumprob[i] = self.prob[i]
            else:           self.sumprob[i] = self.sumprob[i-1] + self.prob[i]

        self.stop = 0 # reset stop flag for batch iteration
        self.iter = 0
        self.indexes = []

    def get_scores(self):
        output, feat = self.model.forward(self.data)
        criterion = nn.CrossEntropyLoss(reduce=False)
        loss = criterion(output, self.target)
        return loss

    def update(self, losses):
        i = 0
        indice = self.indexes
        for idx in indice:
            self.bfs[idx][0] = losses[i] # update loss for corresponding datapoint, rely on the computed index, so index cannot be wrapped into a sampler that is invisible to the training loop
            i = i + 1
        
        self.curii = self.curii + len(indice)

        if (self.pp1 > 0):
            training_flag = self.model.training
            self.model.eval()
            if (self.curii - self.lastii > self.ntraining / self.pp1):
                self.lastii = self.curii
                stopp = 0
                iii = 0
                bs_here = 32
                maxpt = int(self.ntraining * self.pp2)
                while (stopp == 0):
                    indexes = []
                    stop1 = 0
                    while (stop1 == 0):
                        index = iii
                        indexes.append(index)
                        iii = iii + 1
                        if (len(indexes) == bs_here) or (iii == maxpt):
                            stop1 = 1

                    if (iii == maxpt):
                        stopp = 1

                    idxs = []
                    for idx in indexes:
                        idxs.append(int(self.bfs[idx][1]))

                    # import pdb; pdb.set_trace()
                    inputs = self.data[idxs] 
                    targets = self.target[idxs]
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
            
                    with torch.no_grad():
                        losses = self.model(inputs, targets)
                    i = 0
                    for idx in indexes:
                        self.bfs[idx][0] = losses[i]
                        i = i + 1
            if training_flag:
                self.model.train() # switch back to train

    def __iter__(self):
        while (self.stop == 0):
            self.indexes = []
            wrt_sorted = 0
            if (self.epoch > 0):
                wrt_sorted = 1
                if (self.sorting_evaluations_ago >= self.sorting_evaluations_period):
                    self.bfs = self.bfs[self.bfs[:,0].argsort()[::-1]]
                    self.sorting_evaluations_ago = 0
            
            stop1 = 0
            while (stop1 == 0):
                index = self.iter
                if (wrt_sorted == 1):
                    randpos = min(random.random(), self.sumprob[-1])
                    index = bisect_right(self.sumprob, randpos)  # O(log(ntraining)), cheap
                self.indexes.append(index)
                self.iter = self.iter + 1
                if (len(self.indexes) == self.batch_size) or (self.iter == len(self.data_source)):
                    stop1 = 1

            self.sorting_evaluations_ago = self.sorting_evaluations_ago + self.batch_size
            
            if (self.iter == len(self.data_source)):
                # traversed the whole training dataset, proceed to the next epoch
                self.stop = 1

            idxs = []
            for idx in self.indexes:
                idxs.append(int(self.bfs[idx][1]))
            
            yield idxs

    def __len__(self):
        return len(self.data_source)
