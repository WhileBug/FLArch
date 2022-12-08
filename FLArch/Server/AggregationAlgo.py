import copy
import torch

class AggregationAlgo:
    def __init__(self):
        pass

    def average_weights(self, weigh_update_list:[torch.nn]):
        """
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(weigh_update_list[0])
        for key in w_avg.keys():
            for i in range(1, len(weigh_update_list)):
                w_avg[key] += weigh_update_list[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weigh_update_list))
        return w_avg