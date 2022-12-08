from torch.nn import Module

class WeightUpdatePruning:
    def noPruning(
            self,
            weightUpdate:Module
    ):
        return weightUpdate