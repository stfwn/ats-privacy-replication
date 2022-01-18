from typing import List

from original.searchalg.batch_generate import backtracing


class AugmentationSearch:
    def __init__(self):
        pass

    def get_policies(self, num = 80*20, max_augset_size:int =3):
        '''
        :param num: The number of random policy sets, default is 1600- basically it get's random subset of policy sets.
        :param max_augset_size: maximum number of the policies (augmentations) in the set
        :returns: list of policy sets
        '''
        return backtracing(num=int(num/80), max_augset_size=max_augset_size)

    def find_bests(self, policies: List[List[int]]):
        '''
        Finds best sets of policies from all. Firstly by evaluating ... (4.2) then by evaluating ... (4.3)
        Saves the evaluation results
        Then it runs filtering algorithm (4.4)
        :returns: list of chosen policies (stats?)
        '''
        # run them (not bad if in parralel) while also saving the results (Spri(4.2), accuracy from 4.3) PSNR(?)
        # run algorithm that filters out bad sets and gets the best augmentations set (4.4)
        pass


if __name__=="__main__":
    # (set of augmentations that are best, results of why are they best)
    # their instruction
    #
    # - this generates list of different arguments for search_transform_attack.py:
    # python -u searchalg/batch_generate.py  --arch=ResNet20-4 --data=cifar100 > batch_generate.sh
    # - here all the search_transform_attack.py are run:
    # bash batch_generate.sh
    # - this looks for best alg
    # python -u searchalg/search_best.py --arch=ResNet20-4 --data=cifar100

    aug_search = AugmentationSearch()
    policies = aug_search.get_policies()
    best_policies, stats = aug_search.find_bests(policies)

    # generate list of different setups/augmentations to test
    # run them (not bad if in parralel) while also saving the results (Spri(4.2), accuracy from 4.3) PSNR(?)
    # run algorithm that filters out bad sets and gets the best augmentations set (4.4)