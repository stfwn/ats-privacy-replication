# python -u searchalg/batch_generate.py  --arch=ResNet20-4 --data=cifar100 
import copy, random
import argparse


def write(scheme_list, num_per_gpu=20):
    '''
    I propose to delete this later
    This write to the console run commands of benchmark/search_transform_attack.py foreach policy set in scheme list. Uses harcoded mode "aug" and 100 epochs params.
    I guess it divides them in "num_per_gpu" sets for pararel(?)
    '''
    for i in range(len(scheme_list) // num_per_gpu):
        print('{')
        for idx in range(i * num_per_gpu, i * num_per_gpu + num_per_gpu):
            sch_list = [str(sch) for sch in scheme_list[idx]]
            suf = '-'.join(sch_list)

            cmd = 'CUDA_VISIBLE_DEVICES={} python benchmark/search_transform_attack.py --aug_list={} --mode=aug --arch={} --data={} --epochs=100'.format(
                i % 8, suf, opt.arch, opt.data)
            print(cmd)
        print('}')


def backtracing(num:int = 20, max_augset_size: int = 3):
    '''
    This function produces 80*num different policies number
    PS: Don't know what the name comes from and what it's supposed to mean.
    :param max_augset_size: maximum number of the policies (augmentations) in the set
    :param num: something that I have no idea what it is but from it the number of policies depends
    '''

    '''
    This way producing subsets has probability of one policy in the set to not exist(subset of 2)
    of 1/50. 
    I think better solution would be to first randomly choose size of subset from <1,max_augset_size> and then pick those. 
    The distribution would change then so I didn't change it
    
    What if new policy is empty?
    '''
    scheme_list = []
    for _ in range(80 * num):
        new_policy = [random.randint(-1, 50) for _ in range(max_augset_size)]
        # what if all chosen policies are -1???
        new_policy = [item for item in new_policy if item != -1]
        scheme_list.append(new_policy)
    return scheme_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
    parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
    parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
    parser.add_argument('--number-per-gpu', default=20, required=True, type=int, help='Vision dataset.')
    opt = parser.parse_args()

    number_per_gpu = opt.number_per_gpu

    policies_list = backtracing(number_per_gpu)
    write(policies_list, number_per_gpu)
