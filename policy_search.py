import random
import sys
import torch.cuda

from joblib import Parallel, delayed
from queue import Queue
from argparse import Namespace

sys.path.insert(0, "./original/")
from original.benchmark.search_transform_attack import (
    main as search_transform_main,
)


EMPTY_TRANSFORMATION = -1
NUM_TRANSFORMATIONS = 50


def parallel_policy_search(
    num_schemes: int,
    model: str,
    data: str,
    epochs: int,
    num_transform: int = 3,
    num_per_gpu: int = 20,
    num_images: int = 1,
):
    gpu_queue = Queue()
    num_gpu = max(torch.cuda.device_count(), 1)
    for gpu_ids in range(num_gpu):
        for _ in range(num_per_gpu):
            gpu_queue.put(gpu_ids)

    schemes = create_schemes(num_schemes, num_transform)
    Parallel(n_jobs=num_gpu * num_per_gpu, require="sharedmem")(
        delayed(search_transform_attack)(
            scheme, model, data, epochs, num_images, gpu_queue
        )
        for scheme in schemes
    )


def search_transform_attack(scheme, model, data, epochs, num_images, gpu_queue):
    gpu = gpu_queue.get()
    try:
        scheme_str = "-".join(map(str, scheme))
        opt = Namespace(
            **dict(
                aug_list=scheme_str,
                mode="aug",
                arch=model,
                data=data,
                epochs=epochs,
                num_images=num_images,
            )
        )
        if torch.cuda.is_available():
            with torch.cuda.device(gpu):
                search_transform_main(opt)
        else:
            search_transform_main(opt)
    finally:
        gpu_queue.put(gpu)


def create_schemes(num_schemes: int, num_transform: int = 3):
    schemes = []
    for _ in range(num_schemes):
        scheme = [random_transformation() for _ in range(num_transform)]
        scheme = [t for t in scheme if t != EMPTY_TRANSFORMATION]
        schemes.append(scheme)
    return schemes


def random_transformation():
    return random.randint(EMPTY_TRANSFORMATION, NUM_TRANSFORMATIONS)
