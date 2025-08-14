import os
import random
import numpy as np
import torch


# Reproducibility
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Option
    if hasattr(torch, "use_deterministic_algorithm"):
        torch.use_deterministic_algorithms(True)


# Checking hardware's capability
def is_bf16_supported():
    if not torch.cuda.is_available():
        return False

    cuda_capability, _ = torch.cuda.get_device_capability()
    return cuda_capability >= 8
