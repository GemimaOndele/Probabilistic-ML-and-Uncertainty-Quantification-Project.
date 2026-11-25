"""Run after installing the complete requirements on CPU."""
import json
import random
import numpy as np
import torch
import torchvision
from evaluation.contracts import validate_predictions
from experiment import seed_python

seed_python(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
x = torch.rand(2, 3)
p = torch.softmax(x, dim=1).tolist()
validate_predictions(["smoke-0", "smoke-1"], [0, 1], p)
print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__,
                  "numpy": np.__version__, "seed": 42, "status": "ok"}))
