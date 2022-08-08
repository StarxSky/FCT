#
# © 2022 Apple Inc.
#

import torch

version = torch.__version__
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(f'PyTorch version :{version}')
print(f"Device :{device}")
