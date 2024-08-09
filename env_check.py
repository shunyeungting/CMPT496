import torch
import datasets


# check if MPS is available
if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("mps")
else:
    print("MPS is not available")
    device = torch.device("cpu")

x = torch.randn(3, 3, device=device)
print(x)
print(datasets[0])
