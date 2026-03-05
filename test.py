import torch

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPU 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Get the number of available CUDA devices
print("Number of CUDA devices:", torch.cuda.device_count())

# Get the current CUDA device
print("Current CUDA device:", torch.cuda.current_device())

# Create a random tensor on the GPU
tensor = torch.rand(1000, 1000).cuda()

# Perform a matrix multiplication on the GPU
result = torch.matmul(tensor, tensor.T)

# Print the result
print("Matrix multiplication result:\n", result)
