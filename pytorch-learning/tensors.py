"""
Tensors.
"""
import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

x_ones = torch.ones_like(x_data)
print(x_ones)

x_ones_float = torch.ones_like(x_data, dtype=torch.float)
print(x_ones_float)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)


shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

rand_tensor = torch.rand(*shape)
print(f"Random Tensor: \n {rand_tensor} \n")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

tensor = torch.rand(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")
print(f"Second last and last column:\n {tensor[:, -2:]}")
tensor[:, 1] = 0
print(tensor, end='\n\n')

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1, end='\n\n')

# This computes the matrix multiplication between two tensors. y1, y2, y3 will
# have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(f'y1: {y1}')
print(f'y2: {y2}')
print(f'y3: {y3}', end='\n\n')


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f'z1: {z1}')
print(f'z2: {z2}')
print(f'z3: {z3}', end='\n\n')

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"tensor (before): \n{tensor} \n")
tensor.add_(5)
print(f"tensor (after): \n{tensor} \n")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


n = np.ones(5)
print(f"n: {n}")
t = torch.from_numpy(n)
print(f"t: {t}")

np.add(n, 1, out=n)
print(f"n: {n}")
print(f"t: {t}")
