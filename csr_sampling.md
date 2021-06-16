---
author: Pearu Peterson
created: 2021-06-16
---

# CSR random sampling

This blog post is inspired by https://github.com/pytorch/pytorch/issues/59379 that seeks
for a better sampling method for generating random CSR tensors to be used for testing PyTorch CSR
tensor support. In the following, we'll review the currently used method, then define what is a good sampling method,
and finally, propose a better sampling method for CSR tensors.

## Current state

At the time of writting this, PyTorch [implements](https://github.com/pytorch/pytorch/blob/8c4e78129ec8d71d587ac5d143ad17e4b95b3576/torch/testing/_internal/common_utils.py#L1068-L1092)
the following algorithm for random CSR tensor samples (here given a slightly modified version for clarity):

```python
# Inputs: n_rows, n_cols, nnz
# Outputs: crow_indices, col_indices, values
nnz_per_row = nnz // n_rows
crow_indices = torch.zeros(n_rows + 1)
if nnz_per_row > 0:
    crow_indices[1:] = nnz_per_row
else:
    crow_indices[1:nnz + 1] = 1
crow_indices.cumsum_(dim=0)
actual_nnz = crow_indices[-1]
col_indices = torch.randint(0, n_cols, size=[actual_nnz])
values = make_tensor([actual_nnz], low=-1, high=1)
```

Notes:
- while `col_indices` and `values` are random, `crow_indices` are not. This means that the current sampler always
  produces CSR tensors with highly regular `crow_indices`
