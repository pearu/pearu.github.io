---
author: Pearu Peterson
created: 2021-06-16
---

# CSR random sampling

This blog post is inspired by https://github.com/pytorch/pytorch/issues/59379 that seeks
for a better sampling method for generating random CSR tensors to be used for testing PyTorch CSR
tensor support. In the following, we'll review the currently used method, then define what is a good sampling method,
and finally, propose a new sampling method for CSR tensors.

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

Pros:
- the algorithm is very simple

Cons:
- while `col_indices` and `values` are random, `crow_indices` are not. This means that this sampler always
  produces CSR tensors with regular `crow_indices` with a specific structure while other possible structures are
  not generated for some fixed `nnz` value no matter how large is the number of samples. 
- specifying `nnz` does not guarantee that the number of specified values in the generated sample will be the same
  (except when `nnz` is a integer multiple of `n_rows`).

As an example, the following animation generates a series of samples with specified `nnz` varying from `0` to `n_rows * c_cols`.

![PyTorch 17x5 sample - current](distribute_column_indices_17x5_pytorch.gif)

Observations:
- for a wide range of `nnz` values, rows with no entries or rows with all columns specified,
  are never generated (min/max number of columns per row is not equal to `0/n_cols`),
- only for few specified values of `nnz`, it will be equal to the actual `nnz` value of the sample (observe the actual NNZ value),
- the distribution of specified elements looks uniform, however, this is not a good property, see the next section below.

## Quality of random samples

To test the correctness of some functionality, using random samples with uniformly distributed specified elements
are not always optimal because edge cases such as existence of rows with no specified elements, or with no unspecified elements at all,
are practically never generated. However, often only the edge cases may reveal possible bugs in the corresponding
algorithms/implementations.

In general, the quality of random samples that are used for testing is not about the
quality of the distribution of random placements of specified elements of the sparse tensor. Instead, it is about
maximally stressing the algorithms logic with a minimal effort. That is, a *good sample* has the following properties:
- it is small in order to minimize computation time when processing it in the algorithms,
- it will make all possible branches in the algorithm to be alive to maximize testing coverage,
- its structure is highly variable in order reveal any possible shortcomings of the algorithm,
- it must have some intrinsic randomness that will increase the variability of the structure even more when
  tests are run multiple times and/or from different environments,
- finally, its generation must be efficient.

In the case of generating samples of CSR tensors, the current quality of random samples is limited by the quality of `crow_indices`
samples. So, in the following we aim at applying the properties of a good sample specifically to `crow_indices` and
at the same time fix the structural issues of the current sampler method, e.g., ensure that the actual `nnz`
will be equal to the specified `nnz` parameter, as well as, ensure that samples will contain various edge cases if possible.

## Sampling of `crow_indices`

To compute `crow_indices`, we are using the following model:

```python
crow_indices = cumsum([0] + counts)
```

where `counts` is a list of integers with the following properties:

- `len(counts) == n_rows`
- `counts[i]` is the number of specified entries in the `i`-th row
- `0 <= counts.min()` and `counts.max() <= n_cols`
- `counts.sum() == nnz`
- `counts[-1] - counts[0]` is as large as possible to maximize structural variability
- the distibution of different count values is as uniform as possible to have some balance between normal and the edge cases

In addition, we require that the computation of `counts` has complexity not greater than `O(max(n_rows, n_cols))`.

Clearly, there exists many solutions to `counts` that satisfy the above listed properties.

Here we propose a new algorithm that is based on computing the `counts` values from the following
histogram:

```
      ^ counts is the number of columns per row
      |
      |

        *   *   *   *   ###
       **  **  **  **+ o###
      *** *** *** ***+oo###
      @@@@@@@@@@@@@@@@@@###
      @@@@@@@@@@@@@@@@@@###      --> row indices
```
where different parts of the histogram are denoted as follows:
- `<space>` - no counts
- `+` - final correction
- `o` - an incomplete sawtooth
- `*` - a sequence of full sawteeth
- `@` - lower rectangle
- `#` - right rectangle

Pseudo-code for computing the above histogram is as follows:

```python
# Inputs: n_rows, n_cols, nnz
# Outputs: counts

counts = zeros(n_rows)

def N(n, m):
    # compute the total number of counts in the sequence of sawteeth
    M = (n_cols - m) * (n_cols - m + 1) // 2
    K = (n_rows - n) % (n_cols - m + 1)
    return M * ((n_rows - n) // (n_cols - m + 1)) + K * (K - 1) // 2

# Find n such that N(n, 0) == 0 or nnz < max(N(n, 0), n_cols)
if n > 0:
    counts[-n:] = n_cols                        - this fills the region denoted by #

# Find m such that N(n, m) == 0 or nnz - n * n_cols < max(N(n, m), n_cols)
if m > 0:
    counts[:n_rows - n] = m                     - this fills the region denoted by @

if N(n, m) == 0:  # no sawteeth
    counts[0] = nnz - n * n_cols - m * n_rows
else:
    M = (n_cols - m) * (n_cols - m + 1) // 2
    q = ((nnz - n * n_cols - m * n_rows) // M) * (n_cols - m + 1)
    # Find k such that k*(k+1)/2 <= (nnz - n * n_cols - m * n_rows) % M
    corr = (nnz - n * n_cols - m * n_rows) % M - k * (k + 1) // 2
    counts[:q] = arange(q) % (n_cols - m + 1)   - this fills the region denoted by *
    counts[q:q+k+1] += arange(k + 1)            - this fills the region denoted by o
    counts[q] += corr                           - this fills the region denoted by +
```

Notice that the filling of `counts` can use vectorized operations.
To solve the equations `N(n, m) == 0` for integers `n` and `m`, one can use the bisection algorithm.

For example, the following animation uses the above described construction of `crow_indices`:

![PyTorch 17x5 sample - new](distribute_column_indices_17x5_new.gif)
