# CSR tensor support in PyTorch

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2021-04-07      |

The aim of this blog post is to propose a roadmap for completing the
implementation of the CSR layout for PyTorch tensors that is started
in [PR 50937](https://github.com/pytorch/pytorch/pull/50937).

The CSR layout was introduced to resolve the issue of slow matrix
multiplication with sparse matrices when using the existing COO layout
support in PyTorch. See [Roadmap for torch.sparse Matrix Product
API](https://github.com/dhavide/rfcs/blob/master/RFC-0004-pyTorch-sparse-matmul-roadmap.md)
for overview. The PR 50937 resolves this issue with high success: the
performance of matrix multiplication of sparse matrix-vector and
sparse matrix-dense matrix increases by ten-fold when using the CSR
layout, another five-fold increase is achieved when using Intel MKL
library tools, see [CSR vs COO
benchmarks](https://github.com/pytorch/pytorch/pull/44190#issue-479538842).

However, maintaining the PR 50937 and preparing it for landing has
become an increasingly difficult task because of 
- the constant flux in PyTorch and used libraries APIs that causes
  periodic merge conflicts,
- inherit inference between different PyTorch features (Autograd,
  TorchScript, Testing, etc) that must be taken into account when
  implementing a new storage layout,
- the current conversation count (> 250), the discussion items count
  (ca 400), and the commit count (ca 260) makes hard to grasp the
  overall state of the PR.

[It has been
agreed](https://github.com/pytorch/pytorch/pull/50937#issuecomment-811253895)
that the PR will land as it is but the work on the CSR layout support
needs to continue to meet the (still envolving) PyTorch coding and
testing standards.

Here, an attempt will be made to organize and discuss the follow-up
tasks for completing the CSR layout support.

##  Unresolved discussion items

- https://github.com/pytorch/pytorch/pull/50937#discussion_r562636792 - simplify under contiguity assumption
- https://github.com/pytorch/pytorch/pull/50937#discussion_r562646336, https://github.com/pytorch/pytorch/pull/50937#discussion_r562648465 - hardcoded development style
- https://github.com/pytorch/pytorch/pull/50937#discussion_r562656595 - docs
- https://github.com/pytorch/pytorch/pull/50937#discussion_r563184880 - VS build failure
- https://github.com/pytorch/pytorch/pull/50937#discussion_r563185162 - int32 vs int64 performance
- https://github.com/pytorch/pytorch/pull/50937#issuecomment-766777337 - is it fixed?
- https://github.com/pytorch/pytorch/pull/50937#issuecomment-766778930 - is it fixed?
- https://github.com/pytorch/pytorch/pull/50937#issuecomment-803905480 - is it fixed?
- https://github.com/pytorch/pytorch/pull/50937#discussion_r603533421 - is it fixed?
- https://github.com/pytorch/pytorch/pull/50937#discussion_r603535187 - is it fixed?
- https://github.com/pytorch/pytorch/pull/50937#discussion_r603536011 - resize internal tensors for memory efficiency
- https://github.com/pytorch/pytorch/pull/50937#discussion_r603537183 - eliminate memory format
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604353856 - test return value
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604354153 - use C++ RAII
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604379873 - dims/size checks
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604968109

## MKL and Windows build issues

- https://github.com/pytorch/pytorch/pull/50937#issuecomment-778288404
  and conversation following that. The issue is summarized in
  https://github.com/pytorch/pytorch/pull/50937#issuecomment-779272492
  with possible solutions.

## MKL and macOS build issues

- https://github.com/pytorch/pytorch/pull/50937#issuecomment-803909578

## CSR indices support int32 and int64

- https://github.com/pytorch/pytorch/pull/50937#discussion_r603538191 - expect one int type
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604349014 - why not default to int32?
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604969458

IIUC, it would be preferred to support a single dtype for CSR indices
tensors. (Explain why this preference).  While COO uses only int64 as
dtype for indices, a natural choice for the dtype of CSR
`crow/col_indices` would also be `int64`. As a result, the situation
would be simpler for users, conversion from COO to CSR and CSR to COO
would be memory/processor efficient, etc. However, a big performance
gain in matrix multiplication is achieved when using Intel MKL library
tools but with the current MKL support in PyTorch, only int32 indices
can be used as inputs to MKL routines. 

So, we could fix dtype to int64 (as in COO) but at the expense that
one needs int64->int32 conversion (and inverse?) whenever calling an
MKL routine.

We could also fix dtype to int32 (that would be most efficient when
using MKL support) but at the expense that all conversions between COO
and CSR would be more expensive than necessary.

Btw, the conversion between COO and CSR is important because the COO
layout is the most human-friendly layout for constructing sparse
tensors while the CSR layout is computationally much more efficient
than COO.

## COO to CSR conversion

- https://github.com/pytorch/pytorch/pull/50937#discussion_r604346660 - slow because implemented in Python
- https://github.com/pytorch/pytorch/pull/50937#discussion_r608326213 - dense-csr without coo

Not much to discuss here: for efficiency, implement the direct dense
to CSR conversion in C++. This will be important when we decide that
the indices of COO and CSR will have different dtypes. Otherwise, I
would not expect much performance gain.

## Testing

- https://github.com/pytorch/pytorch/pull/50937#discussion_r604348471 - see modern COO testing
- https://github.com/pytorch/pytorch/pull/50937#discussion_r608327057 - don't set default_dtype_type
- https://github.com/pytorch/pytorch/pull/50937#discussion_r608327057 - avoid using numpy
- https://github.com/pytorch/pytorch/pull/50937#discussion_r608675701 - inefficient CSR samples

## Avoid COO-isms

- https://github.com/pytorch/pytorch/pull/50937#discussion_r604364248
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604365304 - empty indices
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604369842 - aliasing, what happens to CSR after resizing values?
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604379081 - always require contiguity?
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604379537 - device testing
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604384386 - introduce sparse_csr namespace
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604974231
- https://github.com/pytorch/pytorch/pull/50937#discussion_r608312412 - docs

## Code quality

- https://github.com/pytorch/pytorch/pull/50937#discussion_r604972242
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604972910 - CPU logic
- https://github.com/pytorch/pytorch/pull/50937#discussion_r604989807 - avoid auto-deduction
- https://github.com/pytorch/pytorch/pull/50937#discussion_r608326624 - use assertExpectedInline


## Deal for landing PR 50937

- https://github.com/pytorch/pytorch/pull/50937#issuecomment-811253895 - the deal of landing
- https://github.com/pytorch/pytorch/pull/50937#issuecomment-812888029 - tentative plan
- https://github.com/pytorch/pytorch/pull/50937#discussion_r608312830 - dtype/device parameters
- https://github.com/pytorch/pytorch/pull/50937#issuecomment-814944029 - more todo items

## Main features missing

- CUDA support in the CSR layout.
- Inference with Autograd - this is relevant also to COO layout as to
  sparse tensor support in general.
- Generalization of CSR as N-dimensional tensor
