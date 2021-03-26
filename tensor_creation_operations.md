# Tensor creation operations in PyTorch

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2021-03-22      |

The aim of this blog post is to propose a classification of PyTorch
tensor creation operations and seek for the corresponding testing
patterns.

This blog post is inspired by the [new PyTorch testing
framework](https://github.com/pytorch/pytorch/wiki/Writing-tests-in-PyTorch-1.8)
that introduces the OpInfo pattern to simplify writing tests for
PyTorch. However, it does not yet provide a solution to the problem of
testing tensor creation operations that would be required in [PR
54187](https://github.com/pytorch/pytorch/pull/54187), for instance.

# Introduction

In general, Tensor instances can be created from other Tensor
instances as a result of tensor operations. But not only. Here we
consider tensor creation operations that inputs can be arbitrary
Python objects from which new Tensor instances can be constructed.
For instance, tensors can be constructed from objects that
implement Array Interface protocol, or from Python sequences that
represent array-like structures, or from Python integers as in the
case of torch.zeros, torch.arange, etc. The created tensors may or
may not share the memory with the input objects depending on the
particular operation as well as on the used device or dtype
parameter values.

# Tensor creation operations

To distinguish tensor creation operations from other operations, we
define the tensor creation operations as operations that result
a Tensor instance with user-specified

- device property: `cpu`, `cuda`, etc
- dtype property: `torch.int8`, `torch.int16`, ..., `torch.float32`, `torch.float64`, ..., `torch.complex64`, `torch.complex128`, ...
- layout property: `torch.strided`, `torch.sparse_coo`, `torch.sparse_csr`
- or are constructed from non-Tensor objects.

According to this definition, PyTorch implements the following
tensor creation operations:

- construction from user-specified data and layout:
  ```python
  # N-D strided tensors, always copy
  torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor

  # N-D strided tensors, memory may be shared
  torch.as_tensor(data, dtype=None, device=None) → Tensor
  torch.from_numpy(ndarray) → Tensor

  # N-D sparse tensors
  torch.sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False) → Tensor

  # 2-D sparse tensors
  torch.sparse_crs_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, requires_grad=False) → Tensor
  ```

- construction from user-specified shape and data is computed:
  ```python
  # 1-D tensors
  torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  torch.logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

  # 2-D tensors
  torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

  # N-D tensors
  torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
  torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
  torch.full_like(input, fill_value, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
  ```

- construction from user-specified shape but data is left unspecified:
  ```python
  # N-D tensors
  torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor
  torch.empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

  # N-D strided tensors
  torch.empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) → Tensor
  ```

- construction from user-specified shape and data is random (computed from pseudo-random generator):
  ```python
  # N-D tensors
  torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  torch.rand(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

  torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
  torch.rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
  torch.randint_like(input, low=0, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

  # 1-D tensor with size n
  torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor
  ```

- construction from user-specified dtypes and Tensor data:
  ```python
  # N-D tensors with complex dtype
  torch.complex(real, imag, *, out=None) → Tensor
  torch.polar(abs, angle, *, out=None) → Tensor

  # N-D tensors with dtypes quint8, qint8, and qint32
  torch.quantize_per_tensor(input, scale, zero_point, dtype) → Tensor
  torch.quantize_per_channel(input, scales, zero_points, axis, dtype) → Tensor
  torch.dequantize(tensor) → Tensor
  ```

### Notes

- Many of the tensor creation operations use `requires_grad` input
  parameter. In the context of testing tensor creation operations, the
  `requires_grad` argument does not define the content and data
  location of Tensor data and hence can be ignored here.
- The `memory_format` input parameter is used in all `torch.*_like`
  functions and it has an affect to the strides information of tensors
  using strided layout.
- Few tensor creation operations use `pin_memory` input parameter that
  affects the choice of memory allocation address space and allows
  accessing data buffers from CPU or CUDA processes without the need
  to explicitly copy Tensor data in between devices.
- In the case of `torch.*_like` functions, the default values of
  `size`, `dtype`, `device`, and `layout` are determined by the input
  tensor.
- The behaviour of the `out` input parameter is defined [in
  here](https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch,
  although a lot of existing behaviour is left undefined, for
  instance, the cases where `out` already contains data, etc. When
  `out` parameter is specified, it will be the tensor object returned
  from the operation, possibly resized.
- The CSR layout support exists currently in [PR
  50937](https://github.com/pytorch/pytorch/pull/50937).


## Testing tensor creation operations

In general, testing of an operation for a correct behaviour (as
specified in its documentation, or defined by mathematics, or
determined by some supported protocol, etc) involves checking if a
given input produces an expected result. In the case of tensor
creation operations, the input parameters define not only the content
of tensor data buffers (dtype, values, shape) but also how and where
the data is stored in memory (layout, strides, device, input memory
shared or not, etc). Different from OpInfo framework goals, testing
with respect to Autograd support is mostly irrelevant as the inputs to
tensor creation operations are not PyTorch Tensor objects (except for
few cases like `torch.sparse_coo_tensor`, `torch.complex`, etc, and
for view operations).

Let `Op(...) -> Tensor` be a tensor creation operation.  There exists
a number of patterns that can be defined for testing the correctness
of `Op`:

1.  Specifying a tensor property in an argument list must result in a
    tensor that has this property:

    ```python
    Op(..., dtype=dtype).dtype == dtype
    Op(..., device=device).device == device
    Op(..., layout=layout).layout == layout
    Op_like(input, ...).dtype == input.dtype
    Op_like(input, ...).device == input.device
    Op_like(input, ...).layout == input.layout
    ```
    whereas the explict definition of a property in `Op_like` argument
    list overrides the corresponding property of `input` tensor.

2.  Specification of `out` parameter will lead to the same result as in
    the case of unspecified `out` parameter, unless the `out` argument
    specification is erroneous (e.g. it has a different dtype from the
    dtype of an expected result):

    ```
    Op(*args, out=a, **kwargs) == a == Out(*args, **kwargs)
    ```

3.  For Tensor creation operations that have NumPy analogies, such as
    `zeros`, `ones`, `arange`, etc, use NumPy functions as reference
    implementations:

    ```python
    torch.Op(...).numpy() == numpy.Op(...)
    ```

    Warnings: `numpy.Op` may have a different user-interface from the
    corresponding `torch.Op` one. Using NumPy functions as reference is
    based on the assumption that NumPy functions behave
    correctly. Seldomly, the definitions of correctness may vary in
    between projects.

4.  The content of a tensor does not depend on the used storage
    device, data storage layout, nor memory format:

    ```python
    torch.Op(..., device=device).to(device='cpu') == torch.Op(..., device='cpu')
    torch.Op(..., layout=layout).to_dense() == torch.Op(..., layout=torch.strided)
    torch.Op(..., memory_format=memory_format).to(memory_format=torch.contiguous_format) == torch.Op(..., memory_format=torch.contiguous_format)
    ```

5.  Tensors created using `pin_memory=True` must be accessible from CUDA device:

    ```python
    class W:
        def __init__(self, tensor):
           self.__cuda_array_interface__ = tensor.__array__.__array_interface__

    x = torch.Op(..., pin_memory=True)
    assert x.device == 'cpu'
    t = torch.as_tensor(W(x), device='cuda')
    assert w.device.startswith('cuda')
    x += 1                                   # modify x in-place
    assert (t.to(device='cpu') == x).all()   # changes in x are reflected in t
    ```

6.  When `Op` represents a random tensor creation operation, its
    correctness must be verified using statistical methods (e.g. by
    computing the statistical moments of results, and comparing these
    with the expected values, approximately).

7.  Tensor constructor `torch.tensor` must be able to construct a
    tensor from the following objects:

    - NumPy ndarray objects
    - nested sequences of numbers
    - objects implementing CPU/CUDA Array Interface protocols (as in [PR 54187](https://github.com/pytorch/pytorch/pull/54187))
    - objects implementing PEP 3118 Buffer protocol

    whereas the resulting tensor must *not* share the memory with the
    input data buffer.

8.  Tensor constructor `torch.as_tensor` must be able to construct a
    tensor from the following objects:

    - PyTorch Tensor instance
    - NumPy ndarray objects
    - nested sequences of numbers
    - objects implementing CPU/CUDA Array Interface protocols (as in [PR 54187](https://github.com/pytorch/pytorch/pull/54187))
    - objects implementing PEP 3118 Buffer protocol

    whereas the resulting tensor may share the memory with the input
    data buffer.

9.  All view operations must be tested by modifying the view result,
    and then checking if the corresponding changes appear in the
    original tensor:

    ```
    a = Op(x)  # create a view of x
    a += 1     # modify the view in-place
    a == Op(x) # recreating the view gives the modified view
    ```

## The current state of testing tensor creation operations in PyTorch

Clearly, many of the tensor creation operations are heavily used in
PyTorch testing suite for creating inputs to various tests of PyTorch
functionality. However, when extending the functionality of tensor
creation operations, it is not always obvious where the corresponding
tests should be implemented. Also, not all tensor creation operations
are systematically tested.

For instance, let us consider `torch.as_tensor` operation. It has
unit-tests implemented in
[test/test_tensor_creation_ops.py:TestTensorCreation.test_as_tensor()](https://github.com/pytorch/pytorch/blob/e5b97777e3bfd426681fa1b242573bd232eb675d/test/test_tensor_creation_ops.py#L2058-L2114)
that covers the following parameter cases:

- Only CPU device case is tested. While the test method contains `if
  torch.cuda.is_available():` blocks containing CUDA device cases but
  these are disabled by the usage of `onlyCPU` decorator.

- Only the following dtype cases are used in tests: `float64`,
  `float32`, `int64`, `int8`, `uint8`. Although, this can be
  considered a negligible problem because it is unlikely that untested
  dtype cases would have serious issues.

- Only Tensor and list instances are used as inputs to
  `torch.as_tensor` to test the tensor creation operation.

- Only Tensor instances are used as inputs to `torch.as_tensor` to test
  the copy and non-copy effect of the operation.

[test/test_tensor_creation_ops.py:TestTensorCreation.test_tensor_ctor_device_inference()](https://github.com/pytorch/pytorch/blob/e5b97777e3bfd426681fa1b242573bd232eb675d/test/test_tensor_creation_ops.py#L2470) covers `torch.as_tensor(..., device=device).device == device` for CPU and CUDA devices and float32/64 dtype.

[test/test_tensor_creation_ops.py:TestTensorCreation.test_tensor_factories_empty()](https://github.com/pytorch/pytorch/blob/e5b97777e3bfd426681fa1b242573bd232eb675d/test/test_tensor_creation_ops.py#L2535) covers creating an empty tensor from list via `torch.as_tensor()`.

[test/test_numba_integration.py:TestNumbaIntegration.test_cuda_array_interface](https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/test/test_numba_integration.py#L19)
covers testing CUDA Array Interface via `torch.as_tensor()`.


Finally, when implementing [PR
54187](https://github.com/pytorch/pytorch/pull/54187), I noticed that
`torch.tensor` was creating a non-copy Tensor instance from an object
that implements Array Interface. The operation `torch.tensor` should
always copy the data buffers but the current test-suite did not catch
the non-copy behavior while ideally it should have. It means that not
all requirements specified in the `torch.tensor` documentation have
been tested.
