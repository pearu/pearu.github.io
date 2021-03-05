# Overview of array and buffer protocols in Python

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2021-02-05      |

The aim of this blog post is to review the current state of CPU/CUDA Array Interfaces and PEP 3118 Buffer Protocol
in the context of NumPy and PyTorch, and give recommendations to improve the PyTorch support to these protocols.
We also indicate the overall usage of array interfaces in different Python libraries.

This blog post is inspired by a [PyTorch issue 51156](https://github.com/pytorch/pytorch/issues/51156).

## CPU Array Interface

[Array Interface (Version
3)](https://numpy.org/doc/stable/reference/arrays.interface.html)
defines a protocol for objects to re-use each other's data buffers.
It was created in 2005 within the [NumPy](https://numpy.org/) project for CPU array-like
objects. The implementation of the array interface is defined by the
existence of the following attributes or methods:

- `__array_interface__` - a Python dictionary that contains the shape,
  the element type, and optionally, the data buffer address and the
  strides of an array-like object.

- `__array__()` - a method returning NumPy ndarray view of an array-like object

- `__array_struct__` - holds a pointer to [PyArrayInterface
  C-structure](https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_struct__).


## CUDA Array Interface

Numba introduces [CUDA Array Interface (Version 2)](https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html)
for GPU array-like objects. The implementation of the CUDA array
interface is defined by the existence of the attribute

- `__cuda_array_interface__`

that holds the same information about an array-like object as `__array_interface__` except
the data buffer address will point to GPU memory area.

## Buffer Protocol

[PEP 3118 Buffer Protocol](https://www.python.org/dev/peps/pep-3118/) defines Python C/API
for re-using data buffers of buffer-like objects.
The Buffer protocol can implemented for extension types using Python C/API. Notice that
the buffer protocol cannot be implemented for types defined in Python:
this has been requested and discussed but no solution yet.
In Python, the data buffers of extension types can be accessed using `memoryview` object.


# Using array/buffer interfaces in the context of NumPy arrays

NumPy ndarray object implements CPU Array Interface as well as Buffer Protocol for sharing its data buffers:

```python
>>> import numpy
>>> arr = numpy.array([1, 2, 3, 4])
>>> arr.__array__()
array([1, 2, 3, 4])
>>> arr.__array_interface__
{'data': (93870472383472, False), 'strides': None, 'descr': [('', '<i8')], 'typestr': '<i8', 'shape': (4,), 'version': 3}
>>> arr.__array_struct__
<capsule object NULL at 0x7fa9e86c9750>
>>> memoryview(arr)
<memory at 0x7fab16419340>
```

NumPy ndarray can be used for wrapping arbitrary objects that implement the CPU Array Interface or Buffer Protocol:
```python
>>> data = numpy.array([1, 2, 3, 4, 5])  # this will be the only place where memory will be located for data
>>> class A1:
...     def __array__(self): return data
... 
>>> class A2:
...     __array_interface__ = data.__array_interface__
... 
>>> class A3:
...     __array_struct__ = data.__array_struct__
... 
>>> a1 = numpy.asarray(A1())
>>> a1[0] = 11
>>> a2 = numpy.asarray(A2())
>>> a2[1] = 21
>>> a3 = numpy.asarray(A3())
>>> a3[2] = 31
>>> m4 = memoryview(data)
>>> a4 = numpy.frombuffer(m4, dtype=m4.format)
>>> a4[3] = 41
>>> data
array([11, 21, 31, 41,  5])
```

## Recommendations

By default, `numpy.frombuffer(buf)` returns a NumPy ndarray with `dtype==numpy.float64` but discards `buf.format`.
I think it would make sense to use the `buf.format` for determing the `dtype` of the `numpy.frombuffer` result, as
demostrated in the following:
```python
>>> data = numpy.array([1, 2, 3, 4, 5])
>>> buf = memoryview(data)
>>> numpy.frombuffer(buf)
array([4.9e-324, 9.9e-324, 1.5e-323, 2.0e-323, 2.5e-323])
>>> numpy.frombuffer(buf, dtype=buf.format)
array([1, 2, 3, 4, 5])
```

# Using array/buffer interfaces in the context of PyTorch tensors

The following examples use PyTorch version 1.9.0a0.

PyTorch Tensor object implements the CPU Array Interface partly and does not implement Buffer Protocol:
```python
>>> import torch
>>> t = torch.tensor([1, 2, 3, 4, 5])
>>> arr = t.__array__()  # equivalent to numpy.asarray(t)
>>> arr[0] = 99
>>> t
tensor([99,  2,  3,  4,  5])
>>> t.__array_interface__
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Tensor' object has no attribute '__array_interface__'
>>> t.__array_struct__
AttributeError: 'Tensor' object has no attribute '__array_struct__'
>>> memoryview(t)
TypeError: memoryview: a bytes-like object is required, not 'Tensor'
```
However, since the `Tensor.__array__()` method returns a NumPy ndarray as a view of tensor data buffer,
the CPU Array Interface is effective to PyTorch tensors:
```
 >>> t.__array__().__array_interface__
{'data': (93915843345344, False), 'strides': None, 'descr': [('', '<i8')], 'typestr': '<i8', 'shape': (5,), 'version': 3}
>>> t.__array__().__array_struct__
<capsule object NULL at 0x7f4b9694e990>
>>> memoryview(t.__array__())
<memory at 0x7f4b96303d00>
```

PyTorch Tensor object implements the CUDA Array Interface:
```python
>>> t = torch.tensor([1, 2, 3, 4, 5], device='cuda')
>>> t.__cuda_array_interface__
{'typestr': '<i8', 'shape': (5,), 'strides': None, 'data': (139961292554240, False), 'version': 2}
```

PyTorch Tensor object cannot be used for wrapping arbitrary objects that implement the CPU Array Interface:
```python
>>> data = numpy.array([1, 2, 3, 4, 5])
>>> class A1:
...     def __array__(self): return data
... 
>>> class A2:
...     __array_interface__ = data.__array_interface__
... 
>>> class A3:
...     __array_struct__ = data.__array_struct__
... 
>>> t1 = torch.as_tensor(A1())
RuntimeError: Could not infer dtype of A1
>>> t2 = torch.as_tensor(A2())
RuntimeError: Could not infer dtype of A2
>>> t3 = torch.as_tensor(A3())
RuntimeError: Could not infer dtype of A3
```
However, wrapping the objects with NumPy ndarray first, one can effectively wrap arbitrary objects using PyTorch Tensor object:
```python
>>> t1 = torch.as_tensor(numpy.asarray(A1()))
>>> t1[0] = 101
>>> t2 = torch.as_tensor(numpy.asarray(A2()))
>>> t2[1] = 102
>>> t3 = torch.as_tensor(numpy.asarray(A3()))
>>> t3[2] = 103
>>> data
array([101, 102, 103,   4,   5])
```

PyTorch Tensor object implements the Buffer Protocol partly (or incorrectly):
```python
>>> m4 = memoryview(data)
>>> t4 = torch.as_tensor(m4)  # A copy of memoryview buffer is made!!!
>>> t4[3] = 104
>>> data
array([101, 102, 103,   4,   5])

```
but wrapping with NumPy ndarray provides a workaround:
```python
>>> t4 = torch.as_tensor(numpy.frombuffer(m4, dtype=m4.format))
>>> t4[3] = 104
>>> data
array([101, 102, 103, 104,   5])
```

PyTorch Tensor object can be used for wrapping arbitrary objects that implement the CUDA Array Interface:
```python
>>> cuda_data = torch.tensor([1, 2, 3, 4, 5], device='cuda')
>>> class A5:
...     __cuda_array_interface__ = cuda_data.__cuda_array_interface__
... 
>>> t5 = torch.as_tensor(A5(), device='cuda')  # device must be specified explicitly
>>> t5[4] = 1005
>>> cuda_data
tensor([   1,    2,    3,    4, 1005], device='cuda:0')
```

## Recommendations

1. Implement `torch.Tensor.__array_interface__` and `torch.Tensor.__array_struct__` attributes to fully support the CPU Array Interfaced.
2. `torch.as_tensor(obj)` should succeed when `obj` implements the CPU Array Interface but is not NumPy ndarray nor PyTorch Tensor object.
3. `torch.as_tensor(obj)` should use `device='cuda'` by default when `obj` implements the CUDA Array Interface. Currently, a CPU copy of a CUDA data buffer is returned from `torch.as_tensor(obj)` while it would be more natural to return a CUDA view of the CUDA data buffer, IMHO.
4. `torch.as_tensor(buf)` should return a view of data buffer when `buf` is `memoryview` object. Currently, a copy of data buffer is made.

# The current usage of array interfaces in different Python libraries - an estimate.

Many Python libraries have adopted the above mentioned array
interfaces. We do not attempt to compose a complete list of such
libraries here.  Instead, to get a rough idea of the explicit usage of
array interfaces, we use GitHub code search tool and report the code
hits for relevant search patterns as shown below (all queries were executed on March 5, 2021).

The search results about using/exposing array interfaces in Python codes:
```
extension:.py "__array__"                              102,931 hits
extension:.py "__array_interface__"                     50,970 hits
extension:.py "__array_struct__"                         9,478 hits
extension:.py "__cuda_array_interface__"                   424 hits
```

as well as in C/C++ codes:
```
extension:.c extension:.cpp "__array_struct__"           1,530 hits
extension:.c extension:.cpp "__array_interface__"        1,202 hits
extension:.c extension:.cpp "__cuda_array_interface__"      91 hits
extension:.c extension:.cpp "__array__"                  1,574 hits (lots of unrelated hits)
```

The search results about exposing array interfaces in Python codes:
```
extension:.py "def __array__("                          57,445 hits
extension:.py "def __array_interface__("                11,097 hits
extension:.py "def __cuda_array_interface__("              146 hits
extension:.py "def __array_struct__("                       19 hits
```

The search results for some popular Python methods, given here for
reference purposes only:
```
extension:.py "def __init__("                       33,653,170 hits
extension:.py "def __getitem__("                     2,185,139 hits
extension:.py "def __len__("                         1,802,131 hits
```

Clearly, the most used array interface details are `__array__` and
`__array_interface__`.

Currently, PyTorch implements hooks for
`__array__` and `__cuda_array_interface__` but not for
`__array_interface__` nor `__array_struct__`, although, workarounds exists
when using NumPy ndarray as intermediate wrapper of data buffers (see above).
