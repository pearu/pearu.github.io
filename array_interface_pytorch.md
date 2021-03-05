# Overview of array and buffer protocols in Python

## CPU Array Interface

[Array Interface (Version
3)](https://numpy.org/doc/stable/reference/arrays.interface.html)
defines a protocol for objects to re-use each other's data buffers.
It was created in 2005 within the [NumPy](https://numpy.org/) project for CPU array-like
objects. The implementation of the array interface is defined by the
existence of the following attributes/methofs:

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
In Python, the extension types data buffers can be accessed using `memoryview` object.

# The current usage of array interfaces

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

Clearly, the most popular array interface details are `__array__` and
`__array_interface__`.

Currently, PyTorch implements hooks for
`__array__` and `__cuda_array_interface__` but not for
`__array_interface__` nor `__array_struct__`.

# Using array interfaces in the context of NumPy arrays

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
>>> class A1:
...     def __init__(self, data):
...         self._data = numpy.array(data)
...     def __array__(self):
...         return self._data
... 
>>> a = A1([1, 2, 3, 4])
>>> arr = numpy.asarray(a)
>>> arr[0] = 99
>>> a._data  # notice the shared memory efect
array([99,  2,  3,  4])
```

# Using array interfaces in the context of PyTorch tensors

