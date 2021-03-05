# Overview

[Array Interface (Version
3)](https://numpy.org/doc/stable/reference/arrays.interface.html)
defines a protocol for objects to re-use each other's data buffers.
It was created in 2005 within the [NumPy](https://numpy.org/) project for CPU array-like
objects. The implementation of the array interface is defined by the
existence of the following attributes:

- `__array_interface__` - a Python dictionary that contains the shape,
  the element type, and optionally, the data buffer address and the
  strides of an array-like object.

- `__array__` - a NumPy ndarray view of an array-like object

- `__array_struct__` - holds a pointer to [PyArrayInterface
  C-structure](https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_struct__).

Numba introduces [CUDA Array Interface (Version 2)](https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html)
for GPU array-like objects. The implementation of the CUDA array
interface is defined by the existence of the attribute

- `__cuda_array_interface__`

that holds the same information about an array-like object as `__array_interface__` except
the data buffer address will point to GPU memory area.

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
