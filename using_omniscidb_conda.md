# Using OmniSciDB from conda environment

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2022-03-30      |

The aim of this blog post is to describe the usage of omniscidb from
conda environment. In the following, we'll exemplify this using
OmniSciDB version 5.10 installed as a conda-forge package. At the
moment of writing this post, the rebranded software --
[https://github.com/heavyai/heavydb](HeavyDB (formely OmniSciDB)) --
is not available via conda-forge yet.

# Prerequisites

As a minimal prerequsite, install
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) to your
system. On Linux, the process is roughly as follows:

```bash
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

and follow the given instructions (if you have not done this before, accepting
all defaults is fine).

Run

```bash
$ eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"
```

to activate the conda ``base`` environment. This will make ``conda``
command line tool available that we'll use below for various tasks.

## Create conda environment for running omniscidb server

To create an environment for running omniscidb, run

```bash
$ conda create -n omniscidb-env -c conda-forge omniscidb
```

that will install the latest ``omniscidb`` conda forge package and all
its prerequisites.

Notice that this will install CPU-only version of the omniscidb software.

To install CUDA-enabled version of the omniscidb software, use

```bash
$ conda create -n omniscidb-cuda-env -c conda-forge omniscidb=*=*cuda
```

Do not install CPU-only and CUDA-enabled omniscidb to the same environment!

## Create conda environment for running omniscidb clients

To create an environment for running omniscidb client programs, run
```bash
$ conda create -n omnisci-user -c conda-forge pyomnisci rbc
```
that will install

- [Pyomnisci](https://github.com/heavyai/pyomnisci) - provides a
  Python package ``pyomnisci`` for interfacing with the OmnisciDB
  server

- [RBC - Remote Backend
  Compiler](https://github.com/xnd-project/rbc/) - provides a Python
  package ``rbc`` for defining and registering user-defined functions
  (UDF/UDTFs) to the OmnisciDB server

and all their prerequisites.

We create two different environments ``omniscidb-env`` and
``omnisci-user`` because running the omniscidb server and interfacing
with the server happen from different processes or even from different
hosts.

# Preparing and running omniscidb server

First, you must activate the omniscidb server conda environment:

```bash
$ conda activate omniscidb-env
```

To check the successful installation of omniscidb server, run

```bash
$ omnisci_server --version
OmniSci Version: 5.10.2-20220321-9c57af4ef5
```

If this works, let's create a DB storage ``mydata`` for the omniscidb server:

```bash
$ mkdir mydata && omnisci_initdb -f mydata
```

Finally, we are ready to run the server:

```bash
$ omnisci_server --data mydata --enable-runtime-udf
```

where we used ``--enable-runtime-udf`` to allow clients such as rbc to
define new SQL operators to the omniscidb server.

# Clients connecting to omnscidb server

Assuming that the omniscidb server is running (see above), let's try
to connect to it using different clients. For the following
instructions, use a different shell prompt from the one that runs the
server. When opening the new shell prompt, you'll may need to
initialize conda bash hook:

```bash
$ eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"
```

(or run, `/path/to/miniconda3/bin/conda init` to include this
functionality to ``.bashrc`` so that each shell prompt will have conda
``base`` environment activated automatically).

## Using omnisql program

The ``omnisql`` program is an interactive SQL interpreter that is
provided by the omniscidb package. Hence, we'll need to activate
``omniscidb-env`` conda environment before we can use ``omnisql``:

```bash
$ conda activate omniscidb-env
$ omnisql -p HyperInteractive -u admin
User admin connected to database omnisci
omnisql> CREATE TABLE example_table (i INT, x DOUBLE[]);
omnisql> INSERT into example_table VALUES (123, ARRAY[1, 2, 3, 4]);
omnisql> INSERT into example_table VALUES (456, ARRAY[5, 6, 7]);
omnisql> SELECT i, x FROM example_table;
i|x
123|{1, 2, 3, 4}
456|{5, 6, 7}
omnisql> ^D
User admin disconnected from database omnisci
$ conda deactivate
```

Here we created a new DB table ``example_table`` that we'll shall use
in the other examples below.

## Using pyomnisci Python package

To access the running omniscidb server from Python, we'll need to
activate the ``omnisci-user`` environment:

```bash
$ conda activate omnisci-user
```

Now, in Python, we can retrieve the data from previously created
example DB table:

```python
>>> from pyomnisci import connect
>>> con = connect(user="admin", password="HyperInteractive", host="localhost", dbname="omnisci")
>>> df = con.select_ipc('SELECT i FROM example_table')
>>> df
     i
0  123
1  456
```

See [pyomnisci
documentation](https://pyomnisci.readthedocs.io/en/latest/?badge=latest)
for more information.


```bash
$ conda deactivate
```

## Using rbc to register new SQL operators to omniscidb server

Assuming that the omniscidb server is running, let's define a UDF in
Python and register it to omniscidb server. First, let's activate the
``omnisci-user`` environment:

```bash
$ conda activate omnisci-user
```

In Python, let's connect to omniscidb server using rbc:
```python
>>> import rbc
>>> omnisci = rbc.omniscidb.RemoteOmnisci(user="admin", password="HyperInteractive", host="localhost", dbname="omnisci")
```

Next, define a UDF that increments its input by the given increment:
```python
>>> @omnisci('T(T, T)', T=['int32', 'int64', 'float32', 'float64'])
... def myincr(i, di):
...     return i + di
... 
```

where we decorated Python function ``myincr`` with ``omnisci`` object
that compiles the Python function into omnisci SQL operator
``MYINCR``. To register newly defined operations to the omniscidb
server, run:

```python
>>> omnisci.register()
```

Now we can use ``MYINCR`` in a SQL query to the server:
```python
>>> _, r = omnisci.sql_execute('SELECT i, MYINCR(i, 3) FROM example_table')
>>> list(r)
[(123, 126), (456, 459)]
```

For testing purposes, rbc supports calling ``omnisci`` decorated
functions on Python data. Of course, the actual execution of the
function is carried out remotely within the omniscidb server:

```python
>>> myincr(2, 3)
OmnisciQueryCapsule('SELECT myincr(CAST(2 AS BIGINT), CAST(3 AS BIGINT))')
>>> myincr(2, 3).execute()
5
```

In the above, ``MYINCR`` represents the so-called scalar UDF that is
applied to DB table rows. We can also define so-called tabular UDFs
(UDTFs) that are applied to DB table columns. To illustrate this, let's create
a Python script containing

```python
# File: test_colincr.py

import rbc
omnisci = rbc.omniscidb.RemoteOmnisci(user="admin", password="HyperInteractive", host="localhost", dbname="omnisci")

@omnisci('int32(TableFunctionManager, Column<T> inp, T di, OutputColumn<T> output)', T=['int32', 'int64', 'float32', 'float64'])
def colincr(mgr, inp, di, out):
    mgr.set_output_row_size(len(inp))  # allocates out buffer
    for i in range(len(inp)):
        out[i] = inp[i] + di
    return len(out)  # must return the length of the out buffer

omnisci.register()

_, r = omnisci.sql_execute('SELECT output FROM TABLE(COLINCR(CURSOR(SELECT i FROM example_table), 3))')

print(list(r))
```

and run it:

```bash
$ python test_colincr.py 
[(126,), (459,)]
```


## Defining omniscidb load-time UDFs (CPU-only)

In the above, we used rbc to create UDFs that are registered as new
SQL operators to running omniscidb server. Such UDFs are called
runtime UDFs and these can be re-defined or removed during the server
runtime. OmnisciDB server supports also defining so-called load-time
UDFs that are implemented in C++ and that are compiled to SQL
operators at the start up of the omniscidb server and these will
persists til the server exits.

To illustrate the usage of C++ UDFs, will need to stop the omniscidb
server and update the ``omniscidb-env`` environment as follows:

```bash
conda activate omniscidb-env
conda install -c conda-forge clangxx clangdev=11 gxx_impl_linux-64=9 gxx_linux-64=9
wget https://raw.githubusercontent.com/conda-forge/omniscidb-feedstock/main/recipe/get_cxx_include_path.sh
source get_cxx_include_path.sh
export CPLUS_INCLUDE_PATH=`get_cxx_include_path`:$CONDA_PREFIX/include/omnisci/QueryEngine/
```

Let's create a C++ file:
```c++
// File: array_sum.cpp

#include "OmniSciTypes.h"

EXTENSION_NOINLINE
double array_sum(Array<double> arr) {
  double result = 0.0;
  for (int64_t i = 0; i < arr.getSize(); i++) {
    if (arr[i] != arr.null_value()) {
      result += arr[i];
    }
  }
  return result;
}
```

that will be used to build a new SQL operator ``ARRAY_SUM`` that
computes the sum of array elements.

Let's re-start the omniscidb server as follows:
```bash
omnisci_server --data mydata --enable-runtime-udf --udf array_sum.cpp
```

An example of using the ``ARRAY_SUM`` operator:

```
$ conda activate omniscidb-env
$ omnisql -p HyperInteractive -u admin
User admin connected to database omnisci
omnisql> SELECT x, array_sum(x) FROM example_table;
x|EXPR$1
{1, 2, 3, 4}|10
{5, 6, 7}|18
omnisql> 
```

## Defining omniscidb load-time UDFs (CUDA-enabled)

Unfortunately, the header file ``OmniSciTypes.h`` shipped with conda
package omniscidb 5.10 is broken for CUDA-enabled omniscidb
servers. To fix it, create a file ``OmniSciTypes.patch`` containing

```
--- OmniSciTypes.h-orig	2022-03-30 19:04:08.885277966 +0300
+++ OmniSciTypes.h	2022-03-30 19:10:28.521666496 +0300
@@ -214,6 +214,11 @@
   DEVICE int32_t getOutputSrid() const { return output_srid; }
 };
 
+#ifdef __CUDACC__
+template <typename T>
+static DEVICE __constant__ T Column_null_value;
+#endif
+
 template <typename T>
 struct Column {
   T* ptr_;        // row data
@@ -224,12 +229,8 @@
 #ifndef __CUDACC__
       throw std::runtime_error("column buffer index is out of range");
 #else
-      static DEVICE T null_value;
-      if constexpr (std::is_same_v<T, TextEncodingDict>) {
-        set_null(null_value.value);
-      } else {
-        set_null(null_value);
-      }
+      auto& null_value = Column_null_value<T>;
+      set_null(null_value);
       return null_value;
 #endif
     }
@@ -316,6 +317,7 @@
   corresponding instances share `this` but have different virtual
   tables for methods.
 */
+#ifndef __CUDACC__
 struct TableFunctionManager {
   static TableFunctionManager* get_singleton() {
     return reinterpret_cast<TableFunctionManager*>(TableFunctionManager_get_singleton());
@@ -342,3 +344,4 @@
 
 #endif
 };
+#endif
```
and run:

```bash
patch $CONDA_PREFIX/include/omnisci/QueryEngine/OmniSciTypes.h < OmniSciTypes.patch
```


Here we assume that NVIDIA driver is functional, that is,

```bash
$ nvidia-smi
```
displays the status of the driver, and one has CUDA installed, that is,
```
$CUDA_HOME/nvvm/libdevice/libdevice.10.bc
```
exists. Update ``CUDA_HOME`` environment variable to meet your system
configuration if needed.

When using CUDA enabled omniscidb server, use the following updates
applied to the previous section:

```bash
conda activate omniscidb-cuda-env
conda install -c conda-forge clangxx clangdev=11 gxx_impl_linux-64=9 gxx_linux-64=9 nvcc_linux-64=11.5
test -f get_cxx_include_path.sh || wget https://raw.githubusercontent.com/conda-forge/omniscidb-feedstock/main/recipe/get_cxx_include_path.sh
source get_cxx_include_path.sh
export CPLUS_INCLUDE_PATH=`get_cxx_include_path`:$CONDA_PREFIX/include/omnisci/QueryEngine/:$CUDA_HOME/include
```

Notice that the CUDA version of the conda package ``nvcc_linux-64``
must match with the CUDA version installed in the ``CUDA_HOME``
directory. Here we use CUDA version 11.5 as an example.

When done, the ``ARRAY_SUM`` example from the previous example would
be executed on a CUDA device.
