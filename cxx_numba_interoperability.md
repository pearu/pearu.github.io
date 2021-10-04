# C++ and numba interoperability

The aim of this document is to explore the problem of calling C++ functions from numba njit-ed functions.

Consider the following example:

```c++int
// File: foo.hpp

int foo(int a);

struct FooStruct {
  int a;
  int b;
};

class FooCls {
  FooCls(int a): a_(a) {}
  int get_a() {return a};
  private:
    int a_;
};

// File: foo.cpp

#include "foo.hpp"

int foo(int a) {
  return a + 123;
}
```
