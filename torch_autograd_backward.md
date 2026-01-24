# Implementing PyTorch autograd backward methods for tensor operations

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2026-01-24      |

The aim of this blog post is to provide a practical technique for
deriving implementations of backward methods to tensor operations. The
technique is demonstrated on a number of examples including
element-wise operations, matrix operations, reductions,
normalizations, and loss functions. All backward expressions in
examples are numerically verified for correctness.

[Click here to view the document with rendered math.](https://github.com/pearu/pearu.github.io/blob/main/torch_autograd_backward.md)

## Theory and torch.autograd API

Consider a functional $l = L(f)$ that is a function on the output of a
tensor operation $F$:
```math
l = L(F(A))
```
where $A$ is an $N$-dimensional tensor and $F(A)$ is an
$M$-dimensional tensor. Let $i$ be an $N$-tuple. Then $A_i$ is an
element of the tensor $A$ with the index $i$.

Let's find
```math
\frac{\partial l}{\partial A_i} = \sum_j \frac{\partial L}{\partial f_j} * \frac{\partial F(A)_j}{\partial A_i}
```
where $j$ is an $M$-tuple denoting the index of $M$-dimensional tensor elements and $*$ denotes scalar multiplication.
We'll denote $G = \partial L/\partial f$ which is an $M$-dimensional tensor.

For simplicity of using calculus, in the following we'll use 1-based
indices. It is not going to be a problem in most cases when writting
expressions using array operations rather than element-wise
operations.

When defining a
[torch.autograd.Function](https://docs.pytorch.org/docs/stable/autograd.html#function)
class that represents a tensor operation $F$ as defined above, one
needs to implement `forward` and `backward` static methods:
```python
class MyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A):
        # returns F(A), a M-dimensional tensor

    @staticmethod
    def backward(ctx, G):
        # if A.required_grad:
        #     return sum_j(G_j * d(F(A)_j)/d(A_i)), a N-dimensional tensor
        # else:
        #     return None
```
where `A` is passed from `forward` method to `backward` method
using [ctx.save_for_backward/ctx.saved_tensors](https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html) API.

### Hint

For tensor operations with multiple arguments and/or multiple return values, say,
```python
@staticmethod
def forward(ctx, *inputs):
    # inputs is a tuple of forward arguments
    # outputs is a tuple of forward return values
    return outputs

@staticmethod
def backward(ctx, *backward_inputs):
    # backward_inputs is a tuple of backward arguments
    # backward_outputs is a tuple of backward return values
    return backward_outputs
```
then the following invariants must hold:
```python
len(inputs) == len(backward_outputs)
len(backward_inputs) == len(outputs)
inputs[i].requires_grad == backward_outputs[i] is not None
backward_outputs[i].shape == inputs[i].shape
outputs[i].shape == backward_inputs[i].shape
```

To verify that `backward` method has correct implementation, use
[torch.autograd.gradcheck](https://docs.pytorch.org/docs/stable/autograd.html#module-torch.autograd.gradcheck)
tool.


## Element-wise operations

Let $F$ be an element-wise operation with a derivative $F'$. Then $M = N$, $F(A)_j = F(A_j)$, and
```math
\frac{\partial F(A)_j}{\partial A_i} = F'(A_j) * \frac{\partial A_j}{\partial A_i} = F'(A_j) * \delta_{j,i}
```
where $\delta_{j,i}$ is $1$ when $j$ and $i$ are equal, otherwise $0$.

As a result,
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = \sum_j G_j * F'(A_j) * \delta_{j,i}  = G_i * F'(A_i),
```
that is
```python
def backward(ctx, G):
    # return G * F'(A)
```

For instance, if $F$ is the sine function, then we'll have
```python
def forward(ctx, A):
      ctx.save_fow_backward(A)
      return torch.sin(A)

def backward(ctx, G):
      A, = ctx.saved_tensors
      return G * torch.cos(A)
```
These methods must be decorated with `@staticmethod` that I skipped
here for simplicity of presentation.

## Matric operations

### Transpose: `F(A) = A.T`

We have $M = N = 2$. Let's denote $A_i\equiv A[i_1, i_2]$ as the element of tensor $A$ with an index $i \equiv (i_1, i_2)$, then
```math
F(A)_j = F(A)[j_1, j_2] = A[j_2, j_1]
```
```math
\frac{\partial F(A)_j}{\partial A_i} = \frac{\partial A[j_2, j_1]}{\partial A[i_1, i_2]} = \delta_{i_1,j_2} * \delta_{i_2,j_1}
```
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = \sum_j G[j_1, j_2] * \delta_{i_1,j_2} * \delta_{i_2,j_1} = G[i_2, i_1]
```
that is,
```python
def backward(ctx, G):
     return G.T
```

### Matrix multiplication from left: `F(A) = A @ B`

We have $M = N = 2$, then
```math
F(A)_j = F(A)[j_1, j_2] = \sum_k A[j_1, k] * B[k, j_2]
```
```math
\frac{\partial F(A)_j}{\partial A_i} = \frac{\partial \sum_k A[j_1, k] * B[k, j_2]}{\partial A[i_1, i_2]}\\
                 = \sum_k \frac{\partial A[j_1, k]}{\partial A[i_1, i_2]} * B[k, j_2] \\
                 = \sum_k \delta_{j_1,i_1} * \delta_{k,i_2} * B[k, j_2] \\
                 = \delta_{j_1,i_1} * B[i_2, j_2]
```
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = \sum_j G[j_1, j_2] * \delta_{j_1,i_1} * B[i_2, j_2]
                              = \sum_{j_2} G[i_1, j_2] * B[i_2, j_2]
```
that is,
```python
def backward(ctx, G):
     return G @ B.T
```

### Matrix multiplication from right: `F(A) = B @ A`

We have
```math
F(A)_j = F(A)[j_1, j_2] = \sum_k B[j_1, k] * A[k, j_2]
```
```math
\frac{\partial F(A)_j}{\partial A_i} = \frac{\partial \sum_k B[j_1, k] * A[k, j_2]}{\partial A[i_1, i_2]} \\
                 = \sum_k B[j_1, k] * \frac{\partial A[k, j_2]}{\partial A[i_1, i_2]} \\
                 = \sum_k B[j_1, k] * \delta_{k,i_1} * \delta_{j_2,i_2}
                 = B[j_1, i_1] * \delta_{j_2,i_2}
```
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = \sum_j G[j_1, j_2] * B[j_1, i_1] * \delta_{j_2,i_2}\\
                              = \sum_{j_1} G[j_1, i_2] * B[j_1, i_1]
```
that is,
```python
def backward(ctx, G):
     return G.T @ B
```

## Reduction operations

### Sum along specific dimension: `sum(A, dim=d, keepdim=keepdim)`

We have $M = N - 1$, then
```math
F(A)_j = \sum_k A[j_1,\ldots,j_{d-1}, k, j_{d+1}, \ldots, j_{N-1}]
```
```math
\frac{\partial F(A)_j}{\partial A_i} = \frac{\partial \sum_k A[j_1,\ldots,j_{d-1}, k, j_{d+1}, \ldots, j_{N-1}]}{\partial A[i_1,\ldots,i_{d-1}, i_d, i_{d+1}, \ldots, i_{N}]}
```
```math
                                     = \sum_k \delta_{j_1, i_1} * \cdots * \delta_{j_{d-1}, i_{d-1}} * \delta_{k, i_{d}} * \delta_{j_d, i_{d+1}} * \cdots *\delta_{j_{N-1}, i_{N}} = \delta_{j_1, i_1} * \cdots * \delta_{j_{d-1}, i_{d-1}} * \delta_{j_{d}, i_{d+1}} * \cdots *\delta_{j_{N-1}, i_{N}}
```
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = \sum_j G[j_1,\ldots,j_{d-1},j_{d},\ldots,j_{N-1}] * \delta_{j_1, i_1} * \cdots * \delta_{j_{d-1}, i_{d-1}} * \delta_{j_{d}, i_{d+1}} * \cdots *\delta_{j_{N-1}, i_{N}}
```
```math
= G[i_1,\ldots,i_{d-1},i_{d+1},\ldots,i_{N}] \qquad \forall i_{d}
```
that is,
```python
def backward(ctx, G):
     if keepdim:
         return G.expand(A.shape)
     return G.unsqueeze(d).expand(A.shape)
```

### Max along specific dimension: `max(A, dim=d, keepdim=keepdim)`


We have
```math
F(A)_j = \max_k A[j_1,\ldots,j_{d-1}, k, j_{d}, \ldots, j_{N-1}] = A[j_1,\ldots,j_{d-1}, \mathrm{arg\,max}(F(A)_j), j_{d}, \ldots, j_{N-1}]
```
```math
\frac{\partial F(A)_j}{\partial A_i} = \frac{\partial \max_k A[j_1,\ldots,j_{d-1}, k, j_{d}, \ldots, j_{N-2}]}{\partial A[i_1,\ldots,i_{d-1}, i_d, i_{d+1}, \ldots, i_{N}]}
```
```math
                                     = \delta_{j_1, i_1} * \cdots * \delta_{j_{d-1}, i_{d-1}} * \delta_{\mathrm{arg\,max}(F(A)_j), i_{d}} * \delta_{j_d, i_{d+1}} * \cdots *\delta_{j_{N-1}, i_{N}}
```
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = \sum_j G[j_1,\ldots,j_{d-1},j_{d},\ldots,j_{N-1}] * \delta_{j_1, i_1} * \cdots * \delta_{j_{d-1}, i_{d-1}} * \delta_{\mathrm{arg\,max}(F(A)_j), i_{d}} * \cdots *\delta_{j_{N-1}, i_{N}}
```
```math
= G[i_1,\ldots,i_{d-1},i_{d+1},\ldots,i_{N}] * \delta_{\mathrm{arg\,max}(F(A)_j), i_{d}}
```
that is,
```python
def backward(ctx, G):
     # for best performance, compute mask in forward
     mask = torch.zeros_like(A)
     _, indices = A.max(dim=d, keepdim=True)
     mask.scatter_(d, indices, 1)
     # expand(A.shape) is not required as mul broadcasts G to proper shape
     if keepdim:
         return G * mask
     return G.unsqueeze(d) * mask
```

## Normalizations

### Softmax along specific dimension: `softmax(A, dim=d)`

We have $N=M$ and
```math
F(A)_j = \frac{\exp(A[j_1,\ldots,j_{d-1}, j_{d}, j_{d+1}, \ldots, j_{N}])}{\sum_k \exp(A[j_1,\ldots,j_{d-1}, k, j_{d+1}, \ldots, j_{N}])}
```
```math
\frac{\partial F(A)_j}{\partial A_i} = \frac{\partial \frac{\exp(A[j_1,\ldots,j_{d-1}, j_{d}, j_{d+1}, \ldots, j_{N}])}{\sum_k \exp(A[j_1,\ldots,j_{d-1}, k, j_{d+1}, \ldots, j_{N}])}}{\partial A[i_1,\ldots,i_{d-1}, i_d, i_{d+1}, \ldots, i_{N}]}
```
```math
= \delta_{j_1, i_1} * \cdots * \delta_{j_{d-1}, i_{d-1}} * \frac{\partial \frac{\exp(A[\ldots,j_{d},\ldots])}{\sum_k \exp(A[\ldots, k, \ldots])}}{\partial A[\ldots, i_d, \ldots]}  *  \delta_{j_{d+1}, i_{d+1}} * \dots * \delta_{j_{N}, i_{N}}
```
Let's define $a[n] \equiv A[j_1,\ldots,j_{d-1}, n, j_{d+1}, \ldots, j_{N}]$ and find
```math
\frac{\partial \frac{\exp(a[j_{d}])}{\sum_k \exp(a[k])}}{\partial a[i_d]}
= \frac{\exp(a[j_{d}])}{\sum_k \exp(a[k])} \delta_{j_d, i_d}
- \frac{\exp(a[j_{d}])}{\sum_k \exp(2 * a[k])} \sum_{k'} \exp(a[k']) \delta_{k', i_d}
= \frac{\exp(a[j_{d}])}{\sum_k \exp(a[k])} * \left(
  \delta_{j_d, i_d} -  \frac{\exp(a[i_{d}])}{\sum_k \exp(a[k])}
  \right)
```
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = \sum_j G[j_1,\ldots,j_{d-1},j_{d},j_{d+1},\ldots,j_{N}] *
\delta_{j_1, i_1} * \cdots * \delta_{j_{d-1}, i_{d-1}} * \frac{\exp(a[j_{d}])}{\sum_k \exp(a[k])} * \left(
  \delta_{j_d, i_d} -  \frac{\exp(a[i_{d}])}{\sum_k \exp(a[k])}
  \right)  *  \delta_{j_{d+1}, i_{d+1}} * \dots * \delta_{j_{N}, i_{N}}
```
```math
= \sum_{j_d} G[i_1,\ldots,i_{d-1},j_{d},i_{d+1},\ldots,i_{N}] *
\frac{\exp(a[j_{d}])}{\sum_k \exp(a[k])} * \left(
  \delta_{j_d, i_d} -  \frac{\exp(a[i_{d}])}{\sum_k \exp(a[k])}
  \right)
```
```math
= \left (G[i_1,\ldots,i_{d-1},i_{d},i_{d+1},\ldots,i_{N}] 
  -
  \sum_{j_d} G[i_1,\ldots,i_{d-1},j_{d},i_{d+1},\ldots,i_{N}] * \frac{\exp(a[j_{d}])}{\sum_k \exp(a[k])}
  \right) * \frac{\exp(a[i_{d}])}{\sum_k \exp(a[k])}
```
that is,
```python
def backward(ctx, G):
    S = softmax(A, dim=d)
    return (G - (G * S).sum(dim=d, keepdim=True)) * S
```

## Loss functions

### Negative log likelihood loss: `nll_loss(A, T, weight=W, ignore_index=ii, reduction='mean')`

We'll consider the case where `T` contains class indices. Hence, $N=2$, $M=0$ if `reduction != 'none'`, otherwise $M=1$.

If `reduction == 'mean'` then
```math
F(A) =  \frac{\sum_n -W[T[n]] * (1-\delta_{T[n], ii}) * A[n, T[n]]}{\sum_{n} W[T[n]] * (1-\delta_{T[n], ii})}
```
If `reduction == 'sum'` then
```math
F(A) = \sum_n -W[T[n]] * (1-\delta_{T[n], ii}) * A[n, T[n]]
```
If `reduction == 'none'` then
```math
F(A)_j = -W[T[j]] * (1-\delta_{T[j], ii}) * A[j, T[j]]
```

For the sum and mean reduction cases, let's find
```math
\frac{\partial F(A)}{\partial A[i_1, i_2]}
= \sum_n -W[T[n]] * (1 - \delta_{T[n], ii}) * \delta_{i_1, n} * \delta_{i_2, T[n]}
= -W[T[i_1]] * (1 - \delta_{T[i_1], ii}) * \delta_{i_2, T[i_1]}
```
```math
\sum_j G_j * \frac{\partial F(A)_j}{\partial A_i}  = -G * W[T[i_1]] * (1 - \delta_{T[i_1], ii}) * \delta_{T[i_1], i_2}
```
that is,
```python
def backward(ctx, G):
    wmask = torch.zeros_like(A).scatter_(1, T.unsqueeze(1), W.index_select(0, T).unsqueeze(1))
    if ii >= 0:
        wmask.select(1, ii).zero_()
    if reduction == "mean":
        wmask /= W.index_select(0, T).sum()
    return -G * wmask
```

### Linear cross-entropy: `linear_cross_entropy(A, L, T, bias=b, weight=W, ignore_index=ii, reduction='mean', label_smoothing=0.0)`

We'll first consider the case where `T` contains class indices. Hence,
$N=2$, $M=0$ if `reduction != 'none'`, otherwise $M=1$.

Let's define
```math
X[n_1, n_2] = \sum_{k} A[n_1, k] * L[n_2, k] + b[n_1]
```
We have
```math
\frac{\partial X[n_1, n_2]}{\partial A[i_1, i_2]} = \sum_{k} \delta_{n_1,i_1}*\delta_{k, i_2} * L[n_2, k] =  \delta_{n_1,i_1} * L[n_2, i_2],
```
```math
\frac{\partial X[n_1, n_2]}{\partial L[i_1, i_2]} = \sum_{k} A[n_1, k] * \delta_{n_2,i_1}*\delta_{k, i_2} =  \delta_{n_2,i_1} * A[n_1, i_2],
```
```math
\frac{\partial X[n_1, n_2]}{\partial b[i_1]} = \delta_{n_1, i_1}, \qquad \forall n_2.
```

In the following, when $ii >= 0$, we'll set `W[ii] = 0` that will eliminate the $(1-\delta_{T[n], ii})$ term in the `nll_loss` function.

If `reduction == 'sum'` then
```math
F(A, L, b) = \sum_n -W[T[n]] * \log \mathrm{softmax}(X, dim=1)_{n, T[n]} 
```
```math
= \sum_n -W[T[n]] * \log \frac{\exp(X[n, T[n]])}{\sum_{n'}\exp(X[n,n'])}
```
```math
= \sum_n -W[T[n]] * \left(X[n, T[n]] - \log\sum_{n'}\exp(X[n,n'])\right)
```

```math
\frac{\partial F(A, L, b)_j}{\partial A_i} =
\sum_n -W[T[n]] * \left(\delta_{n,i_1} * L[T[n], i_2] - \frac{\sum_{n'} \exp(X[n,n']) * \delta_{n,i_1} * L[n', i_2]}{\sum_{n'}\exp(X[n,n'])}\right)
```
```math
= -W[T[i_1]] * \left(L[T[i_1], i_2] - \frac{\sum_{n'} \exp(X[i_1,n']) * L[n', i_2]}{\sum_{n'}\exp(X[i_1,n'])}\right)
```
```math
= -W[T[i_1]] * \left(L[T[i_1], i_2] - \sum_{n'} \mathrm{softmax}(X, dim=1)_{i_1, n'} * L[n', i_2]\right)
```
```math
\frac{\partial F(A, L, b)_j}{\partial L_i} =
\sum_n -W[T[n]] * \left(\delta_{T[n],i_1} * A[n, i_2] - \frac{\sum_{n'} \exp(X[n,n']) * \delta_{n',i_1} * A[n, i_2]}{\sum_{n'}\exp(X[n,n'])}\right)
```
```math
= \sum_n -W[T[n]] * \left(\delta_{T[n],i_1} * A[n, i_2] - \mathrm{softmax}(X, dim=1)_{n,i_1}* A[n, i_2]\right)
```
```math
= \sum_n -W[T[n]] * \left(\delta_{T[n],i_1} - \mathrm{softmax}(X, dim=1)_{n,i_1}* \right) * A[n, i_2]
```
```math
\frac{\partial F(A, L, b)_j}{\partial b_i} 
= \sum_n -W[T[n]] * \left(\delta_{n, i_1} - \frac{\sum_{n'}\exp(X[n, n']) * \delta_{n, i_1}}{\sum_{n''}\exp(X[n, n''])}\right) = 0
```

```math
\sum_j G_j * \frac{\partial F(A, L, b)_j}{\partial A_i} = -G * W[T[i_1]] * \left(L[T[i_1], i_2] - \sum_{n'} \mathrm{softmax}(X, dim=1)_{i_1, n'} * L[n', i_2]\right)
```
```math
\sum_j G_j * \frac{\partial F(A, L, b)_j}{\partial L_i} = -G *
\sum_n W[T[n]] * \left(\delta_{T[n],i_1} - \mathrm{softmax}(X, dim=1)_{n,i_1} \right) * A[n, i_2]
```
```math
\sum_j G_j * \frac{\partial F(A, L, b)_j}{\partial b_i} = 0
```

that is,
```python
def backward(ctx, G):
    if ii >= 0:
        W = W.clone()
        W[ii] = 0
    X = A @ L.T + b
    lS = log_softmax(X, dim=1)
    S = exp(lS)
    w = W.index_select(0, T).unsqueeze(1)
    grad_A = w * L.index_select(0, T) - (w * S) @ L
    Wx = torch.zeros_like(L).scatter_reduce_(0,
                                             T.unsqueeze(1).expand(x.shape),
                                             A * w,
                                             'sum',
                                             include_self=False)
    grad_L =  Wx - (w * S).T @ A
    if reduction == "mean":
        d = W.index_select(0, T).sum()
        grad_A /= d
        grad_L /= d
    return -G * grad_A, -G * grad_L, torch.zeros_like(b)
```


# Conclusion

Hopefully, the provided examples are helpful for starting to implement
the autograd backward methods for tensor operations.