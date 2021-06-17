# Author: Pearu Peterson
# Created: June 2021


import torch
import math
import numpy

def distribute_column_indices(n_rows, n_cols, nnz, backend='numpy'):
    """Return a 1D integer tensor of counts such that

    - len(counts) == n_rows
    - 0 <= counts.min()
    - counts.max() <= n_cols
    - counts.sum() == nnz
    - counts[-1] - counts[0] is as large as possible value
    - the distribution of different count values is as uniform as
      possible

    Application
    -----------
 
    cumsum(cat([0], distribute_column_indices(n_rows, n_cols, nnz))
    gives a valid crow indices of the CSR tensor format.

    Description of the algoritm
    ---------------------------

    The counts can be represented graphically as the following
    histogram:

      ^ number of columns per row
      |
      |

        *   *   *   *   ###
       **  **  **  **+ o###
      *** *** *** ***+oo###
      @@@@@@@@@@@@@@@@@@###
      @@@@@@@@@@@@@@@@@@###      --> row indices

    where regions are named and marked as follows

    - no counts                    : space
    - final correction             : +
    - an incomplete sawtooth       : o
    - a sequence of full sawteeth  : *
    - lower rectangle              : @
    - right rectangle              : #

    The sizes of these regions depend on the input parameter and are
    found such that the following invariants hold:

    - the number of all non-space symbols is equal to nnz
    - the width of the histogram is equal to n_rows
    - the height of the histogram is equal to n_cols

    For very small nnz values, only the incomplete sawtooth is
    generated. For increasing nnz and when n_rows < n_cols, the lower
    rectangle is introduced. Further increasing of nnz leads to
    generation of full sawteeth followed by the incomplete
    sawtooth. For larger values of nnz, the right rectangle will
    emerge which will decrease the width of the lower rectangle as
    well as the sequence of sawteeth. Finally, for very large nnz
    values, the right rectangle will squash the incomplete sawtooth to
    zero-width so that at nnz == n_rows * n_cols the histogram
    contains only the right rectangle. During the whole process, the
    first position of the incomplete rectangle, denoted as the final
    correction, is most volatile to increments of nnz. Notice that the
    lower rectangle and the sequence of full sawteeth never appear in
    the same histogram.

    Performance notes
    -----------------

    Computing
      counts = distribute_column_indices(...)
    takes about the same time as computing the tuple
      (counts.sum(), counts.max(), counts.min())

    Using numpy backend is about twice faster than torch backend.

    Using python backend is about 1.5 times faster than numpy backend
    for small nnz sizes but 4 or more times slower for large nnz
    because of explicit python for loop.

    For maximum performance, distribute_column_indices should be
    implemented in C/C++.

    These performance results are obtained by calling
    distribute_column_indices for all allowed nnz values. For specific
    nnz values, the performance results may vary creatly.
    """
    assert n_rows > 0 and n_cols > 0 and nnz >= 0 and nnz <= n_rows * n_cols

    if backend == 'torch':
        index_dtype = torch.int32
        counts = torch.zeros(n_rows, dtype=index_dtype)
    elif backend == 'numpy':
        index_dtype = numpy.int32
        counts = numpy.zeros(n_rows, dtype=index_dtype)
    elif backend == 'python':
        counts = [0] * n_rows
    else:
        raise NotImplementedError(backend)
        
    # the number of counts per single full sawtooth:
    M = n_cols * (n_cols + 1) // 2
    # the width of an incomplete sawtooth:
    K = n_rows % (n_cols + 1)
    # the total number of counts in the sequence of sawteeth
    N = M * (n_rows // (n_cols + 1)) + K * (K - 1) // 2

    # Right rectangle:
    if N == 0 or nnz < max(N, n_cols):
        n = 0
    else:
        # Find the width of the right rectangle
        left, n, N = 0, n_rows-1, 0
        while n - left > 1:
            middle = (n + left) // 2
            K = (n_rows - middle) % (n_cols + 1)
            N2 = M * ((n_rows - middle) // (n_cols + 1)) + K * (K - 1) // 2
            if N2 == 0 or middle * n_cols > nnz - max(N2, n_cols):
                n, N = middle, N2
            else:
                left = middle
        # Fill the right rectangle:
        if backend == 'torch':
            counts[-n:].fill_(n_cols)
        elif backend == 'python':
            counts[-n:] = [n_cols] * n
        else:
            counts[-n:] = n_cols

    nnz2 = nnz - n * n_cols
    n_rows2 = n_rows - n

    # Lower rectangle:
    if N == 0 or nnz2 < max(N, n_rows2):
        m = 0
    else:
        # Find the height of lower rectangle
        left, m, M, N = 0, n_cols-1, 0, 0
        while m - left > 1:
            middle = (m + left) // 2
            K = n_rows2 % (n_cols - middle + 1)
            M2 = (n_cols - middle) * (n_cols - middle + 1) // 2
            N2 = M2 * (n_rows2 // (n_cols - middle + 1)) + K * (K - 1) // 2
            if N2 == 0 or middle * n_rows2 > nnz2 - max(N2, n_rows2):
                m, M, N = middle, M2, N2
            else:
                left = middle
        # Fill the lower rectangle:
        if backend == 'torch':
            counts[:n_rows2].fill_(m)
        elif backend == 'python':
            counts[:n_rows2] = [m] * n_rows2
        else:
            counts[:n_rows2] = m

    n_cols2 = n_cols - m
    nnz2 = nnz2 - m * n_rows2

    if N == 0:
        # no sawteeth
        counts[0] = nnz2
    else:
        offset = (nnz2 // M) * (n_cols2 + 1)
        k = math.isqrt(2 * (nnz2 % M))
        if k * (k + 1) > 2 * (nnz2 % M):
            k -= 1
        #correction = nnz2 - k * (k + 1) // 2 - (nnz2 // M) * M
        correction = nnz2 % M - k * (k + 1) // 2
        if backend == 'torch':
            tmp = torch.arange(max(offset, k + 1), dtype=counts.dtype)
        elif backend == 'numpy':
            tmp = numpy.arange(max(offset, k + 1), dtype=counts.dtype)
        elif backend == 'python':
            pass
        else:
            raise NotImplementedError(backend)
        if offset:
            # Fill the sequence of full sawteeth:
            if backend == 'python':
                for i in range(offset):
                    counts[i] = i % (n_cols2 + 1)
            else:
                counts[:offset] = tmp[:offset] % (n_cols2 + 1)
            # Since the height of the full sawteeth is n_cols, the
            # lower rectangle must have zero-height.
        if correction:
            # Final correction to the incomplete sawtooth
            if backend != 'python':
                tmp[0] += correction
        if k:
            # The incomplete sawtooth added on top of the bottom
            # rectangle:
            if m:
                if backend == 'python':
                    for i in range(k + 1):
                        counts[offset + i] += i
                    counts[offset] += correction
                else:
                    counts[offset:offset+k+1] += tmp[:k + 1]
            else:
                if backend == 'python':
                    for i in range(k + 1):
                        counts[offset + i] = i
                    counts[offset] += correction
                else:
                    counts[offset:offset+k+1] = tmp[:k + 1]

    if 1:
        # Check the invariants (it will increase the computation time by 2x or more):
        assert sum(counts) == nnz, (counts, nnz)
        assert min(counts) >= 0
        assert max(counts) <= n_cols

    return counts


def animate(n_rows, n_cols, label='new', fmt='gif', fps=15, enable_random=True):

    import torch
    import torch.testing._internal.common_utils
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    import random

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ax.set_xlim(-0.5, n_rows-0.5)
    ax.set_ylim(-0.5, n_cols-0.5)
    ax.set_xlabel('Row indices')
    ax.set_ylabel('Column indices')

    row_indices = list(range(n_rows))
    col_indices = list(range(n_cols))

    def make_image(nnz):
        image = np.zeros((n_rows, n_cols), dtype=int)

        if label == 'pytorch':
            t = torch.testing._internal.common_utils.TestCase()
            b = t.genSparseCSRTensor((n_rows, n_cols), nnz,
                                     device='cpu',
                                     dtype=torch.float32,
                                     index_dtype=torch.int32)
            if not enable_random:
                b.col_indices()[:] = torch.tensor(sorted(b.col_indices()), dtype=b.col_indices().dtype)
            b.values().fill_(1)
            coo = b.to_dense().to_sparse().coalesce()

            for i, j in coo.indices().T:
                image[i, j] = 255
        else:
            counts = distribute_column_indices(n_rows, n_cols, nnz, backend=('torch', 'numpy', 'python')[1])
            if enable_random:
                random.shuffle(row_indices)
            for i, c in enumerate(counts):
                if enable_random:
                    random.shuffle(col_indices)
                image[row_indices[i], np.array(col_indices)[:c]] = 255

        assert len(image.sum(axis=1)) == n_rows

        counts = image.sum(axis=1) // 255
        actual_nnz = image.sum() // 255

        ax.set_title(f'NNZ={nnz} (actual:{actual_nnz})\nmin/max number per row={counts.min()}/{counts.max()}, n_cols={n_cols}')

        return image.T

    nnz0 = n_cols * n_rows // 2
    im = plt.imshow(make_image(nnz0), animated=True)

    def animation_frame(nnz):
        print(nnz)
        im.set_array(make_image(nnz))
        return im,

    animation = FuncAnimation(fig, func=animation_frame, frames=[nnz0] + list(range(1, n_rows * n_cols + 1)), interval=10)

    if enable_random:
        label2 = label + '_random'
    else:
        label2 = label + '_norandom'
    
    if fmt=='mp4':
        fn = f'distribute_column_indices_{n_rows}x{n_cols}_{label2}.mp4'
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={'artist': 'pearu'}, bitrate=1800)
        animation.save(fn, writer)
    elif fmt=='gif':
        fn = f'distribute_column_indices_{n_rows}x{n_cols}_{label2}.gif'
        animation.save(fn, dpi=80, writer='imagemagick', fps=fps)


if __name__ == "__main__":

    #animate(50, 15)
    if 0:
        animate(200, 60, label='pytorch')
        animate(200, 60, label='new')
    elif 0:
        animate(50, 15, label='pytorch')
        animate(50, 15, label='new')
    elif 1:
        animate(17, 5, label='pytorch', fps=5)
        animate(17, 5, label='new', fps=5)
        animate(17, 5, label='pytorch', fps=5, enable_random=False)
        animate(17, 5, label='new', fps=5, enable_random=False)

    n_rows, n_cols = 1000, 1000
    for nnz in range(0, n_rows*n_cols+1):
        break
        counts = distribute_column_indices(n_rows, n_cols, nnz, backend=('torch', 'numpy', 'python')[2])

    for n_cols in range(1, 30):
        break
        for n_rows in range(1, 30):
            for nnz in range(0, n_rows*n_cols+1):
                counts = distribute_column_indices(n_rows, n_cols, nnz,
                                                   backend=('torch', 'numpy', 'python')[2])
                print(f'{counts=} {nnz=}')

    
