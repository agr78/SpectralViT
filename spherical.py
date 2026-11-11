"""Icosphere mesh, its graph Laplacian eigenbasis, and shape-to-sphere projection.

The Laplacian basis of sec:tp is only distinguishable from the DCT on a domain that is not
a regular grid: on a lattice with Neumann boundaries the graph Laplacian eigenvectors are
the DCT basis, so a grid experiment cannot separate the two. A sphere has no separable
cosine basis, so the Laplacian is the natural choice there and the property becomes
testable.

Shapes are reduced to a signal on one shared icosphere by their support function,
h(d) = max_v <v, d>, evaluated at each sphere vertex direction. This is vectorized dot
products rather than ray casting, needs no mesh library, and -- the point -- gives every
shape a signal on the SAME graph, so one Laplacian serves all of them and coefficient j
means the same thing for every object. Per-shape meshes would give per-shape eigenvectors
with arbitrary sign and unstable ordering, which is the correspondence failure of sec:corr
in its worst form.

Correctness of the basis is checkable without any data. On a continuous sphere the
Laplace-Beltrami eigenvalues are l(l+1) with multiplicity 2l+1, so the modes up to degree l
number (l+1)^2. An icosphere has icosahedral rather than full rotational symmetry, which
splits the higher multiplets, but the cumulative counts 1, 4, 9, 16 survive; test_basis()
asserts those along with orthonormality and a constant lowest mode.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def icosphere(subdiv=3):
    """Vertices and faces of a subdivided icosahedron, vertices on the unit sphere."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        mid = {}
        nf = []

        def midpoint(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                m = v[a] + v[b]
                mid[k] = len(v) + len(new)
                new.append(m / np.linalg.norm(m))
            return mid[k]

        new = []
        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.vstack([v, np.array(new)])
        f = np.array(nf)
    return v, f


def laplacian(v, f):
    """Combinatorial graph Laplacian of the mesh's edge set, as a sparse matrix."""
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    n = len(v)
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    return sparse.diags(np.asarray(a.sum(1)).ravel()) - a


def basis(subdiv=3, k=64):
    """The k lowest graph Laplacian eigenvectors of a shared icosphere, as (V, k).

    Ordered by eigenvalue, which is the Laplacian's statistical hierarchy: a low eigenvalue
    mode varies slowly across the mesh, so truncating discards what oscillates fastest over
    the connectivity rather than over any pixel lattice.
    """
    v, f = icosphere(subdiv)
    w, u = eigsh(laplacian(v, f).astype(np.float64), k=k, sigma=-1e-6, which='LM')
    o = np.argsort(w)
    return v, w[o], u[:, o]


def support(verts, dirs):
    """Support function h(d) = max_v <v, d> of one shape at each direction.

    Scale is removed by normalizing each shape to unit radius first, so the signal encodes
    form rather than size.
    """
    x = verts - verts.mean(0, keepdims=True)
    r = np.linalg.norm(x, axis=1).max()
    if r > 0:
        x = x / r
    return (x @ dirs.T).max(0)


def read_off(path):
    """Vertices of an OFF mesh. Faces are not needed for the support function."""
    with open(path, 'r', errors='ignore') as fh:
        txt = fh.read().split()
    # Some ModelNet files write "OFF" and the counts on one line, e.g. "OFF6 8 12".
    if txt[0] != 'OFF':
        txt = ['OFF', txt[0][3:]] + txt[1:]
    nv = int(txt[1])
    vals = np.array(txt[4:4 + 3 * nv], dtype=np.float64)
    return vals.reshape(nv, 3)


def test_basis(subdiv=2, k=30):
    """The modes must organize by spherical harmonic degree.

    On a continuous sphere the eigenvalues are l(l+1) with multiplicity 2l+1, so the modes
    up to degree l number (l+1)^2: 1, 4, 9, 16. An icosphere has icosahedral rather than
    full rotational symmetry, which splits the higher multiplets -- degree 3's seven modes
    appear as 3+4, since the icosahedral group's irreps have dimensions 1, 3, 3, 4 and 5 --
    so the degeneracies are not preserved but the cumulative counts are. That is the
    property worth asserting: truncating the basis keeps whole harmonic degrees.
    """
    _, w, u = basis(subdiv, k)
    assert abs(w[0]) < 1e-8, f'first eigenvalue should be 0, got {w[0]}'
    groups, cur = [], [w[1]]
    for x in w[2:]:
        if x - cur[-1] < 0.05 * max(cur[-1], 1e-9):
            cur.append(x)
        else:
            groups.append(len(cur)); cur = [x]
    groups.append(len(cur))
    cum = np.cumsum([1] + groups)
    for sq in (1, 4, 9, 16):
        assert sq in cum, f'degree boundary {sq} missing from cumulative counts {cum}'
    assert np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-6), 'basis not orthonormal'
    assert np.std(u[:, 0]) < 1e-8, 'first mode should be constant'
    return groups[:5], cum[:6].tolist()


if __name__ == '__main__':
    print('multiplicities:', test_basis())
    v, w, u = basis(3, 64)
    print(f'icosphere subdiv 3: {len(v)} vertices, {u.shape[1]} modes, '
          f'eigenvalues {w[0]:.3f} to {w[-1]:.3f}')
