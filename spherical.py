"""Icosphere mesh, its graph Laplacian eigenbasis, and shape-to-sphere projection."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def icosphere(subdiv=3):
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
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    n = len(v)
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    return sparse.diags(np.asarray(a.sum(1)).ravel()) - a


def basis(subdiv=3, k=64):
    v, f = icosphere(subdiv)
    w, u = eigsh(laplacian(v, f).astype(np.float64), k=k, sigma=-1e-6, which='LM')
    o = np.argsort(w)
    return v, w[o], u[:, o]


def support(verts, dirs):
    x = verts - verts.mean(0, keepdims=True)
    r = np.linalg.norm(x, axis=1).max()
    if r > 0:
        x = x / r
    return (x @ dirs.T).max(0)


def read_off(path):
    with open(path, 'r', errors='ignore') as fh:
        txt = fh.read().split()
    if txt[0] != 'OFF':
        txt = ['OFF', txt[0][3:]] + txt[1:]
    nv = int(txt[1])
    vals = np.array(txt[4:4 + 3 * nv], dtype=np.float64)
    return vals.reshape(nv, 3)


def test_basis(subdiv=2, k=30):
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
