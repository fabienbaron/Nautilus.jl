"""Dump reference values from the Python nautilus package into fixtures.h5.

Run this once (and again only if the Python package is updated):

    python test/fixtures/dump_fixtures.py

The Julia suite reads the result to check the paths that are deterministic given their input --
above all the MLP forward pass, where loading scikit-learn's own trained weights separates
"the network evaluates correctly" from "the network trains equivalently". The first must be
exact; only the second is a statistical claim.
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.environ.get(
    "NAUTILUS_PY", os.path.expanduser("~/SOFTWARE/nautilus")))

from nautilus.bounds.basic import (  # noqa: E402
    minimum_volume_enclosing_ellipsoid, Ellipsoid, UnitCubeEllipsoidMixture)
from nautilus.bounds.union import ellipsoids_overlap  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.h5")


def main():
    with h5py.File(OUT, "w") as f:
        # -- MVEE on the hypersphere-boundary fixture ------------------------------------
        n_dim = 10
        p = np.zeros((2 * n_dim, n_dim)) + 0.5
        for i in range(2 * n_dim):
            p[i, i // 2] += 1 if i % 2 else -1
        ct = np.median(p, axis=0)
        np.random.seed(0)
        pts = np.concatenate([p, np.atleast_2d(ct + np.random.random() - 0.5)])
        c, A, A_inv = minimum_volume_enclosing_ellipsoid(pts)
        g = f.create_group("mvee")
        g.create_dataset("points", data=pts)
        g.create_dataset("c", data=c)
        g.create_dataset("A", data=A)

        # -- UnitCubeEllipsoidMixture dimension selection --------------------------------
        rng = np.random.default_rng(42)
        narrow = 0.5 + 0.01 * rng.normal(size=(1000, 2))
        wide = rng.random(size=(1000, 2))
        mix_pts = np.hstack([narrow, wide])
        mix = UnitCubeEllipsoidMixture.compute(mix_pts, rng=np.random.default_rng(0))
        g = f.create_group("mixture")
        g.create_dataset("points", data=mix_pts)
        g.create_dataset("dim_cube", data=mix.dim_cube.astype(np.int8))
        g.attrs["log_v"] = mix.log_v

        # -- ellipsoids_overlap on random pairs ------------------------------------------
        rng = np.random.default_rng(7)
        centers, overlaps = [], []
        base = rng.normal(size=(200, 3))
        base /= np.linalg.norm(base, axis=1)[:, None]
        base *= rng.random(200)[:, None] ** (1 / 3)
        for k in range(60):
            shift = 0.1 * k
            e1 = Ellipsoid.compute(base, rng=np.random.default_rng(0))
            e2 = Ellipsoid.compute(base + shift, rng=np.random.default_rng(0))
            centers.append(shift)
            overlaps.append(bool(ellipsoids_overlap([e1, e2])))
        g = f.create_group("overlap")
        g.create_dataset("base", data=base)
        g.create_dataset("shift", data=np.array(centers))
        g.create_dataset("overlap", data=np.array(overlaps, dtype=np.int8))

        # -- rank-score construction ------------------------------------------------------
        rng = np.random.default_rng(3)
        log_l = rng.normal(size=500)
        log_l[:20] = log_l[20]          # a plateau, to exercise tied ranks
        log_l_min = np.median(log_l)
        select = log_l >= log_l_min
        score = np.zeros(len(log_l))
        score[select] = 0.5 * (1 + (rankdata(log_l[select]) - 0.5) / np.sum(select))
        score[~select] = 0.5 * ((rankdata(log_l[~select]) - 0.5) / np.sum(~select))
        g = f.create_group("score")
        g.create_dataset("log_l", data=log_l)
        g.attrs["log_l_min"] = log_l_min
        g.create_dataset("score", data=score)

        # -- a trained MLP: weights in, predictions out -----------------------------------
        # The Julia forward pass must reproduce these to machine precision.
        rng = np.random.default_rng(1)
        n_dim, n_points = 5, 1000
        x = rng.random((n_points, n_dim))
        y = np.argsort(np.argsort(np.linalg.norm(x - 0.5, axis=1))) / float(n_points)
        net = MLPRegressor(hidden_layer_sizes=(100, 50, 20), alpha=0,
                           learning_rate_init=1e-2, max_iter=10000, tol=0,
                           n_iter_no_change=10, random_state=0).fit(x, y)
        g = f.create_group("mlp")
        g.create_dataset("x", data=x)
        g.create_dataset("y", data=y)
        g.create_dataset("pred", data=net.predict(x))
        g.attrs["n_layers"] = net.n_layers_
        g.attrs["n_iter"] = net.n_iter_
        g.attrs["loss"] = net.loss_
        g.attrs["rmse"] = float(np.sqrt(np.mean((y - net.predict(x)) ** 2)))
        g.attrs["std_y"] = float(np.std(y))
        for k in range(net.n_layers_ - 1):
            g.create_dataset(f"coefs_{k}", data=net.coefs_[k])
            g.create_dataset(f"intercepts_{k}", data=net.intercepts_[k])

    print(f"wrote {OUT}")
    with h5py.File(OUT) as f:
        print("  groups:", ", ".join(f.keys()))
        print(f"  mlp: n_iter={f['mlp'].attrs['n_iter']} "
              f"loss={f['mlp'].attrs['loss']:.6g} "
              f"rmse={f['mlp'].attrs['rmse']:.6g} "
              f"0.3*std_y={0.3 * f['mlp'].attrs['std_y']:.6g}")


if __name__ == "__main__":
    main()
