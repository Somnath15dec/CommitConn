from types import SimpleNamespace

import numpy as np
import torch

from src.analyze_readiness import content_desc, kmeans, residual_kmeans, sid_stats


def test_content_desc_strips_title_and_artist():
    r = SimpleNamespace(track_id="x", track_name="Song. Part 2", artists="A;B",
                        desc="Song. Part 2 by A, B. Genre: rock. Sound: calm. Good for: sleep.")
    assert content_desc(r) == "Genre: rock. Sound: calm. Good for: sleep."


def test_kmeans_separates_blobs():
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(50, 2, generator=g) * 0.1, torch.randn(50, 2, generator=g) * 0.1 + 10])
    a, _ = kmeans(x, 2, iters=10)
    assert len(set(a[:50].tolist())) == 1 and len(set(a[50:].tolist())) == 1 and a[0] != a[50]


def test_sid_stats_collisions():
    codes = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 2, 3]])
    s = sid_stats(codes)
    assert s["collision_rate"] == 0.5 and s["colliding_groups"] == 1 and s["l1_used"] == 2


def test_residual_kmeans_shape():
    x = torch.randn(40, 4)
    assert residual_kmeans(x, levels=3, k=4).shape == (40, 3)
