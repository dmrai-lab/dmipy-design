"""Tests for Protocol save/load and dmipy scheme conversion.

Test IDs
--------
T3-PROTO-1  Protocol survives a save/load roundtrip
T3-PROTO-2  to_dmipy_scheme returns a valid acquisition scheme
"""

import jax
jax.config.update("jax_enable_x64", True)


def test_protocol_save_load_roundtrip(tmp_path):
    """Protocol survives a save/load roundtrip."""
    import numpy as np
    from dmipy_design.protocols import Protocol
    p = Protocol(
        bvalues=np.array([0.0, 1000e6, 2000e6]),
        bvecs=np.eye(3, dtype=np.float32),
        delta=np.full(3, 0.02),
        Delta=np.full(3, 0.04),
    )
    path = str(tmp_path / "test_protocol.npz")
    p.save(path)
    p2 = Protocol.load(path)
    np.testing.assert_allclose(p.bvalues, p2.bvalues)
    np.testing.assert_allclose(p.delta, p2.delta)
    np.testing.assert_allclose(p.bvecs, p2.bvecs)


def test_protocol_to_dmipy_scheme():
    """to_dmipy_scheme returns a valid acquisition scheme."""
    import numpy as np
    from dmipy_design.protocols import Protocol
    n = 8
    bvalues = np.concatenate([np.zeros(2), np.full(6, 1000e6)])
    bvecs = np.vstack([np.tile([0, 0, 1], (2, 1)),
                       np.random.default_rng(0).standard_normal((6, 3))])
    bvecs[2:] /= np.linalg.norm(bvecs[2:], axis=1, keepdims=True)
    p = Protocol(bvalues=bvalues, bvecs=bvecs,
                 delta=np.full(n, 0.02), Delta=np.full(n, 0.04))
    scheme = p.to_dmipy_scheme()
    assert hasattr(scheme, 'bvalues')
    assert len(scheme.bvalues) == n
