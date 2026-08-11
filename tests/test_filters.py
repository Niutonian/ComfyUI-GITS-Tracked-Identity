from nodes.tracking_core import OneEuroFilter


def test_constant_signal_is_stable():
    filt = OneEuroFilter(24, 1.2, 0.035)
    values = [filt.apply(10.0) for _ in range(20)]
    assert values == [10.0] * 20


def test_beta_responds_faster():
    slow = OneEuroFilter(24, 1.0, 0.0)
    fast = OneEuroFilter(24, 1.0, 1.0)
    slow.apply(0.0)
    fast.apply(0.0)
    assert fast.apply(100.0) > slow.apply(100.0)
