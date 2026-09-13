from app.__main__ import _is_benign


def test_known_platform_warnings_are_filtered():
    assert _is_benign("QSocketNotifier: Can only be used with threads started with QThread")
    assert _is_benign("Wayland does not support QWindow::requestActivate()")


def test_real_warnings_pass_through():
    assert not _is_benign("QObject::connect: No such signal Foo::bar()")
    assert not _is_benign("Traceback (most recent call last):")
