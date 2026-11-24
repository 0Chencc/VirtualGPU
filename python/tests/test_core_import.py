import virtualgpu
from virtualgpu import _core


def test_core_version_matches_package():
    assert _core.__version__ == virtualgpu.__version__
