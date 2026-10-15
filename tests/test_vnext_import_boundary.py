"""L1/L2 imports should not instantiate or import historical Runtime."""
import subprocess
import sys

def test_vnext_import_does_not_touch_legacy_model():
    script = """
import sys
from codeccat.vnext_strong_l2 import TrainableDetector
assert 'codeccat.model' not in sys.modules
assert 'codeccat.api' not in sys.modules
assert 'codeccat.model_io' not in sys.modules
from codeccat import Detector, RuntimeBundle
assert Detector is not None and RuntimeBundle is not None
"""
    result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

def test_unknown_attribute_has_standard_attribute_error():
    import codeccat
    import pytest
    with pytest.raises(AttributeError):
        getattr(codeccat, "totally_missing_symbol")
