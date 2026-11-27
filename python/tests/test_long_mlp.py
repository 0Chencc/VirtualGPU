import json

from microbench import host_quiet
from microbench.long_mlp import host_qualified


def test_integrated_long_read_requires_replayable_host_quiet_evidence():
    samples = [
        {"total": 20000, "busy": 1200, "ncpu": 20, "procs": {}},
        {"total": 22000, "busy": 1320, "ncpu": 20, "procs": {}},
    ]
    roots = [7]
    check = host_quiet.check(samples, roots)
    assert check["status"] == "passed"
    record = {
        "device_query": {"integrated": True},
        "host_quiet_gate": {"check": check},
        "host_quiet_check": check,
        "host_samples": samples,
        "host_quiet_roots": roots,
    }
    assert host_qualified(json.loads(json.dumps(record)))
    assert not host_qualified({**record, "host_quiet_gate": None})
    assert not host_qualified({**record, "host_samples": None})
    samples[-1]["busy"] = 1500
    assert not host_qualified(record)
