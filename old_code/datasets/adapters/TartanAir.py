# Delegate to the canonical implementation in /data2/d4rt/code.
# This file exists only because verify_datasets1.py inserts /data1/zbf/my_dfrt
# at the front of sys.path. All real logic lives in:
#   /data2/d4rt/code/datasets/adapters/TartanAir.py
import importlib.util, sys
_spec = importlib.util.spec_from_file_location(
    "datasets.adapters.TartanAir_canonical",
    "/data2/d4rt/code/datasets/adapters/TartanAir.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TartanAirAdapter = _mod.TartanAirAdapter
quat_to_rotation_matrix = _mod.quat_to_rotation_matrix
