import sys
from app.analysis import terrain
print("Import OK")
try:
    terrain.compute_dem("USGS/SRTMGL1_003", "dummy.geojson", None)
except Exception as e:
    import traceback
    traceback.print_exc()
