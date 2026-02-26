# Quick Start Guide

## Compute AKE Index

```python
from aerotica.ake import AKEComposite

ake = AKEComposite(
    site_id="tokyo",
    climate_zone="temperate",
    site_type="urban"
)

ake.load_parameters({
    "KED": 0.83, "TII": 0.76, "VSR": 0.89,
    "AOD": 0.34, "THD": 0.72, "PGF": 0.65,
    "HCI": 0.59, "ASI": 0.71, "LRC": 0.44
})

result = ake.compute()
print(f"AKE Score: {result['score']:.3f}")
print(f"Classification: {result['classification']}")
```

Run Gust Pre-Alerting

```python
from aerotica.alerts import GustPreAlertEngine

engine = GustPreAlertEngine(
    site_config={"location": "tokyo"}
)

alert = engine.evaluate(observations)
if alert:
    print(f"⚠️ Gust detected! Lead time: {alert['lead_time_seconds']}s")
```

Urban Wind Assessment

```python
from aerotica.urban import BuildingWindAssessor

assessor = BuildingWindAssessor(
    lidar_dem="data/lidar/tokyo.tif"
)

sites = assessor.identify_sites()
print(f"Found {len(sites)} viable sites")
```

