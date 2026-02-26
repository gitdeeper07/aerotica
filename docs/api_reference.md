### `aerotica.offshore`

- `OffshoreOptimizer` - Main optimizer interface
- `WakeModel` - Wake effect modeling
- `TurbineLayout` - Layout optimization
- `OffshoreResource` - Resource assessment

## CLI Commands

```bash
aerotica compute-ake --site tokyo
aerotica gust-alert --site tokyo --interval 30
aerotica urban-assessment --lidar data/lidar/tokyo.tif
aerotica offshore-optimize --lat 55 --lon -3 --depth 50
aerotica serve --host 0.0.0.0 --port 8000
aerotica dashboard --port 8501
```

Configuration

See configuration documentation for details.
