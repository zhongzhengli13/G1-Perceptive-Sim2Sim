from __future__ import annotations

from typing import Any

TerrainSpec = tuple[str, dict[str, Any]]

DEFAULT_TERRAIN_SPECS: tuple[TerrainSpec, ...] = (
    (
        "pyramid_stairs_with_gaps",
        {
            "name": "pyramid_stairs_with_gaps_1",
            "center": (6.0, 0.0, 0.0),
            "width": 8.0,
            "length": 8.0,
            "num_layers": 5,
            "layer_height": 0.10,
            "step_width": 0.30,
            "center_platform_width": 2.0,
            "gap_width": 0.5,
        },
    ),
    (
        "pyramid_stairs",
        {
            "center": (16.0, 0.0, 0.0),
            "width": 8.0,
            "length": 8.0,
            "num_layers": 10,
            "layer_height": 0.15,
            "step_width": 0.30,
        },
    ),
    (
        "boxes",
        {
            "center": (26.0, 0.0, 0.0),
            "width": 8.0,
            "length": 8.0,
            "nrow": 12,
            "ncol": 12,
            "min_height": 0.05,
            "max_height": 0.20,
        },
    ),
    (
        "slope_pyramid",
        {
            "center": (36.0, 0.0, 0.0),
            "width": 8.0,
            "length": 8.0,
            "top_width": 1.0,
            "top_length": 1.0,
            "height": 0.8,
        },
    ),
    (
        "random_rough",
        {
            "center": (46.0, 0.0, 0.0),
            "width": 8.0,
            "length": 8.0,
            "nrow": 24,
            "ncol": 24,
            "height_scale": 0.20,
            "smooth_passes": 1,
        },
    ),
)
