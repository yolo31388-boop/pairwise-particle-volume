"""向后兼容入口: 体积渲染核心已迁移到 particles/volume.py."""
from particles.volume import (
    VolumeRenderer,
    Particle,
    catmull_rom_sample,
    bezier_cubic,
    pcg32_next,
    pcg32_seed,
)

__all__ = [
    "VolumeRenderer",
    "Particle",
    "catmull_rom_sample",
    "bezier_cubic",
    "pcg32_next",
    "pcg32_seed",
]
