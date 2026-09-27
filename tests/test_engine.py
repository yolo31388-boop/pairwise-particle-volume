"""粒子体积渲染测试"""
import pytest, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import VolumeRenderer, Particle

class TestSorting:
    def test_sort_by_nearest_point(self):
        r = VolumeRenderer()
        big = Particle([0, 10], [0,0], 10, [1,1,1,1], 1, 1)
        small = Particle([0, 2], [0,0], 1, [1,1,1,1], 1, 1)
        r.particles = [big, small]
        r.sort_particles([0, 0])
        # 大粒子最近点更近，应排在后面（远的先画）
        assert r.particles[0] is small or r.particles[-1] is big

class TestBlend:
    def test_volume_absorption(self):
        r = VolumeRenderer()
        dst = [0, 0, 0, 1]
        for _ in range(10):
            dst = r.blend([1, 0, 0, 0.5], dst)
        assert dst[0] < 0.99, "体积渲染无吸收效果，越叠越亮"

class TestCollision:
    def test_collision_considers_radius(self):
        r = VolumeRenderer()
        p = Particle([0, 5], [0,-10], 3, [1,1,1,1], 1, 1)
        r.collide(p, 0)
        assert p.pos[1] >= 3, f"粒子半径未考虑: {p.pos[1]}"
