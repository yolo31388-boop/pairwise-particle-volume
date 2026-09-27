"""particles.volume 的八个测试, 覆盖六个高难度 bug 的修复."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particles.volume import (
    VolumeRenderer,
    Particle,
    catmull_rom_sample,
)


class TestSorting:
    def test_sort_by_nearest_point(self):
        # bug1: 大粒子中心更远, 但包围球最近面更近 -> 应后画 (盖在上面)
        r = VolumeRenderer()
        big = Particle([0, 10], [0, 0], 10, [1, 1, 1, 1], 1, 1)
        small = Particle([0, 2], [0, 0], 1, [1, 1, 1, 1], 1, 1)
        r.particles = [big, small]
        r.sort_particles([0, 0])
        assert r.particles[0] is small
        assert r.particles[-1] is big

    def test_sort_large_particle_covers_small(self):
        # 近相机的小粒子被中心更远的大粒子体积覆盖, 排序必须与中心距离排序相反
        r = VolumeRenderer()
        near_small = Particle([0, 3.0], [0, 0], 0.5, [1, 1, 1, 1], 1, 1)
        far_big = Particle([0, 6.0], [0, 0], 5.0, [1, 1, 1, 1], 1, 1)
        assert r.nearest_distance(near_small, [0, 0]) > r.nearest_distance(
            far_big, [0, 0]
        )
        r.particles = [near_small, far_big]
        r.sort_particles([0, 0])
        assert r.particles[0] is near_small
        assert r.particles[1] is far_big


class TestBlend:
    def test_volume_absorption(self):
        # bug2: 指数吸收, 叠加不收敛到纯白且每层辐射有界
        r = VolumeRenderer()
        dst = [0.0, 0.0, 0.0, 1.0]
        for _ in range(10):
            dst = r.blend([1, 0, 0, 0.5], dst)
        assert dst[0] < 0.99, "体积渲染无吸收效果, 越叠越亮"
        for _ in range(200):
            dst = r.blend([1, 0, 0, 0.5], dst)
        assert dst[0] < 1.0
        assert dst[1] < 1e-9 and dst[2] < 1e-9, "绿蓝通道不应被红色层点亮"


class TestBillboard:
    def test_billboard_roll_invariant(self):
        # bug3: 同一观察方向, 相机 roll 后 up/right 不同, billboard 基应保持一致
        r = VolumeRenderer()
        p = Particle([0, 0, -5], [0, 0, 0], 1, [1, 1, 1, 1], 1, 1)
        forward = [0, 0, -1]
        base = r.billboard(p, camera_forward=forward, world_up=[0, 1, 0])
        rolled = r.billboard(
            p,
            camera_up=[0.3, 0.9539, 0],
            camera_right=[0.9539, -0.3, 0],
            camera_forward=forward,
            world_up=[0, 1, 0],
        )
        for axis in range(2):
            for component in range(3):
                assert abs(base[axis][component] - rolled[axis][component]) < 1e-9
        right, up = base
        assert abs(right[0] - 1.0) < 1e-9
        assert abs(up[1] - 1.0) < 1e-9
        assert abs(sum(right[i] * up[i] for i in range(3))) < 1e-9

    def test_axial_billboard(self):
        # 轴向 billboard: up 对齐粒子轴在屏幕平面的投影, right 与之正交且在视平面内
        r = VolumeRenderer()
        p = Particle(
            [0, 0, -5], [0, 0, 0], 1, [1, 1, 1, 1], 1, 1,
            axis=[0, 1, 0],
        )
        forward = [0, 0, -1]
        right, up = r.billboard(p, camera_forward=forward)
        assert abs(up[1] - 1.0) < 1e-9
        assert abs(right[0] - 1.0) < 1e-9
        dot_ru = sum(right[i] * up[i] for i in range(3))
        assert abs(dot_ru) < 1e-9
        dot_rf = sum(right[i] * forward[i] for i in range(3))
        assert abs(dot_rf) < 1e-9


class TestRandom:
    def test_per_particle_deterministic(self):
        # bug4: 每粒子独立 PCG 状态, 更新顺序不影响结果, 无全局随机依赖
        def simulate(order):
            r = VolumeRenderer()
            particles = [
                Particle([0, 0], [0, 0], 1, [1, 1, 1, 1], 10, 10, seed=7),
                Particle([0, 0], [0, 0], 1, [1, 1, 1, 1], 10, 10, seed=23),
                Particle([0, 0], [0, 0], 1, [1, 1, 1, 1], 10, 10, seed=99),
            ]
            r.particles = particles
            for _ in range(50):
                for index in order:
                    r.update_particle(particles[index], 1 / 60)
            return [list(p.pos) + list(p.vel) for p in particles]

        forward_order = simulate([0, 1, 2])
        reverse_order = simulate([2, 1, 0])
        for a, b in zip(forward_order, reverse_order):
            for x, y in zip(a, b):
                assert x == y


class TestLifeCurve:
    def test_catmull_rom_smooth(self):
        # bug5: Catmull-Rom 关键帧处 C1 连续, 端点精确, 曲线过关键帧
        keys = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
        for t, v in keys:
            assert abs(catmull_rom_sample(keys, t) - v) < 1e-12

        def slope(t):
            h = 1e-5
            return (catmull_rom_sample(keys, t + h) -
                    catmull_rom_sample(keys, t - h)) / (2 * h)

        for knot in (1.0, 2.0):
            left = slope(knot - 1e-4)
            right = slope(knot + 1e-4)
            assert abs(left - right) < 1e-3, f"关键帧 {knot} 处导数不连续"

        # 非等距关键帧也必须 C1 连续 (验证时间轴归一化正确)
        uneven = [(0.0, 0.0), (0.1, 1.0), (2.0, 0.0), (3.0, 1.0)]
        for t, v in uneven:
            assert abs(catmull_rom_sample(uneven, t) - v) < 1e-12
        for knot in (0.1, 2.0):
            h = 1e-7
            left = (catmull_rom_sample(uneven, knot - h) -
                    catmull_rom_sample(uneven, knot - 2 * h)) / h
            right = (catmull_rom_sample(uneven, knot + 2 * h) -
                     catmull_rom_sample(uneven, knot + h)) / h
            assert abs(left - right) < 1e-3

        # 渲染器默认生命周期曲线也必须是平滑可导的
        r = VolumeRenderer()
        p = Particle([0, 0], [0, 0], 1, [1, 1, 1, 1], 2.0, 2.0, seed=1)
        left = (r.sample_life_curve(p, 0.5 - 1e-5) -
                r.sample_life_curve(p, 0.5 - 2e-5)) / 1e-5
        right = (r.sample_life_curve(p, 0.5 + 2e-5) -
                 r.sample_life_curve(p, 0.5 + 1e-5)) / 1e-5
        assert abs(left - right) < 1e-3


class TestCollision:
    def test_collision_considers_radius(self):
        # bug6: 碰撞面按半径扩展, 接触时偏移到 plane + radius, 反弹 + 切向摩擦
        r = VolumeRenderer()
        p = Particle([0, 2], [3, -10], 3, [1, 1, 1, 1], 1, 1)
        r.collide(p, 0)
        assert p.pos[1] >= 3
        assert p.vel[1] > 0
        assert p.vel[0] < 3, "切向速度未施加滑动摩擦"

        # 尚未接触的粒子不应被错误吸附到平面
        floating = Particle([0, 4], [0, -1], 3, [1, 1, 1, 1], 1, 1)
        assert r.collide(floating, 0) is False
        assert floating.pos[1] == 4
