"""粒子体积渲染核心.

修复六个高难度问题:
1. 排序: 按包围球到相机的最近点距离 (中心距离 - 半径) 做画家算法排序.
2. 融合: 吸收 + 发射体积渲染方程, Beer-Lambert 指数吸收, 不再越叠越白.
3. billboard: 用相机 forward 与世界 up 重建正交基, 不受相机 roll 影响, 支持轴向 billboard.
4. 随机数: 每个粒子持有独立 PCG32 状态, 与更新顺序/线程无关, 完全确定.
5. 生命周期曲线: Catmull-Rom 平滑插值, 关键帧处 C1 连续可导.
6. 碰撞: 用粒子半径扩展碰撞面, 接触时按半径偏移, 法向反弹 + 切向滑动摩擦.
"""

import math
from dataclasses import dataclass, field


MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
PCG_INC = 0x853C49E6748FEA9B | 1


def _vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def _vec_scale(a, s):
    return [v * s for v in a]


def _vec_dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def _vec_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _vec_length(a):
    return math.sqrt(sum(v * v for v in a))


def _vec_normalize(a):
    length = _vec_length(a)
    if length < 1e-12:
        return list(a)
    return [v / length for v in a]


def pcg32_next(state):
    """推进一次 PCG32 状态, 返回 (随机 float32 in [0,1), 新状态)."""
    state = (state * 6364136223846793005 + PCG_INC) & MASK64
    xorshifted = (((state >> 18) ^ state) >> 27) & MASK32
    rot = (state >> 59) & 31
    rotated = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK32
    return rotated / float(1 << 32), state


def pcg32_seed(seed):
    """由任意 64bit 种子初始化 PCG32 状态."""
    state = 0
    _, state = pcg32_next(state)
    state = (state + (seed & MASK64)) & MASK64
    _, state = pcg32_next(state)
    return state


def catmull_rom_sample(points, t):
    """对一组 (time, value) 控制点做 Catmull-Rom 平滑插值.

    关键帧处位置与切线都连续 (C1)。端点用幻影点钳制。
    """
    points = sorted(points, key=lambda kv: kv[0])
    if not points:
        return 0.0
    if t <= points[0][0]:
        return float(points[0][1])
    if t >= points[-1][0]:
        return float(points[-1][1])

    n = len(points)
    for i in range(n - 1):
        t0, p0 = points[i]
        t1, p1 = points[i + 1]
        if t0 <= t <= t1:
            def _time(j):
                if 0 <= j < n:
                    return points[j][0]
                return t0 - (t1 - t0) if j < 0 else t1 + (t1 - t0)

            times = [_time(j) for j in (i - 1, i, i + 1, i + 2)]
            vals = [points[j][1] if 0 <= j < n else None
                    for j in (i - 1, i, i + 1, i + 2)]
            if vals[0] is None:
                vals[0] = 2 * p0 - p1
            if vals[3] is None:
                vals[3] = 2 * p1 - p0
            # 中心差分切线 (跨段共用同一切线 => 关键帧处 C1)
            m0 = (vals[2] - vals[0]) / (times[2] - times[0])
            m1 = (vals[3] - vals[1]) / (times[3] - times[1])
            dt = t1 - t0
            u = (t - t0) / dt
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            return h00 * p0 + h10 * m0 * dt + h01 * p1 + h11 * m1 * dt
    return float(points[-1][1])


def bezier_cubic(p0, p1, p2, p3, t):
    """三次贝塞尔曲线 (作为平滑曲线的备选实现)."""
    u = 1.0 - t
    return (
        u ** 3 * p0
        + 3 * u ** 2 * t * p1
        + 3 * u * t ** 2 * p2
        + t ** 3 * p3
    )


@dataclass
class Particle:
    pos: list
    vel: list
    size: float
    color: list
    life: float
    max_life: float
    rotation: float = 0.0
    # 每粒子独立随机数状态与种子 (bug4 修复)
    seed: int = 0
    rng_state: int = 0
    # 轴向 billboard 的自转轴 (单位向量), None 表示球面 billboard
    axis: object = None
    # 每粒子生命周期曲线 [(time, alpha), ...], None 用渲染器默认曲线
    life_curve: object = None


class VolumeRenderer:
    def __init__(self, restitution=0.5, friction=0.8):
        self.particles = []
        # bug2 修复后不再需要全局随机; 保留属性仅为向后兼容
        self.global_random = 0.5
        # 生命周期默认曲线: 出生到死亡 alpha 1 -> 0, 形状与旧线性端点一致
        self.default_life_curve = [(0.0, 0.0), (1.0, 1.0)]
        self.restitution = restitution
        self.friction = friction
        self._spawn_counter = 0

    def add_particle(self, p):
        if p.rng_state == 0:
            seed = p.seed if p.seed else (self._spawn_counter + 1)
            p.rng_state = pcg32_seed(seed)
        self._spawn_counter += 1
        self.particles.append(p)
        return p

    # -- bug1: 包围球最近点距离排序 ------------------------------------
    def nearest_distance(self, p, camera_pos):
        """粒子包围球表面到相机的最近距离 = 中心距离 - 半径."""
        center_dist = math.sqrt(
            sum((p.pos[i] - camera_pos[i]) ** 2 for i in range(len(camera_pos)))
        )
        return center_dist - p.size

    def sort_particles(self, camera_pos):
        # 画家算法: 远的先画 (最近点距离大的在前), 近的后画覆盖之
        self.particles.sort(
            key=lambda p: self.nearest_distance(p, camera_pos), reverse=True
        )

    # -- bug2: 吸收 + 发射体积渲染方程 ----------------------------------
    def blend(self, src, dst):
        """back-to-front 体积积分 (中点发射 + Beer-Lambert 指数吸收).

        tau = src[3] 为该粒子层的光学厚度;
        T = exp(-tau) 为该层透射率;
        背景被指数吸收, 粒子发射按层中点透射率 T_mid = exp(-tau/2) 计入。
        多层叠加收敛到有限辐射亮度而不是纯白。
        """
        tau = max(0.0, float(src[3]))
        transmittance = math.exp(-tau)
        mid_transmittance = math.exp(-tau * 0.5)
        return [
            src[i] * tau * mid_transmittance + dst[i] * transmittance
            for i in range(3)
        ] + [
            (1.0 - transmittance) + transmittance * dst[3]
        ]

    # -- bug3: roll 无关的 billboard 基 ---------------------------------
    def billboard(self, particle, camera_up=None, camera_right=None,
                  camera_forward=None, world_up=None):
        """返回粒子四边形的 [right, up] 正交基.

        - 提供 camera_forward 时: 与 world_up 重建基, 相机 roll 不影响结果;
          particle.axis 非空时使用轴向 billboard (up 对齐粒子轴投影)。
        - 仅提供相机 up/right 时: 归一化并做 Gram-Schmidt 正交化 (2D/3D 均可)。
        """
        if camera_forward is not None:
            forward = _vec_normalize(camera_forward)
            reference_up = _vec_normalize(world_up) if world_up else [0.0, 1.0, 0.0]
            if particle is not None and getattr(particle, "axis", None) is not None:
                # 轴向 billboard: up 固定为粒子轴在屏幕上的投影,
                # right 随观察方向旋转。
                axis = _vec_normalize(particle.axis)
                up = _vec_sub(axis, _vec_scale(forward, _vec_dot(axis, forward)))
                up = _vec_normalize(up)
                right = _vec_normalize(_vec_cross(forward, up))
                return [right, up]
            right = _vec_normalize(_vec_cross(forward, reference_up))
            up = _vec_cross(right, forward)
            return [right, up]

        if camera_right is not None and camera_up is not None:
            right = _vec_normalize(camera_right)
            up = _vec_normalize(camera_up)
            if len(right) == len(up):
                up = _vec_sub(up, _vec_scale(right, _vec_dot(up, right)))
                up = _vec_normalize(up)
            return [right, up]
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    # -- bug4: 每粒子独立 PCG32 随机数 -----------------------------------
    def _particle_random(self, p):
        if p.rng_state == 0:
            p.rng_state = pcg32_seed(p.seed & MASK64)
        value, p.rng_state = pcg32_next(p.rng_state)
        return value

    # -- bug5: Catmull-Rom 平滑生命周期曲线 ------------------------------
    def sample_life_curve(self, p, t):
        keys = p.life_curve if p.life_curve is not None else self.default_life_curve
        return catmull_rom_sample(keys, t)

    def update_particle(self, p, dt):
        dims = len(p.pos)
        for axis in range(dims):
            noise = self._particle_random(p) - 0.5
            p.vel[axis] += noise * dt
        p.pos = [p.pos[i] + p.vel[i] * dt for i in range(dims)]
        p.life -= dt
        t = max(0.0, min(1.0, p.life / p.max_life))
        p.color[3] = self.sample_life_curve(p, t)

    # -- bug6: 半径扩展碰撞面 + 反弹/滑动摩擦 ----------------------------
    def collide(self, p, plane_y):
        radius = p.size
        contact = plane_y + radius
        if p.pos[1] >= contact:
            return False
        # 按半径偏移到刚好接触, 大粒子不再半个体积穿过平面
        p.pos[1] = contact
        if p.vel[1] < 0:
            p.vel[1] = -p.vel[1] * self.restitution
        # 切向速度施加滑动摩擦 (2D: x; 3D: x/z)
        p.vel[0] *= self.friction
        if len(p.vel) > 2:
            p.vel[2] *= self.friction
        return True
