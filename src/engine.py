"""粒子体积渲染 - 6个bug已修复"""
import math
from dataclasses import dataclass, field


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return [0.0, 0.0, 0.0]
    return [v[0] / n, v[1] / n, v[2] / n]


@dataclass
class Particle:
    pos: list
    vel: list
    size: float
    color: list
    life: float
    max_life: float
    rotation: float = 0.0
    seed: int = 0
    rng_state: int = 0
    curve: list = None  # 生命周期关键帧 [(t, value), ...]，None 用默认曲线


class VolumeRenderer:
    # 默认生命周期曲线：淡入 -> 保持 -> 淡出（关键帧处平滑可导）
    DEFAULT_CURVE = [(0.0, 0.0), (0.2, 1.0), (0.8, 1.0), (1.0, 0.0)]

    def __init__(self):
        self.particles: list[Particle] = []

    # ---- bug1 修复：按包围球最近面（最近点）到相机的距离排序，远的先画 ----
    def sort_particles(self, camera_pos):
        def nearest_distance(p):
            center = math.hypot(p.pos[0] - camera_pos[0], p.pos[1] - camera_pos[1])
            return max(center - p.size, 0.0)  # 最近点距离 = 中心距 - 半径

        self.particles.sort(key=lambda p: -nearest_distance(p))

    # ---- bug2 修复：吸收+发射体积渲染方程，指数吸收（Beer-Lambert）----
    def blend(self, src, dst):
        # src[3] 为光学厚度，透射率 T = exp(-alpha)
        # 发射项穿过自身介质被吸收：emission * alpha * T
        # dst 被指数吸收：dst * T，叠加收敛而非线性增亮
        absorb = math.exp(-src[3])
        emission = src[3] * absorb
        return [
            src[0] * emission + dst[0] * absorb,
            src[1] * emission + dst[1] * absorb,
            src[2] * emission + dst[2] * absorb,
            dst[3] + src[3],  # 累积光学厚度
        ]

    # ---- bug3 修复：由视线方向重建正交基，不受相机 roll 影响；支持轴向 billboard ----
    def billboard(self, particle, camera_up, camera_right, axis=None):
        # 视线方向 = right x up，对 roll 不变（roll 只是绕视线旋转 up/right）
        view = _normalize(_cross(camera_right, camera_up))
        if axis is not None:
            # 轴向 billboard：up 固定为轴方向，right 垂直于轴与视线
            up = _normalize(axis)
            right = _normalize(_cross(up, view))
        else:
            # 用世界 up 重建无 roll 的 right/up
            world_up = [0.0, 1.0, 0.0]
            right = _normalize(_cross(world_up, view))
            if right == [0.0, 0.0, 0.0]:  # 视线与世界 up 平行时退化
                right = [1.0, 0.0, 0.0]
            up = _normalize(_cross(view, right))
        # 粒子自转应用在 billboard 平面内
        if particle.rotation:
            c, s = math.cos(particle.rotation), math.sin(particle.rotation)
            r = [right[i] * c + up[i] * s for i in range(3)]
            u = [up[i] * c - right[i] * s for i in range(3)]
            right, up = r, u
        return [right, up]

    # ---- bug4 修复：每粒子独立 PCG32 随机状态，确定性、不依赖全局随机 ----
    @staticmethod
    def _pcg32(state):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        xorshifted = (((state >> 18) ^ state) >> 27) & 0xFFFFFFFF
        rot = state >> 59
        value = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
        return state, value / 0x100000000

    def _next_random(self, p):
        if p.rng_state == 0:
            # 由粒子自身 seed 确定性初始化（保证非零）
            p.rng_state = (p.seed * 2 + 1) & 0xFFFFFFFFFFFFFFFF or 0x9E3779B97F4A7C15
        p.rng_state, value = self._pcg32(p.rng_state)
        return value

    # ---- bug5 修复：Catmull-Rom（非均匀参数化）平滑插值，关键帧处连续可导 ----
    @staticmethod
    def _tangents(keys):
        # 割线切线 m_i = (v_{i+1}-v_{i-1})/(t_{i+1}-t_{i-1})，端点用单侧割线
        n = len(keys)
        tangents = []
        for i in range(n):
            if i == 0:
                m = (keys[1][1] - keys[0][1]) / (keys[1][0] - keys[0][0])
            elif i == n - 1:
                m = (keys[-1][1] - keys[-2][1]) / (keys[-1][0] - keys[-2][0])
            else:
                m = (keys[i + 1][1] - keys[i - 1][1]) / (keys[i + 1][0] - keys[i - 1][0])
            tangents.append(m)
        return tangents

    def eval_curve(self, keys, x):
        keys = sorted(keys)
        if len(keys) < 2 or x <= keys[0][0]:
            return keys[0][1]
        if x >= keys[-1][0]:
            return keys[-1][1]
        tangents = self._tangents(keys)
        for i in range(len(keys) - 1):
            t1, v1 = keys[i]
            t2, v2 = keys[i + 1]
            if t1 <= x <= t2:
                h = t2 - t1
                u = (x - t1) / h
                u2, u3 = u * u, u * u * u
                # 三次 Hermite（非均匀 Catmull-Rom），对 x C1 连续
                return (
                    (2 * u3 - 3 * u2 + 1) * v1
                    + (u3 - 2 * u2 + u) * h * tangents[i]
                    + (-2 * u3 + 3 * u2) * v2
                    + (u3 - u2) * h * tangents[i + 1]
                )
        return keys[-1][1]

    def update_particle(self, p, dt):
        rnd = self._next_random(p)  # 每粒子独立随机流
        p.vel[0] += (rnd - 0.5) * dt
        p.life -= dt
        age = 1.0 - max(p.life, 0.0) / p.max_life  # 归一化年龄 0->1
        keys = p.curve if p.curve else self.DEFAULT_CURVE
        p.color[3] = self.eval_curve(keys, age)

    # ---- bug6 修复：按粒子半径扩展碰撞面，响应按半径偏移并加滑动摩擦 ----
    def collide(self, p, plane_y, restitution=0.5, friction=0.2):
        radius = p.size  # 碰撞面沿法线外扩一个半径
        if p.pos[1] - radius < plane_y:
            p.pos[1] = plane_y + radius  # 按半径偏移，不再穿模
            p.vel[1] = abs(p.vel[1]) * restitution
            p.vel[0] *= 1.0 - friction  # 滑动摩擦衰减切向速度
