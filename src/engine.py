"""粒子体积渲染 - 带6个bug"""
import math
from dataclasses import dataclass, field

@dataclass
class Particle:
    pos: list
    vel: list
    size: float
    color: list
    life: float
    max_life: float
    rotation: float = 0.0

class VolumeRenderer:
    def __init__(self):
        self.particles: list[Particle] = []
        self.global_random = 0.5  # bug4: 全局随机

    def sort_particles(self, camera_pos):
        # bug1: 用中心点距离排序
        self.particles.sort(key=lambda p: math.hypot(p.pos[0]-camera_pos[0], p.pos[1]-camera_pos[1]))

    def blend(self, src, dst):
        # bug2: 普通alpha混合，无体积吸收
        alpha = src[3]
        return [
            src[0]*alpha + dst[0]*(1-alpha),
            src[1]*alpha + dst[1]*(1-alpha),
            src[2]*alpha + dst[2]*(1-alpha),
            min(1.0, src[3]+dst[3])
        ]

    def billboard(self, particle, camera_up, camera_right):
        # bug3: 不考虑camera roll
        return [camera_right, camera_up]

    def update_particle(self, p, dt):
        # bug4: 用全局随机
        p.vel[0] += (self.global_random - 0.5) * dt
        p.life -= dt
        # bug6: 生命周期曲线线性
        t = p.life / p.max_life
        p.color[3] = t  # 线性，拐点明显

    def collide(self, p, plane_y):
        # bug6: 用中心点碰撞，不考虑粒子大小
        if p.pos[1] < plane_y:
            p.pos[1] = plane_y
            p.vel[1] = abs(p.vel[1]) * 0.5
