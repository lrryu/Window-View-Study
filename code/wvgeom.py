"""
wvgeom.py — 窗景退距几何引擎

坐标约定
--------
窗（玻璃）平面位于 z = 0，洞口矩形 x in [x0, x1], y in [y0, y1]。
观察者眼点位于 (ex, ey, -d)，朝 +z 方向。d 为眼点到玻璃平面的法向距离（退距）。
y 为竖直方向（向上为正），x 为水平方向。眼高 ey 以室内地面为 0。

本模块只做几何：可见立体角、角域边界、多重孔径归约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------
# 1. 矩形孔径的可见立体角（闭式解）
# --------------------------------------------------------------------------

def _primitive(X: np.ndarray | float, Y: np.ndarray | float, d: float):
    """atan2 形式的原函数 F(x, y) = atan(x*y / (d*r))，r = sqrt(x^2+y^2+d^2)。"""
    return np.arctan2(X * Y, d * np.sqrt(X * X + Y * Y + d * d))


def omega_rect(x0: float, x1: float, y0: float, y1: float,
               d: float, ex: float = 0.0, ey: float = 0.0) -> float:
    """矩形洞口对眼点的可见立体角（sr）。

    洞口在 z=0，眼点在 (ex, ey, -d)。d>0。
    对 x、y 各做一次定积分：
        Omega = F(x1,y1) - F(x0,y1) - F(x1,y0) + F(x0,y0)
    """
    if x1 <= x0 or y1 <= y0:
        return 0.0
    if d <= 0.0:
        raise ValueError("d must be > 0; the eye-in-aperture-plane limit is singular")
    X1, X2 = x0 - ex, x1 - ex
    Y1, Y2 = y0 - ey, y1 - ey
    val = (_primitive(X2, Y2, d) - _primitive(X1, Y2, d)
           - _primitive(X2, Y1, d) + _primitive(X1, Y1, d))
    return float(val)


def omega_rect_gl(x0, x1, y0, y1, d, ex=0.0, ey=0.0, n: int = 200) -> float:
    """二维 Gauss-Legendre 求积，作为闭式解的独立校验。"""
    xs, wx = np.polynomial.legendre.leggauss(n)
    ys, wy = np.polynomial.legendre.leggauss(n)
    # 映射到 [x0,x1] x [y0,y1]
    X = 0.5 * (x1 - x0) * xs + 0.5 * (x1 + x0) - ex
    Y = 0.5 * (y1 - y0) * ys + 0.5 * (y1 + y0) - ey
    WX = 0.5 * (x1 - x0) * wx
    WY = 0.5 * (y1 - y0) * wy
    XX, YY = np.meshgrid(X, Y, indexing="xy")
    WW = np.outer(WY, WX)
    integrand = d / np.power(XX * XX + YY * YY + d * d, 1.5)
    return float(np.sum(WW * integrand))


def omega_rect_mc(x0, x1, y0, y1, d, ex=0.0, ey=0.0,
                  n: int = 40_000_000, seed: int = 20261105) -> tuple[float, float]:
    """半球均匀采样蒙特卡洛校验，返回 (估计值, 标准误)。"""
    rng = np.random.default_rng(seed)
    # 在 (u,v) = tan(azimuth), tan(elevation) 的包围盒外扩 20% 内均匀采样，
    # 按立体角雅可比加权：dOmega = du dv / (1+u^2+v^2)^(3/2)。
    u0, u1 = (x0 - ex) / d, (x1 - ex) / d
    v0, v1 = (y0 - ey) / d, (y1 - ey) / d
    pad_u = 0.2 * (u1 - u0) + 1e-6
    pad_v = 0.2 * (v1 - v0) + 1e-6
    # 分块以控制内存
    total = 0.0
    total_sq = 0.0
    done = 0
    chunk = 2_000_000
    while done < n:
        m = min(chunk, n - done)
        U = rng.uniform(u0 - pad_u, u1 + pad_u, m)
        V = rng.uniform(v0 - pad_v, v1 + pad_v, m)
        inside = (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1)
        w = 1.0 / np.power(1.0 + U * U + V * V, 1.5)
        # 采样区域面积
        area = (u1 - u0 + 2 * pad_u) * (v1 - v0 + 2 * pad_v)
        vals = w * inside * area
        total += float(vals.sum())
        total_sq += float((vals * vals).sum())
        done += m
    mean = total / n
    var = max(total_sq / n - mean * mean, 0.0)
    se = np.sqrt(var / n)
    return mean, se


def omega_rect_mc_hem(x0, x1, y0, y1, d, ex=0.0, ey=0.0,
                      n: int = 40_000_000, seed: int = 20261105) -> tuple[float, float]:
    """前向半球均匀采样的蒙特卡洛校验，返回 (估计值, 标准误)。

    半球均匀：phi ~ U(-pi/2, pi/2)，sin(theta) ~ U(-1, 1)。
    """
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    chunk = 4_000_000
    while done < n:
        m = min(chunk, n - done)
        phi = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, m)
        st = rng.uniform(-1.0, 1.0, m)
        ct = np.sqrt(np.maximum(1.0 - st * st, 0.0))
        dx = np.sin(phi) * ct
        dy = st
        dz = np.cos(phi) * ct
        t = d / np.maximum(dz, 1e-12)
        x = ex + t * dx
        y = ey + t * dy
        hit = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        hits += int(hit.sum())
        done += m
    p = hits / n
    omega = 2 * np.pi * p
    se = 2 * np.pi * np.sqrt(max(p * (1 - p), 0.0) / n)
    return float(omega), float(se)


# --------------------------------------------------------------------------
# 2. 任意平面多边形（三角扇 + Van Oosterom–Strackee）
# --------------------------------------------------------------------------

def _omega_tri(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    num = float(np.dot(a, np.cross(b, c)))
    da, db, dc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    den = (da * db * dc + np.dot(a, b) * dc + np.dot(a, c) * db
           + np.dot(b, c) * da)
    return 2.0 * np.arctan2(num, den)


def omega_polygon(pts: Iterable[tuple[float, float]], d: float,
                  ex: float = 0.0, ey: float = 0.0) -> float:
    """平面多边形（位于 z=0）对眼点的可见立体角，pts 为 (x,y) 序列。"""
    P = np.asarray(list(pts), dtype=float)
    if len(P) < 3:
        return 0.0
    v = np.column_stack([P[:, 0] - ex, P[:, 1] - ey, np.full(len(P), d)])
    acc = 0.0
    for i in range(1, len(P) - 1):
        acc += _omega_tri(v[0], v[i], v[i + 1])
    return float(abs(acc))


def circle_polygon(cx: float, cy: float, r: float, n: int = 256):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))


# --------------------------------------------------------------------------
# 3. 多重孔径归约
# --------------------------------------------------------------------------
#
# 引理：设眼点 E=(ex,ey,-d)，某孔径矩形位于平面 z=zf，其 u=tan(方位)、
# v=tan(仰角) 的角域为 u in [(x0-ex)/(d+zf), (x1-ex)/(d+zf)], 同理 v。
# 因此任意多个平行于窗面的矩形孔径（含直边斜套的墙洞）在 (u,v) 角域内
# 都是轴对齐矩形；其交集仍为轴对齐矩形。可见方向集合可归约为一个等效
# 矩形，立体角由该等效矩形在玻璃平面处给出。
#
# 斜套（splay）侧壁是平面，射线与侧壁的相交条件在 z 的两个端点上均为
# 线性约束，两端点都满足即全程满足；故 splay 与同尺寸的 tunnel 具有
# 相同的可见方向集合。

@dataclass
class Frame:
    """一个平行于窗面的矩形孔径。zf=0 为玻璃平面；zf<0 为室内侧墙洞；zf>0 为外凸。"""
    zf: float
    x0: float
    x1: float
    y0: float
    y1: float

    def uv(self, d: float, ex: float, ey: float):
        D = d + self.zf
        if D <= 1e-9:
            return None
        return ((self.x0 - ex) / D, (self.x1 - ex) / D,
                (self.y0 - ey) / D, (self.y1 - ey) / D)


def intersect_uv(rects):
    """rects: [(u0,u1,v0,v1), ...]；返回交集或 None。"""
    u0 = max(r[0] for r in rects)
    u1 = min(r[1] for r in rects)
    v0 = max(r[2] for r in rects)
    v1 = min(r[3] for r in rects)
    if u1 <= u0 or v1 <= v0:
        return None
    return (u0, u1, v0, v1)


@dataclass
class Pane:
    """一个玻璃窗格（可被墙洞裁剪）。zf 为其所在平面。"""
    zf: float
    x0: float
    x1: float
    y0: float
    y1: float
    clipped_uv: tuple | None = None


# --------------------------------------------------------------------------
# 4. 角域读数
# --------------------------------------------------------------------------

def angular_extent(uv) -> tuple[float, float]:
    """返回 (水平角宽, 竖直角宽)，单位度。uv=(u0,u1,v0,v1)。"""
    u0, u1, v0, v1 = uv
    ang_h = np.arctan(u1) - np.arctan(u0)
    ang_v = np.arctan(v1) - np.arctan(v0)
    return float(np.degrees(ang_h)), float(np.degrees(ang_v))


def direction_from_uv(u, v):
    """(u,v) -> 单位方向向量；u=x/z, v=y/z。"""
    n = np.sqrt(1.0 + u * u + v * v)
    return np.stack([u / n, v / n, 1.0 / n], axis=-1)


def uv_from_direction(vec):
    vec = np.asarray(vec, dtype=float)
    return vec[..., 0] / vec[..., 2], vec[..., 1] / vec[..., 2]
