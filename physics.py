"""
Physics engine for Physics Playground.

All formulas sourced from the deep-research knowledge base:
  - computer-science-wiki/software-engineering/physics-simulation-algorithms.md
  - mechanical-engineering-wiki/concepts/newtons-laws.md
  - mechanical-engineering-wiki/mechanics/dynamics.md
  - mechanical-engineering-wiki/mechanics/vibrations.md
  - mechanical-engineering-wiki/concepts/conservation-principles.md
  - mechanical-engineering-wiki/fluids/fluid-mechanics.md
  - mathematics-wiki/optimization/numerical-methods.md
"""

import math
from enum import Enum, auto


class IntegratorType(Enum):
    VERLET = auto()
    SEMI_EULER = auto()
    RK4 = auto()

    @property
    def label(self):
        return {
            IntegratorType.VERLET: "Velocity Verlet",
            IntegratorType.SEMI_EULER: "Semi-Implicit Euler",
            IntegratorType.RK4: "Runge-Kutta 4",
        }[self]


# ─── Constants (scaled for pixel-space) ─────────────────────────
PX_PER_METER = 50
GRAVITY = 9.81 * PX_PER_METER          # px/s² downward
G_WELL_SCALE = 5e5                       # gravity well strength
AIR_DENSITY = 1.204                      # kg/m³ at 20°C, 1 atm
DRAG_CD = 0.44                           # sphere drag coefficient
FRICTION_MU = 0.4                        # kinetic friction coefficient


# ─── Integration Methods ────────────────────────────────────────
# Source: physics-simulation-algorithms.md

def velocity_verlet(px, py, vx, vy, ax, ay, dt, force_fn, mass):
    """
    Symplectic integrator with bounded energy drift.

        v_{n+½} = v_n + (dt/2) · a_n
        x_{n+1} = x_n + dt · v_{n+½}
        a_{n+1} = F(x_{n+1}) / m
        v_{n+1} = v_{n+½} + (dt/2) · a_{n+1}

    Gold standard for oscillators, orbits, springs.
    """
    vhx = vx + 0.5 * dt * ax
    vhy = vy + 0.5 * dt * ay
    npx = px + dt * vhx
    npy = py + dt * vhy
    fax, fay = force_fn(npx, npy, vhx, vhy)
    nax, nay = fax / mass, fay / mass
    nvx = vhx + 0.5 * dt * nax
    nvy = vhy + 0.5 * dt * nay
    return npx, npy, nvx, nvy, nax, nay


def semi_implicit_euler(px, py, vx, vy, dt, force_fn, mass):
    """
    Semi-implicit Euler — standard in Box2D, Jolt, most game engines.

        v^{n+1} = v^n + dt · a^n
        x^{n+1} = x^n + dt · v^{n+1}   (uses NEW velocity)
    """
    fax, fay = force_fn(px, py, vx, vy)
    ax, ay = fax / mass, fay / mass
    nvx = vx + dt * ax
    nvy = vy + dt * ay
    npx = px + dt * nvx
    npy = py + dt * nvy
    return npx, npy, nvx, nvy, ax, ay


def rk4(px, py, vx, vy, dt, force_fn, mass):
    """
    Classical 4th-order Runge-Kutta.

        k1 = f(t, u)
        k2 = f(t + dt/2, u + dt/2 · k1)
        k3 = f(t + dt/2, u + dt/2 · k2)
        k4 = f(t + dt, u + dt · k3)
        u_{n+1} = u + (dt/6)(k1 + 2k2 + 2k3 + k4)

    Most accurate per step; energy drift unbounded over long runs.
    """
    def d(cx, cy, cvx, cvy):
        fx, fy = force_fn(cx, cy, cvx, cvy)
        return cvx, cvy, fx / mass, fy / mass

    dx1, dy1, dvx1, dvy1 = d(px, py, vx, vy)
    dx2, dy2, dvx2, dvy2 = d(
        px + 0.5 * dt * dx1, py + 0.5 * dt * dy1,
        vx + 0.5 * dt * dvx1, vy + 0.5 * dt * dvy1,
    )
    dx3, dy3, dvx3, dvy3 = d(
        px + 0.5 * dt * dx2, py + 0.5 * dt * dy2,
        vx + 0.5 * dt * dvx2, vy + 0.5 * dt * dvy2,
    )
    dx4, dy4, dvx4, dvy4 = d(
        px + dt * dx3, py + dt * dy3,
        vx + dt * dvx3, vy + dt * dvy3,
    )

    npx = px + (dt / 6) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
    npy = py + (dt / 6) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
    nvx = vx + (dt / 6) * (dvx1 + 2 * dvx2 + 2 * dvx3 + dvx4)
    nvy = vy + (dt / 6) * (dvy1 + 2 * dvy2 + 2 * dvy3 + dvy4)
    fx, fy = force_fn(npx, npy, nvx, nvy)
    return npx, npy, nvx, nvy, fx / mass, fy / mass


INTEGRATORS = {
    IntegratorType.VERLET: velocity_verlet,
    IntegratorType.SEMI_EULER: semi_implicit_euler,
    IntegratorType.RK4: rk4,
}


# ─── Force Models ───────────────────────────────────────────────

def gravity_force(mass):
    """F = m·g downward. Source: newtons-laws.md"""
    return 0.0, mass * GRAVITY


def well_gravity(x1, y1, m1, x2, y2, m2):
    """
    Universal gravitation: F = G·m1·m2 / r²
    Softened to prevent singularities.
    Source: newtons-laws.md
    """
    dx, dy = x2 - x1, y2 - y1
    dist_sq = dx * dx + dy * dy
    soft = 40.0
    if dist_sq < soft * soft:
        dist_sq = soft * soft
    dist = math.sqrt(dist_sq)
    mag = G_WELL_SCALE * m1 * m2 / dist_sq
    return mag * dx / dist, mag * dy / dist


def spring_force(x1, y1, x2, y2, vx1, vy1, vx2, vy2, rest, k, c):
    """
    Hooke's law + viscous damping:  F = -k·Δx - c·v
    Source: vibrations.md, springs.md

    Returns (f1x, f1y, f2x, f2y).
    """
    dx, dy = x2 - x1, y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return 0, 0, 0, 0
    nx, ny = dx / dist, dy / dist
    stretch = dist - rest
    rv = (vx2 - vx1) * nx + (vy2 - vy1) * ny
    f = k * stretch + c * rv
    return f * nx, f * ny, -f * nx, -f * ny


def drag_force(vx, vy, radius):
    """
    Aerodynamic drag: F_D = C_D · ½ρ|v|²A  opposing motion.
    Source: fluid-mechanics.md
    """
    spd_sq = vx * vx + vy * vy
    if spd_sq < 0.01:
        return 0.0, 0.0
    spd = math.sqrt(spd_sq)
    # Convert to SI, compute, convert back
    v_ms = spd / PX_PER_METER
    r_m = radius / PX_PER_METER
    area = math.pi * r_m * r_m
    f_n = DRAG_CD * 0.5 * AIR_DENSITY * (v_ms * v_ms) * area
    mag = f_n * PX_PER_METER  # N → pixel-force
    return -mag * vx / spd, -mag * vy / spd


# ─── Collision Resolution ───────────────────────────────────────
# Source: conservation-principles.md, dynamics.md

def resolve_ball_ball(a, b, restitution):
    """
    2D circle-circle collision via impulse method.

    Impulse magnitude:
        J = -(1+e)(v_rel · n̂) / (1/m₁ + 1/m₂)

    Post-collision velocities:
        v₁' = v₁ + (J/m₁)n̂
        v₂' = v₂ - (J/m₂)n̂
    """
    dx, dy = b.x - a.x, b.y - a.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return
    nx, ny = dx / dist, dy / dist
    dvn = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny
    if dvn < 0:
        return
    inv_mass = (0 if a.pinned else 1 / a.mass) + (0 if b.pinned else 1 / b.mass)
    if inv_mass == 0:
        return
    j = -(1 + restitution) * dvn / inv_mass
    if not a.pinned:
        a.vx += (j / a.mass) * nx
        a.vy += (j / a.mass) * ny
    if not b.pinned:
        b.vx -= (j / b.mass) * nx
        b.vy -= (j / b.mass) * ny
    overlap = (a.radius + b.radius) - dist
    if overlap > 0:
        corr = overlap * 0.5
        if not a.pinned:
            a.x -= corr * nx
            a.y -= corr * ny
        if not b.pinned:
            b.x += corr * nx
            b.y += corr * ny


def resolve_ball_wall(ball, wx, wy, ww, wh, restitution):
    """Resolve ball-AABB collision with restitution."""
    closest_x = max(wx, min(ball.x, wx + ww))
    closest_y = max(wy, min(ball.y, wy + wh))
    dx = ball.x - closest_x
    dy = ball.y - closest_y
    dist_sq = dx * dx + dy * dy
    if dist_sq > ball.radius * ball.radius or dist_sq < 1e-6:
        return
    dist = math.sqrt(dist_sq)
    nx, ny = dx / dist, dy / dist
    pen = ball.radius - dist
    if not ball.pinned:
        ball.x += pen * nx
        ball.y += pen * ny
        vn = ball.vx * nx + ball.vy * ny
        if vn < 0:
            ball.vx -= (1 + restitution) * vn * nx
            ball.vy -= (1 + restitution) * vn * ny


# ─── Energy Helpers ──────────────────────────────────────────────
# Source: conservation-principles.md

def kinetic_energy(mass, vx, vy):
    """T = ½mv²"""
    return 0.5 * mass * (vx * vx + vy * vy)


def grav_potential_energy(mass, y, floor_y):
    """V = mgh (screen-Y is inverted)"""
    return mass * GRAVITY * max(0, (floor_y - y) / PX_PER_METER)


def spring_potential_energy(stretch, k):
    """V = ½kx²"""
    return 0.5 * k * stretch * stretch


# ─── Pendulum Dynamics ──────────────────────────────────────────
# Source: dynamics.md (Lagrangian derivation)
#   L = ½mL²θ̇² + mgL cos θ
#   Euler-Lagrange → θ̈ = -(g/L) sin θ

def pendulum_accel(theta, omega, length_px, damping=0.02):
    """θ̈ = -(g/L)sin θ - c·θ̇"""
    g = GRAVITY
    L = max(length_px, 1)
    return -(g / L) * math.sin(theta) - damping * omega
