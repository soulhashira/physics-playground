"""
Game objects for Physics Playground.

Each object carries its own physics state and knows how to
accumulate forces and integrate its motion.
"""

import math
import random
from physics import (
    IntegratorType,
    INTEGRATORS,
    gravity_force,
    well_gravity,
    spring_force,
    drag_force,
    velocity_verlet,
    semi_implicit_euler,
    pendulum_accel,
)

# ─── Colour palette (speed-mapped) ──────────────────────────────

def speed_color(vx, vy, base_hue):
    """Map speed to a bright colour. Returns (r, g, b)."""
    speed = math.sqrt(vx * vx + vy * vy)
    t = min(speed / 800.0, 1.0)
    # Interpolate from base colour toward hot white
    r = int(base_hue[0] + (255 - base_hue[0]) * t)
    g = int(base_hue[1] + (255 - base_hue[1]) * t * 0.3)
    b = int(base_hue[2] * (1 - t * 0.6))
    return (min(255, r), min(255, g), max(0, b))


BALL_COLORS = [
    (80, 160, 255),
    (255, 100, 100),
    (100, 255, 140),
    (255, 200, 60),
    (200, 120, 255),
    (255, 140, 200),
    (100, 220, 220),
]


class Ball:
    """A circular rigid body with mass proportional to area."""

    __slots__ = (
        "x", "y", "vx", "vy", "ax", "ay",
        "radius", "mass", "pinned", "base_color",
        "trail",
    )

    def __init__(self, x, y, radius=15, vx=0, vy=0, pinned=False):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.ax, self.ay = 0.0, 0.0
        self.radius = radius
        self.mass = math.pi * radius * radius * 0.01  # area-proportional
        self.pinned = pinned
        self.base_color = random.choice(BALL_COLORS)
        self.trail: list[tuple[float, float]] = []

    @property
    def color(self):
        return speed_color(self.vx, self.vy, self.base_color)

    def record_trail(self, max_len=40):
        self.trail.append((self.x, self.y))
        if len(self.trail) > max_len:
            self.trail.pop(0)

    def step(self, dt, force_fn, integrator_type):
        if self.pinned:
            self.vx = self.vy = 0
            return
        if integrator_type == IntegratorType.VERLET:
            self.x, self.y, self.vx, self.vy, self.ax, self.ay = velocity_verlet(
                self.x, self.y, self.vx, self.vy, self.ax, self.ay,
                dt, force_fn, self.mass,
            )
        elif integrator_type == IntegratorType.RK4:
            fn = INTEGRATORS[IntegratorType.RK4]
            self.x, self.y, self.vx, self.vy, self.ax, self.ay = fn(
                self.x, self.y, self.vx, self.vy,
                dt, force_fn, self.mass,
            )
        else:
            self.x, self.y, self.vx, self.vy, self.ax, self.ay = semi_implicit_euler(
                self.x, self.y, self.vx, self.vy,
                dt, force_fn, self.mass,
            )


class Spring:
    """Connects two Balls with a spring + damper."""

    __slots__ = ("a", "b", "rest_length", "k", "c")

    def __init__(self, a: Ball, b: Ball, k=300.0, c=5.0, rest_length=None):
        self.a = a
        self.b = b
        if rest_length is None:
            dx, dy = b.x - a.x, b.y - a.y
            self.rest_length = math.sqrt(dx * dx + dy * dy)
        else:
            self.rest_length = rest_length
        self.k = k
        self.c = c

    @property
    def stretch(self):
        dx, dy = self.b.x - self.a.x, self.b.y - self.a.y
        return math.sqrt(dx * dx + dy * dy) - self.rest_length

    def forces(self):
        """Return (f1x, f1y, f2x, f2y)."""
        return spring_force(
            self.a.x, self.a.y, self.b.x, self.b.y,
            self.a.vx, self.a.vy, self.b.vx, self.b.vy,
            self.rest_length, self.k, self.c,
        )


class Wall:
    """Axis-aligned rectangular wall."""

    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x, self.y = x, y
        self.w, self.h = w, h


class GravityWell:
    """Attracts all balls via inverse-square law."""

    __slots__ = ("x", "y", "mass", "radius")

    def __init__(self, x, y, mass=500.0):
        self.x, self.y = float(x), float(y)
        self.mass = mass
        self.radius = 18

    def force_on(self, bx, by, bmass):
        return well_gravity(bx, by, bmass, self.x, self.y, self.mass)


class Pendulum:
    """
    Simple pendulum using exact nonlinear equation.
    Integrated with velocity Verlet on (theta, omega).
    """

    __slots__ = ("anchor_x", "anchor_y", "length", "theta", "omega", "alpha", "bob")

    def __init__(self, ax, ay, length, theta0=0.5):
        self.anchor_x = ax
        self.anchor_y = ay
        self.length = max(length, 20)
        self.theta = theta0
        self.omega = 0.0
        self.alpha = 0.0
        bx = ax + self.length * math.sin(self.theta)
        by = ay + self.length * math.cos(self.theta)
        self.bob = Ball(bx, by, radius=12, pinned=False)

    def step(self, dt):
        """Velocity Verlet on angular coordinates."""
        omega_half = self.omega + 0.5 * dt * self.alpha
        self.theta += dt * omega_half
        new_alpha = pendulum_accel(self.theta, omega_half, self.length)
        self.omega = omega_half + 0.5 * dt * new_alpha
        self.alpha = new_alpha
        self.bob.x = self.anchor_x + self.length * math.sin(self.theta)
        self.bob.y = self.anchor_y + self.length * math.cos(self.theta)
        self.bob.vx = self.length * self.omega * math.cos(self.theta)
        self.bob.vy = -self.length * self.omega * math.sin(self.theta)
