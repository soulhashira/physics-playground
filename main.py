#!/usr/bin/env python3
"""
Physics Playground — Interactive 2D physics sandbox.

A game-like sandbox for experimenting with Newtonian mechanics,
collisions, springs, pendulums, gravity wells, and drag.

All underlying physics formulas are sourced from the deep-research
knowledge base (25 domain wikis covering mathematics, mechanical
engineering, computer science, and fluid mechanics).

Controls
--------
  [1] LAUNCH    Click & drag to slingshot-launch balls
  [2] SPRING    Click two balls to connect with a spring
  [3] PIN       Click a ball to pin it in place
  [4] WALL      Click & drag to create walls
  [5] GRAVITY   Click to place a gravity well

  Space         Pause / Resume
  R             Reset everything
  Tab           Cycle integrator (Verlet / Euler / RK4)
  G             Toggle gravity on/off
  D             Toggle air drag
  E             Cycle restitution (elastic → mixed → inelastic)
  T             Toggle velocity trails
  +/−           Adjust spawn radius
  Backspace     Clear all objects
  Esc / Q       Quit
"""

import sys
import math
import pygame

from physics import (
    IntegratorType,
    GRAVITY,
    gravity_force,
    drag_force,
    resolve_ball_ball,
    resolve_ball_wall,
    kinetic_energy,
    grav_potential_energy,
    spring_potential_energy,
)
from objects import Ball, Spring, Wall, GravityWell, Pendulum

# ─── Display ────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 60
DT = 1.0 / FPS
BG = (18, 18, 24)
GRID_COLOR = (30, 30, 40)
HUD_COLOR = (200, 200, 210)
MUTED = (100, 100, 120)
ACCENT = (80, 200, 255)
WARN = (255, 200, 60)
SPRING_COLOR = (120, 255, 160)
WALL_COLOR = (70, 70, 90)
WELL_INNER = (255, 120, 60)
WELL_OUTER = (60, 30, 15)
PIN_COLOR = (255, 255, 100)

# ─── Modes ──────────────────────────────────────────────────────

class Mode:
    LAUNCH = 1
    SPRING = 2
    PIN = 3
    WALL = 4
    GRAVITY_WELL = 5

MODE_NAMES = {
    Mode.LAUNCH: "LAUNCH",
    Mode.SPRING: "SPRING",
    Mode.PIN: "PIN",
    Mode.WALL: "WALL",
    Mode.GRAVITY_WELL: "GRAVITY",
}


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Physics Playground")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.big_font = pygame.font.SysFont("monospace", 20, bold=True)

        self.reset()

    def reset(self):
        self.balls: list[Ball] = []
        self.springs: list[Spring] = []
        self.walls: list[Wall] = []
        self.wells: list[GravityWell] = []
        self.pendulums: list[Pendulum] = []

        self.mode = Mode.LAUNCH
        self.integrator = IntegratorType.VERLET
        self.gravity_on = True
        self.drag_on = False
        self.restitution = 0.8
        self.show_trails = True
        self.paused = False
        self.spawn_radius = 15

        # Interaction state
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.spring_first: Ball | None = None
        self.wall_start = None

        # Add floor and side walls
        self._add_boundaries()

    def _add_boundaries(self):
        t = 10  # thickness
        self.walls.append(Wall(0, HEIGHT - t, WIDTH, t))       # floor
        self.walls.append(Wall(0, 0, t, HEIGHT))               # left
        self.walls.append(Wall(WIDTH - t, 0, t, HEIGHT))       # right
        self.walls.append(Wall(0, 0, WIDTH, t))                # ceiling

    # ─── Physics step ───────────────────────────────────────────

    def physics_step(self):
        if self.paused:
            return

        dt = DT
        balls = self.balls
        springs = self.springs
        wells = self.wells

        # Accumulate spring forces (pre-compute)
        spring_forces: dict[int, tuple[float, float]] = {}
        for sp in springs:
            f1x, f1y, f2x, f2y = sp.forces()
            aid, bid = id(sp.a), id(sp.b)
            sfx, sfy = spring_forces.get(aid, (0, 0))
            spring_forces[aid] = (sfx + f1x, sfy + f1y)
            sfx, sfy = spring_forces.get(bid, (0, 0))
            spring_forces[bid] = (sfx + f2x, sfy + f2y)

        # Step each ball
        for ball in balls:
            if ball.pinned:
                continue

            bid = id(ball)
            sfx, sfy = spring_forces.get(bid, (0, 0))

            def force_fn(px, py, vx, vy, _b=ball, _sfx=sfx, _sfy=sfy):
                fx, fy = 0.0, 0.0
                if self.gravity_on:
                    gx, gy = gravity_force(_b.mass)
                    fx += gx
                    fy += gy
                if self.drag_on:
                    dx, dy = drag_force(vx, vy, _b.radius)
                    fx += dx
                    fy += dy
                for w in wells:
                    wx, wy = w.force_on(px, py, _b.mass)
                    fx += wx
                    fy += wy
                fx += _sfx
                fy += _sfy
                return fx, fy

            ball.step(dt, force_fn, self.integrator)

            if self.show_trails:
                ball.record_trail()

        # Pendulum integration
        for p in self.pendulums:
            p.step(dt)
            if self.show_trails:
                p.bob.record_trail()

        # Ball-ball collisions O(n²) — fine for a sandbox
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                a, b = balls[i], balls[j]
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < a.radius + b.radius:
                    resolve_ball_ball(a, b, self.restitution)

        # Ball-wall collisions
        for ball in balls:
            for wall in self.walls:
                resolve_ball_wall(ball, wall.x, wall.y, wall.w, wall.h, self.restitution)

        # Pendulum bob vs walls
        for p in self.pendulums:
            for wall in self.walls:
                resolve_ball_wall(p.bob, wall.x, wall.y, wall.w, wall.h, self.restitution)

    # ─── Drawing ────────────────────────────────────────────────

    def draw(self):
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_walls()
        self._draw_springs()
        self._draw_wells()
        self._draw_pendulums()
        self._draw_balls()
        self._draw_interaction()
        self._draw_hud()
        pygame.display.flip()

    def _draw_grid(self):
        for x in range(0, WIDTH, 50):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 50):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

    def _draw_walls(self):
        for w in self.walls:
            pygame.draw.rect(self.screen, WALL_COLOR, (w.x, w.y, w.w, w.h))

    def _draw_springs(self):
        for sp in self.springs:
            # Draw zigzag line
            ax, ay = int(sp.a.x), int(sp.a.y)
            bx, by = int(sp.b.x), int(sp.b.y)
            dx, dy = bx - ax, by - ay
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1:
                continue
            nx, ny = dx / length, dy / length
            px, py = -ny, nx  # perpendicular

            segments = 16
            points = []
            for i in range(segments + 1):
                t = i / segments
                cx = ax + dx * t
                cy = ay + dy * t
                if 0 < i < segments:
                    offset = 8 * (1 if i % 2 == 0 else -1)
                    cx += px * offset
                    cy += py * offset
                points.append((int(cx), int(cy)))

            stretch_ratio = abs(sp.stretch) / max(sp.rest_length, 1)
            if stretch_ratio > 0.5:
                color = (255, 100, 100)
            elif stretch_ratio > 0.2:
                color = (255, 200, 100)
            else:
                color = SPRING_COLOR
            pygame.draw.lines(self.screen, color, False, points, 2)

    def _draw_wells(self):
        for w in self.wells:
            wx, wy = int(w.x), int(w.y)
            # Concentric rings for glow (no alpha surfaces)
            for r in range(50, 10, -5):
                t = 1.0 - (r - 10) / 40
                c = (int(WELL_OUTER[0] + (WELL_INNER[0] - WELL_OUTER[0]) * t),
                     int(WELL_OUTER[1] + (WELL_INNER[1] - WELL_OUTER[1]) * t),
                     int(WELL_OUTER[2] + (WELL_INNER[2] - WELL_OUTER[2]) * t))
                pygame.draw.circle(self.screen, c, (wx, wy), r, 2)
            pygame.draw.circle(self.screen, WELL_INNER, (wx, wy), w.radius)
            pygame.draw.circle(self.screen, (255, 200, 150), (wx, wy), 6)

    def _draw_pendulums(self):
        for p in self.pendulums:
            ax, ay = int(p.anchor_x), int(p.anchor_y)
            bx, by = int(p.bob.x), int(p.bob.y)
            pygame.draw.line(self.screen, MUTED, (ax, ay), (bx, by), 2)
            pygame.draw.circle(self.screen, PIN_COLOR, (ax, ay), 5)
            if self.show_trails:
                self._draw_trail(p.bob)
            pygame.draw.circle(self.screen, p.bob.color, (bx, by), p.bob.radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (bx, by), p.bob.radius, 1)

    def _draw_trail(self, ball):
        trail = ball.trail
        n = len(trail)
        if n < 2:
            return
        bc = ball.base_color
        for i in range(1, n):
            t = i / n
            r = int(bc[0] * t)
            g = int(bc[1] * t)
            b = int(bc[2] * t)
            pygame.draw.line(
                self.screen, (r, g, b),
                (int(trail[i - 1][0]), int(trail[i - 1][1])),
                (int(trail[i][0]), int(trail[i][1])),
                max(1, int(3 * t)),
            )

    def _draw_balls(self):
        for ball in self.balls:
            bx, by = int(ball.x), int(ball.y)
            if self.show_trails:
                self._draw_trail(ball)
            pygame.draw.circle(self.screen, ball.color, (bx, by), ball.radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (bx, by), ball.radius, 1)
            if ball.pinned:
                pygame.draw.circle(self.screen, PIN_COLOR, (bx, by), 4)

    def _draw_interaction(self):
        mx, my = pygame.mouse.get_pos()

        if self.mode == Mode.LAUNCH and self.dragging and self.drag_start:
            sx, sy = self.drag_start
            # Draw slingshot line
            pygame.draw.line(self.screen, ACCENT, (sx, sy), (mx, my), 2)
            # Predicted velocity arrow
            dvx, dvy = (sx - mx) * 3, (sy - my) * 3
            speed = math.sqrt(dvx * dvx + dvy * dvy)
            label = self.font.render(f"{speed:.0f} px/s", True, ACCENT)
            self.screen.blit(label, (mx + 15, my - 10))
            # Preview circle
            pygame.draw.circle(self.screen, ACCENT, (sx, sy), self.spawn_radius, 1)

        elif self.mode == Mode.SPRING and self.spring_first:
            pygame.draw.line(self.screen, SPRING_COLOR,
                             (int(self.spring_first.x), int(self.spring_first.y)),
                             (mx, my), 1)

        elif self.mode == Mode.WALL and self.wall_start:
            wx, wy = self.wall_start
            rw, rh = mx - wx, my - wy
            if rw != 0 and rh != 0:
                rect = pygame.Rect(min(wx, mx), min(wy, my), abs(rw), abs(rh))
                pygame.draw.rect(self.screen, WALL_COLOR, rect, 2)

        elif self.mode == Mode.GRAVITY_WELL:
            pygame.draw.circle(self.screen, (*WELL_INNER, 80), (mx, my), 18, 2)

    def _draw_hud(self):
        # Energy totals
        total_ke = sum(kinetic_energy(b.mass, b.vx, b.vy) for b in self.balls)
        total_ke += sum(kinetic_energy(p.bob.mass, p.bob.vx, p.bob.vy) for p in self.pendulums)
        floor_y = HEIGHT - 10
        total_pe = sum(grav_potential_energy(b.mass, b.y, floor_y) for b in self.balls) if self.gravity_on else 0
        total_spe = sum(spring_potential_energy(s.stretch, s.k) for s in self.springs)
        total_e = total_ke + total_pe + total_spe

        lines = [
            f"MODE: {MODE_NAMES[self.mode]}  [1-5]",
            f"Integrator: {self.integrator.label}  [Tab]",
            f"",
            f"Gravity: {'ON' if self.gravity_on else 'OFF'}  [G]",
            f"Drag: {'ON' if self.drag_on else 'OFF'}  [D]",
            f"Restitution: {self.restitution:.1f}  [E]",
            f"Trails: {'ON' if self.show_trails else 'OFF'}  [T]",
            f"Spawn radius: {self.spawn_radius}  [+/-]",
            f"",
            f"Balls: {len(self.balls)}  Springs: {len(self.springs)}",
            f"Wells: {len(self.wells)}  Pendulums: {len(self.pendulums)}",
            f"",
            f"KE: {total_ke:>10.1f}",
            f"PE: {total_pe:>10.1f}",
            f"Spring PE: {total_spe:>10.1f}",
            f"Total E: {total_e:>10.1f}",
            f"",
            f"FPS: {self.clock.get_fps():.0f}",
        ]

        if self.paused:
            pause_text = self.big_font.render("PAUSED", True, WARN)
            self.screen.blit(pause_text, (WIDTH // 2 - 40, 10))

        x, y = 14, 14
        # Background panel
        panel_h = len(lines) * 17 + 10
        panel = pygame.Surface((220, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        self.screen.blit(panel, (x - 6, y - 4))

        for line in lines:
            color = ACCENT if line.startswith("MODE") else HUD_COLOR
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (x, y))
            y += 17

        # Bottom help bar
        help_text = "Space:Pause  R:Reset  Backspace:Clear  Q:Quit"
        help_surf = self.font.render(help_text, True, MUTED)
        self.screen.blit(help_surf, (WIDTH // 2 - help_surf.get_width() // 2, HEIGHT - 22))

    # ─── Input ──────────────────────────────────────────────────

    def find_ball_at(self, mx, my) -> Ball | None:
        """Find the nearest ball under cursor."""
        for ball in self.balls:
            dx, dy = ball.x - mx, ball.y - my
            if dx * dx + dy * dy <= ball.radius * ball.radius:
                return ball
        return None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_TAB:
                    types = list(IntegratorType)
                    idx = types.index(self.integrator)
                    self.integrator = types[(idx + 1) % len(types)]
                elif event.key == pygame.K_g:
                    self.gravity_on = not self.gravity_on
                elif event.key == pygame.K_d:
                    self.drag_on = not self.drag_on
                elif event.key == pygame.K_e:
                    if self.restitution >= 0.9:
                        self.restitution = 0.5
                    elif self.restitution >= 0.4:
                        self.restitution = 0.0
                    else:
                        self.restitution = 1.0
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                    if not self.show_trails:
                        for b in self.balls:
                            b.trail.clear()
                        for p in self.pendulums:
                            p.bob.trail.clear()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.spawn_radius = min(50, self.spawn_radius + 3)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.spawn_radius = max(5, self.spawn_radius - 3)
                elif event.key == pygame.K_BACKSPACE:
                    self.balls.clear()
                    self.springs.clear()
                    self.wells.clear()
                    self.pendulums.clear()
                    self._add_boundaries()
                elif event.key == pygame.K_1:
                    self.mode = Mode.LAUNCH
                elif event.key == pygame.K_2:
                    self.mode = Mode.SPRING
                elif event.key == pygame.K_3:
                    self.mode = Mode.PIN
                elif event.key == pygame.K_4:
                    self.mode = Mode.WALL
                elif event.key == pygame.K_5:
                    self.mode = Mode.GRAVITY_WELL

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                self._on_click(mx, my)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mx, my = event.pos
                self._on_release(mx, my)

        return True

    def _on_click(self, mx, my):
        if self.mode == Mode.LAUNCH:
            self.dragging = True
            self.drag_start = (mx, my)

        elif self.mode == Mode.SPRING:
            ball = self.find_ball_at(mx, my)
            if ball:
                if self.spring_first is None:
                    self.spring_first = ball
                else:
                    if ball is not self.spring_first:
                        self.springs.append(Spring(self.spring_first, ball))
                    self.spring_first = None

        elif self.mode == Mode.PIN:
            ball = self.find_ball_at(mx, my)
            if ball:
                ball.pinned = not ball.pinned

        elif self.mode == Mode.WALL:
            self.wall_start = (mx, my)

        elif self.mode == Mode.GRAVITY_WELL:
            self.wells.append(GravityWell(mx, my))

    def _on_release(self, mx, my):
        if self.mode == Mode.LAUNCH and self.dragging:
            sx, sy = self.drag_start
            vx = (sx - mx) * 3
            vy = (sy - my) * 3
            ball = Ball(sx, sy, radius=self.spawn_radius, vx=vx, vy=vy)
            self.balls.append(ball)
            self.dragging = False
            self.drag_start = None

        elif self.mode == Mode.WALL and self.wall_start:
            wx, wy = self.wall_start
            rw, rh = mx - wx, my - wy
            if abs(rw) > 5 and abs(rh) > 5:
                x = min(wx, mx)
                y = min(wy, my)
                self.walls.append(Wall(x, y, abs(rw), abs(rh)))
            self.wall_start = None

    # ─── Pendulum creation (right-click) ────────────────────────

    def handle_right_click(self):
        """Right-click always creates a pendulum regardless of mode."""
        buttons = pygame.mouse.get_pressed()
        if not buttons[2]:
            return
        # Debounce
        if not hasattr(self, '_rc_down'):
            self._rc_down = False
        if buttons[2] and not self._rc_down:
            self._rc_down = True
            mx, my = pygame.mouse.get_pos()
            length = 120 + (my % 80)
            theta0 = 0.8 * (1 if mx < WIDTH // 2 else -1)
            self.pendulums.append(Pendulum(mx, my, length, theta0))
        if not buttons[2]:
            self._rc_down = False

    # ─── Main loop ──────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.handle_right_click()
            self.physics_step()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
