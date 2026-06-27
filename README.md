# Physics Playground

Interactive 2D physics sandbox built with Python and pygame.

Experiment with Newtonian mechanics — launch balls, connect them with springs, create pendulums, place gravity wells, and watch the physics play out in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![pygame](https://img.shields.io/badge/pygame-2.5+-green)

## Features

- **Slingshot launcher** — click and drag to launch balls with adjustable velocity
- **Spring connections** — link balls with damped springs (Hooke's law + viscous damping)
- **Pendulums** — right-click to spawn nonlinear pendulums (exact sin(θ) equation)
- **Gravity wells** — place inverse-square attractors anywhere
- **Walls** — draw rectangular obstacles
- **Elastic & inelastic collisions** — adjustable coefficient of restitution
- **Air drag** — optional quadratic drag force
- **3 integrators** — Velocity Verlet, Semi-Implicit Euler, RK4
- **Energy readout** — live kinetic, potential, and spring energy tracking
- **Velocity trails** — visualize trajectories with speed-mapped colours

## Physics

All formulas are sourced from peer-reviewed references in applied mathematics, mechanical engineering, and computational physics:

| Feature | Formula | Domain |
|---------|---------|--------|
| Motion | F = ma (Newton's 2nd law) | Classical mechanics |
| Gravity | F = mg / F = Gm₁m₂/r² | Newtonian gravitation |
| Springs | F = −kx − cv | Hooke's law + damping |
| Collisions | J = −(1+e)(v_rel·n̂)/(1/m₁+1/m₂) | Impulse-momentum |
| Pendulum | θ̈ = −(g/L)sin θ − cθ̇ | Lagrangian mechanics |
| Drag | F_D = C_D·½ρv²A | Fluid mechanics |
| Verlet | v_{n+½}=v+½dt·a, x_{n+1}=x+dt·v_{n+½} | Symplectic integration |
| RK4 | u_{n+1} = u + (dt/6)(k₁+2k₂+2k₃+k₄) | Numerical methods |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| **1** | Launch mode — click & drag to slingshot balls |
| **2** | Spring mode — click two balls to connect |
| **3** | Pin mode — click a ball to pin/unpin |
| **4** | Wall mode — click & drag to create walls |
| **5** | Gravity mode — click to place gravity wells |
| **Right-click** | Spawn a pendulum (any mode) |
| **Space** | Pause / Resume |
| **R** | Reset everything |
| **Tab** | Cycle integrator (Verlet → Euler → RK4) |
| **G** | Toggle gravity |
| **D** | Toggle air drag |
| **E** | Cycle restitution (1.0 → 0.5 → 0.0) |
| **T** | Toggle velocity trails |
| **+/−** | Adjust ball spawn size |
| **Backspace** | Clear all objects |
| **Esc/Q** | Quit |

## License

MIT
