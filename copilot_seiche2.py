# lagoon_1d_wind_seiche.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field

# ----------------------------
# Physical & model parameters
# ----------------------------
@dataclass
class LagoonParams:
    L: float = 10_000.0        # lagoon length [m]
    h: float = 2.0             # uniform depth [m]
    g: float = 9.81            # gravity [m/s^2]
    rho_w: float = 1000.0      # water density [kg/m^3]
    rho_a: float = 1.225       # air density [kg/m^3]
    Cd: float = 1.3e-3         # wind drag coefficient [-]
    r: float = 5.0e-4          # linear bottom friction [1/s] (~0.0005 s^-1)
    nx: int = 201              # number of eta points (cell centers)
    dt: float = 5.0            # time step [s]
    t_total: float = 6*3600    # total simulation time [s] (6 hours)

    # Boundaries: "closed" (u=0) or "open" (specified eta at ends)
    bc_type: str = "closed"

    # Optional open-boundary level time series functions (if bc_type="open")
    eta_left_fn: callable = None   # function t->eta_left(t) [m]
    eta_right_fn: callable = None  # function t->eta_right(t) [m]

    # Wind: function returning U10(t) [m/s], along +x (eastward). Positive winds push water to +x.
    U10_fn: callable = None

    # Derived (filled later)
    dx: float = field(init=False)
    x_eta: np.ndarray = field(init=False)   # centers for eta
    x_u: np.ndarray = field(init=False)     # edges for u

    def __post_init__(self):
        self.dx = self.L / (self.nx - 1)
        self.x_eta = np.linspace(0, self.L, self.nx)                # eta at cell centers
        self.x_u   = np.linspace(0, self.L, self.nx)                # u at cell edges (same count; endpoints used for boundaries)
        # CFL safety check
        c = np.sqrt(self.g * self.h)
        cfl = c * self.dt / self.dx
        if cfl > 1.0:
            raise ValueError(f"Unstable CFL={cfl:.2f} > 1. Reduce dt or increase dx.")
        # Default wind if not provided: zero
        if self.U10_fn is None:
            self.U10_fn = lambda t: 0.0

# ----------------------------
# Wind stress helpers
# ----------------------------
def wind_to_stress(U10, rho_a=1.225, Cd=1.3e-3):
    """Convert 10-m wind speed to along-lagoon wind stress τ [N/m^2]."""
    return rho_a * Cd * U10 * abs(U10)

# Scenario builders per user request
def make_wind_scenario_steady_then_off(T_total, T_steady, U_mag):
    """U(t): +U_mag until T_steady, then 0."""
    def U10(t):
        return U_mag if t < T_steady else 0.0
    return U10

def make_wind_scenario_steady_then_reverse(T_total, T_steady, T_reverse, U_mag):
    """U(t): +U_mag until T_steady, then -U_mag until T_reverse, then 0."""
    def U10(t):
        if t < T_steady:
            return U_mag
        elif t < T_reverse:
            return -U_mag
        else:
            return 0.0
    return U10

# ----------------------------
# Model integrator (C-grid)
# ----------------------------
def run_lagoon(params: LagoonParams):
    """
    Forward–backward explicit time stepping on an Arakawa C-grid (eta at centers, u at edges).
    Returns time array, eta[t, x], u[t, x].
    """
    nx = params.nx
    nt = int(np.round(params.t_total / params.dt)) + 1
    t = np.linspace(0, params.t_total, nt)

    # State
    eta = np.zeros((nt, nx))   # [m] centers: indices 0..nx-1
    u   = np.zeros((nt, nx))   # [m/s] edges: indices 0..nx-1; 0 and nx-1 are walls for closed bc

    g, h, rho_w, r = params.g, params.h, params.rho_w, params.r
    dx, dt = params.dx, params.dt

    # Precompute open boundary level series if requested
    eta_left_ts = None
    eta_right_ts = None
    if params.bc_type == "open":
        if params.eta_left_fn is None or params.eta_right_fn is None:
            raise ValueError("For bc_type='open', provide eta_left_fn and eta_right_fn (t -> η).")
        eta_left_ts  = np.array([params.eta_left_fn(tt) for tt in t])
        eta_right_ts = np.array([params.eta_right_fn(tt) for tt in t])

    # Main loop
    for n in range(nt - 1):
        # Wind stress (uniform in x)
        U10 = params.U10_fn(t[n])
        tau = wind_to_stress(U10, rho_a=params.rho_a, Cd=params.Cd)  # [N/m^2]
        accel_wind = tau / (rho_w * h)                               # [m/s^2]

        # 1) Momentum: update u^{n+1} from eta^n
        # Interior edges j = 1..nx-2 (count nx-2): grad_eta = (eta[j] - eta[j-1]) / dx
        grad_eta = (eta[n, 1:nx-1] - eta[n, 0:nx-2]) / dx            # length nx-2
        u_next = u[n].copy()

        # Update interior edges
        u_next[1:-1] = (u[n, 1:-1]
                        - dt * g * grad_eta
                        - dt * r * u[n, 1:-1]
                        + dt * accel_wind)

        # Boundary edges:
        if params.bc_type == "closed":
            # Reflecting: no normal flow at walls
            u_next[0]  = 0.0
            u_next[-1] = 0.0
        else:
            # Open: one-sided pressure gradients to be consistent with Dirichlet eta at boundaries
            grad_left  = (eta[n, 0]   - eta_left_ts[n])  / dx
            grad_right = (eta_right_ts[n] - eta[n, -1]) / dx
            u_next[0]  = (u[n, 0]  - dt * g * grad_left  - dt * r * u[n, 0]  + dt * accel_wind)
            u_next[-1] = (u[n, -1] - dt * g * grad_right - dt * r * u[n, -1] + dt * accel_wind)

        # 2) Continuity: update eta^{n+1} from u^{n+1}
        # Interior centers i = 1..nx-2 (count nx-2): div = (u[i+1] - u[i]) / dx
        div_centers = (u_next[2:] - u_next[1:-1]) / dx               # length nx-2
        eta_next = eta[n].copy()
        eta_next[1:-1] -= dt * h * div_centers

        # Open boundary Dirichlet levels (override)
        if params.bc_type == "open":
            eta_next[0]  = eta_left_ts[n+1]
            eta_next[-1] = eta_right_ts[n+1]

        # Save
        u[n+1]   = u_next
        eta[n+1] = eta_next

    return t, params.x_eta, params.x_u, eta, u
# ----------------------------
# Animation utilities
# ----------------------------
def animate_eta_u(x_eta, x_u, t, eta, u, title="Lagoon η & u", interval_ms=40, save_path=None):
    """
    Animate η(x,t) and u(x,t) (line plots).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    line_eta, = ax1.plot(x_eta, eta[0], color='navy')
    ax1.set_ylabel("η [m]")
    ax1.grid(True)
    ax1.set_title(title)

    line_u, = ax2.plot(x_u, u[0], color='darkred')
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("u [m/s]")
    ax2.grid(True)

    # Axis limits for stability across frames
    eta_max = max(abs(eta.min()), abs(eta.max()))
    u_max   = max(abs(u.min()), abs(u.max()))
    eta_pad = 0.05 if eta_max == 0 else 0.1 * eta_max
    u_pad   = 0.05 if u_max == 0 else 0.1 * u_max
    ax1.set_ylim(-eta_max - eta_pad, eta_max + eta_pad)
    ax2.set_ylim(-u_max - u_pad,     u_max + u_pad)

    def update(frame):
        line_eta.set_ydata(eta[frame])
        line_u.set_ydata(u[frame])
        ax2.set_title(f"t = {t[frame]/3600:.2f} h")
        return line_eta, line_u

    ani = FuncAnimation(fig, update, frames=len(t), interval=interval_ms, blit=True)
    if save_path:
        try:
            ani.save(save_path, writer='ffmpeg', fps=int(1000/interval_ms))
            print(f"Saved animation to: {save_path}")
        except Exception as e:
            print(f"Could not save MP4 (ffmpeg missing?). Showing inline/window. Error: {e}")
            plt.show()
    else:
        plt.show()

# ----------------------------
# Driver for the two scenarios
# ----------------------------
if __name__ == "__main__":
    # Base parameters (tweak as needed)
    L = 10_000.0   # [m] lagoon length ~10 km
    h = 2.0        # [m] depth
    nx = 201
    dt = 5.0       # [s]
    T_total = 6*3600  # [s] 6 hours
    r = 5e-4       # [1/s] linear bottom friction
    U_mag = 10.0   # [m/s] wind speed magnitude

    # Fundamental seiche period (closed basin, mode 1): T1 = 2L/c
    c = np.sqrt(9.81 * h)
    T1 = 2 * L / c
    print(f"Fundamental seiche: c = {c:.2f} m/s, T1 = {T1/3600:.2f} h")

    # Choose steady duration ~ 1–2 T1 to approach quasi-steady setup
    T_steady = 1.5 * T1         # wind-on duration until drop/reversal
    T_reverse = 2.0 * T1        # end of reversed wind in scenario 2

    # Scenario 1: steady then off
    U10_fn1 = make_wind_scenario_steady_then_off(T_total=T_total, T_steady=T_steady, U_mag=U_mag)
    params1 = LagoonParams(L=L, h=h, nx=nx, dt=dt, t_total=T_total, r=r, bc_type="closed", U10_fn=U10_fn1)
    t1, x_eta1, x_u1, eta1, u1 = run_lagoon(params1)

    # Scenario 2: steady then reverse then off
    U10_fn2 = make_wind_scenario_steady_then_reverse(T_total=T_total, T_steady=T_steady, T_reverse=T_reverse, U_mag=U_mag)
    params2 = LagoonParams(L=L, h=h, nx=nx, dt=dt, t_total=T_total, r=r, bc_type="closed", U10_fn=U10_fn2)
    t2, x_eta2, x_u2, eta2, u2 = run_lagoon(params2)

    # Animate
    animate_eta_u(x_eta1, x_u1, t1, eta1, u1, title="Scenario 1: Steady wind → Off (seiche)", save_path=None)
    animate_eta_u(x_eta2, x_u2, t2, eta2, u2, title="Scenario 2: Steady wind → Reverse → Off", save_path=None)

    # Optional: save MP4s (uncomment if ffmpeg available)
    # animate_eta_u(x_eta1, x_u1, t1, eta1, u1, title="Scenario 1: Steady → Off", save_path="scenario1.mp4")
    # animate_eta_u(x_eta2, x_u2, t2, eta2, u2, title="Scenario 2: Steady → Reverse → Off", save_path="scenario2.mp4")
