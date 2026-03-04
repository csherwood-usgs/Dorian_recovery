import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
g = 9.81          # gravity (m/s^2)
rho = 1025.0      # density (kg/m^3)
h = 5.0           # uniform depth (m)
L = 5000.0        # lagoon length (m)
nx = 201          # number of eta points
r = 1e-4          # linear friction (1/s), small damping

# Wind stress magnitude (N/m^2)
tau0 = 0.1

# Time settings
t_end = 20000.0   # total simulation time (s)
c = np.sqrt(g * h)
dx = L / (nx - 1)
cfl = 0.5
dt = cfl * dx / c
nt = int(t_end / dt)

print(f"dx = {dx:.2f} m, dt = {dt:.2f} s, nt = {nt}")

# Grid
x_eta = np.linspace(0, L, nx)          # eta at cell centers
x_u = 0.5 * (x_eta[:-1] + x_eta[1:])   # u at cell faces
nu = nx - 1

# -----------------------------
# Wind forcing time series
# -----------------------------
def wind_case1(t, t_steady=6000.0):
    """
    Case 1: steady wind until t_steady, then off.
    """
    return tau0 if t < t_steady else 0.0

def wind_case2(t, t_steady=6000.0):
    """
    Case 2: steady wind until t_steady, then instant reversal.
    """
    return tau0 if t < t_steady else -tau0

# -----------------------------
# Solver function
# -----------------------------
def run_simulation(wind_func, store_every=10):
    """
    Run the 1D shallow-water model with a given wind forcing function.

    Returns:
        times: array of stored times
        eta_store: 2D array [ntime, nx]
    """
    # Initial conditions: flat surface, no flow
    eta = np.zeros(nx)
    u = np.zeros(nu)

    eta_store = []
    times = []

    for n in range(nt):
        t = n * dt

        # Wind stress (uniform in space)
        tau_w = wind_func(t)

        # ---- Update u (momentum) ----
        # du/dt = -g dη/dx + τ_w/(ρ h) - r u
        deta_dx = (eta[1:] - eta[:-1]) / dx
        du_dt = -g * deta_dx + tau_w / (rho * h) - r * u
        u_new = u + dt * du_dt

        # Boundary conditions: closed ends => u = 0 at boundaries
        u_new[0] = 0.0
        u_new[-1] = 0.0

        # ---- Update eta (continuity) ----
        # dη/dt = - d(hu)/dx
        hu = h * u_new
        dhu_dx = (hu[1:] - hu[:-1]) / dx
        deta_dt = -dhu_dx
        eta_new = eta.copy()
        eta_new[1:-1] += dt * deta_dt  # interior points

        # (Optional) weakly enforce zero net volume change by removing mean
        eta_new -= np.mean(eta_new)

        # Update state
        eta = eta_new
        u = u_new

        # Store
        if n % store_every == 0:
            eta_store.append(eta.copy())
            times.append(t)

    return np.array(times), np.array(eta_store)

# -----------------------------
# Run both cases
# -----------------------------
print("Running case 1 (wind off)...")
times1, eta1 = run_simulation(wind_case1, store_every=10)

print("Running case 2 (wind reversal)...")
times2, eta2 = run_simulation(wind_case2, store_every=10)

# -----------------------------
# Animation helper
# -----------------------------
def animate_case(x, times, eta_store, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(x, eta_store[0, :], lw=2)
    ax.set_xlim(0, L)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("η (m)")
    ax.set_title(title)

    # Set a reasonable y-limits based on data
    eta_max = np.max(np.abs(eta_store))
    ax.set_ylim(-1.1 * eta_max, 1.1 * eta_max)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        line.set_ydata(eta_store[frame, :])
        time_text.set_text(f"t = {times[frame]:.0f} s")
        return line, time_text

    anim = FuncAnimation(fig, update, frames=len(times), interval=40, blit=True)
    plt.tight_layout()
    plt.show()
    return anim

# -----------------------------
# Make animations
# -----------------------------
anim1 = animate_case(x_eta, times1, eta1, "Case 1: Steady wind, then wind off (seiche)")
anim2 = animate_case(x_eta, times2, eta2, "Case 2: Steady wind, then wind reversal (wind + seiche)")