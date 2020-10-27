import numpy as np
import matplotlib.pyplot as pl


def main():
    D = 1  # diffusion coefficient
    nx = 100  # number of space steps
    dx = 1 / nx  # space step
    dx2 = dx * dx
    dt = 0.5 * dx2 / D  # time step
    n = int(1 / dt) + 1  # number of time steps
    alpha = D * dt / dx2  # CFL

    T = np.zeros((n, nx))  # Unknown temperature, each row is T at time t
    T[0, 0] = 1  # Initial condition

    for i in range(n-1):
        # Boundary conditions
        T[i + 1, 0] = 1
        T[i + 1, nx-1] = 0

        # Scheme: finite differences
        for k in range(1, nx-1):
            T[i+1, k] = T[i, k] + alpha * (T[i, k+1] - 2*T[i, k] + T[i, k-1])

    # Plot
    x = np.linspace(0,1,nx)
    for t in [0, 20, 200, 2000, n-1]:
        pl.plot(x, T[t,:], label=f"$t={t*dt}s$")
    pl.legend()
    pl.xlabel("$x$")
    pl.ylabel("$T$")
    pl.title("Diffusion equation")
    pl.show()


if __name__=="__main__":
    main()
