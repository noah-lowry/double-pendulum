from fractions import Fraction

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from diffrax import Dopri8, ODETerm, PIDController, SaveAt, diffeqsolve

from pendulum import DoublePendulum, system_derivative

jax.config.update("jax_enable_x64", True)
matplotlib.use("QtAgg")

h0 = 0.01
t_max = 100

size = 300
num_sample = 9
eps = jnp.pi / size * 1e-3

run_name = f"{size}x{size}@{t_max}"

def calculate_lyapunov_exponents(sol1, sol2):
    ts = jnp.logspace(jnp.log(0.01) / jnp.log(1.2), jnp.log(t_max) / jnp.log(1.2), num_sample, base=1.2)
    euclidean_err = jnp.hypot(
        sol1.ys.theta1 - sol2.ys.theta1, sol1.ys.theta2 - sol2.ys.theta2
    )
    lyap_exp = (
        jnp.einsum("t,xyt->xy", ts, jnp.log(euclidean_err) - jnp.log(eps))
        / jnp.square(ts).sum()
    )
    return lyap_exp

def render(arr, filename=None):
    if filename is not None:
        dpi = 100
        fig = plt.figure(figsize=(arr.shape[1] / dpi, arr.shape[0] / dpi), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(
            arr,
            cmap="turbo",
            extent=(-jnp.pi, jnp.pi, -jnp.pi, jnp.pi),
            aspect="equal",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_axis_off()
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        fig.savefig(filename, dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        return

    num_axis_pts = 7

    fig, ax = plt.subplots()
    fig.tight_layout()

    ax.imshow(
        arr,
        cmap="turbo",
        extent=(-jnp.pi, jnp.pi, -jnp.pi, jnp.pi),
        aspect="equal",
        origin="lower",
        interpolation="none",
    )

    # TODO: make this axis shit better
    def get_tick_fmt(x, *_):
        axis_fractions_of_pi = [
            Fraction(2 * i, num_axis_pts - 1) - 1 for i in range(num_axis_pts)
        ]
        idx, _ = min(
            enumerate(abs(tick - x / jnp.pi) for tick in axis_fractions_of_pi),
            key=lambda t: t[1],
        )
        frac = axis_fractions_of_pi[idx]

        if frac.numerator == 0:
            return "$0$"
        elif frac.denominator == 1:
            if frac.numerator == 1:
                return r"$\pi$"
            elif frac.numerator == -1:
                return r"$-\pi$"
            return rf"${frac.numerator} \pi$"

        if frac.numerator == 1:
            return rf"$\frac{{\pi}}{{{frac.denominator}}}$"
        elif frac.numerator == -1:
            return rf"$\frac{{-\pi}}{{{frac.denominator}}}$"
        return rf"$\frac{{{frac.numerator} \pi}}{{{frac.denominator}}}$"

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.MultipleLocator(jnp.pi / 4))
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(get_tick_fmt))
    plt.show()


def solve_pendulum(pendulum):
    t_sample = jnp.logspace(jnp.log(0.01) / jnp.log(1.2), jnp.log(t_max) / jnp.log(1.2), num_sample, base=1.2)

    term = ODETerm(lambda t, y, args: system_derivative(y))
    solver = Dopri8()
    saveat = SaveAt(ts=t_sample)
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-8)

    sol1 = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_sample[-1],
        dt0=h0,
        y0=pendulum,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(t_max / h0)
    )
    return sol1


def main():
    theta1, theta2 = jnp.meshgrid(
        jnp.linspace(-jnp.pi, jnp.pi, size),
        jnp.linspace(-jnp.pi, jnp.pi, size),
        indexing="xy",
    )

    solve_all_fn = jax.vmap(jax.vmap(jax.jit(solve_pendulum)))

    pendulum_set1 = DoublePendulum(
        theta1 - eps / 2,
        theta2 - eps / 2,
        jnp.zeros_like(theta1),
        jnp.zeros_like(theta2),
    )
    sol1 = solve_all_fn(pendulum_set1)

    pendulum_set2 = DoublePendulum(
        theta1 + eps / 2,
        theta2 + eps / 2,
        jnp.zeros_like(theta1),
        jnp.zeros_like(theta2),
    )
    sol2 = solve_all_fn(pendulum_set2)

    lyap_exp = calculate_lyapunov_exponents(sol1, sol2)

    os.makedirs("outputs", exist_ok=True)
    np.savez_compressed(
        f"outputs/data_{run_name}.npz",
        theta1=theta1,
        theta2=theta2,
        lyap_exp=lyap_exp,
    )

    render(lyap_exp, filename=f"outputs/lyapunov_exponent_map_{run_name}.png")
    render(lyap_exp)


if __name__ == "__main__":
    main()
