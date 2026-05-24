from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from diffrax import Dopri8, ODETerm, PIDController, SaveAt, diffeqsolve


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["theta1", "theta2", "p1", "p2"],
    meta_fields=[],
)
@dataclass
class DoublePendulum:
    theta1: jax.Array
    theta2: jax.Array
    p1: jax.Array
    p2: jax.Array

    @classmethod
    def from_point(cls, theta1: float, theta2: float):
        return cls(
            jnp.asarray(theta1),
            jnp.asarray(theta2),
            jnp.zeros_like(theta1),
            jnp.zeros_like(theta2),
        )


@jax.jit
def system_derivative(state: DoublePendulum) -> DoublePendulum:
    delta = state.theta1 - state.theta2
    cos_delta, sin_delta = jnp.cos(delta), jnp.sin(delta)
    D = 16 - 9 * cos_delta**2
    theta_dot1 = 6 / D * (2 * state.p1 - 3 * cos_delta * state.p2)
    theta_dot2 = 6 / D * (8 * state.p2 - 3 * cos_delta * state.p1)
    result = DoublePendulum(
        theta_dot1,
        theta_dot2,
        -0.5 * (3 * jnp.sin(state.theta1) + theta_dot1 * theta_dot2 * sin_delta),
        -0.5 * (jnp.sin(state.theta2) - theta_dot1 * theta_dot2 * sin_delta),
    )
    return result


@jax.jit(static_argnames=("t_sample", "h0"))
def solve_pendulum(pendulum, t_sample, h0):

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
        max_steps=int(t_sample[-1] / h0)
    )
    return sol1


@jax.jit(static_argnames=("eps",))
def calculate_lyapunov_exponents(t_sample, sol1, sol2, eps):
    euclidean_err = jnp.hypot(
        sol1.ys.theta1 - sol2.ys.theta1, sol1.ys.theta2 - sol2.ys.theta2
    )
    lyap_exp = (
        jnp.einsum("t,xyt->xy", t_sample, jnp.log(euclidean_err) - np.log(eps))
        / jnp.square(t_sample).sum()
    )
    return lyap_exp