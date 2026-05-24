from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


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
