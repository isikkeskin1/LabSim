"""Demonstrate adaptive time stepping on exponential decay."""

import math

from labsim import integrate_adaptive, max_state_error


solution = integrate_adaptive(
    lambda _time, state: (-state[0],),
    (1.0,),
    duration=8.0,
    dt=0.5,
    rtol=1e-6,
    atol=1e-9,
    max_dt=0.5,
)

error = max_state_error(
    solution,
    lambda time, _state: (math.exp(-time),),
)

print(f"accepted samples: {len(solution)}")
print(f"final value: {solution.final_state[0]:.8e}")
print(f"maximum error: {error:.3e}")
