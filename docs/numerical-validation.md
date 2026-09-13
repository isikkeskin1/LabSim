# Numerical validation

Numerical simulation is useful only when the implementation can be checked against something independent of the implementation itself. LabSim therefore treats validation as part of the core workflow.

## Exact-solution checks

When a model has a closed-form solution, `max_state_error` compares every numerical sample with that reference and reports the largest Euclidean state error.

The harmonic oscillator is the first reference problem because `x(t) = cos(ωt)` and `v(t) = -ω sin(ωt)` for the unit-amplitude, zero-velocity initial condition at `ω = 1`.

## Convergence checks

`run_convergence_study` repeats the same experiment at several step sizes and estimates the observed order from the error ratio. For a stable smooth problem, halving the step size should reduce the leading truncation error according to the method's order.

LabSim currently uses first-order Euler and fourth-order classical RK4. The convergence test therefore acts as a regression detector for the expected RK4 behavior rather than relying on a single tolerance at one step size.

## Event localization

Threshold events can be detected from sampled trajectories with `first_crossing`. When the crossing occurs between samples, `first_crossing_linear` linearly interpolates both time and state between the bracketing samples. This improves reporting precision without pretending that the interpolated point is an additional integration step.

For high-accuracy event-driven integration, a future solver API can expose dense output or bracketed root finding directly from integration stages.

## Physical invariants

Not every system has an analytical solution. Physical invariants provide another independent check. For example, an ideal harmonic oscillator conserves mechanical energy, while the normalized SIR equations conserve the total population fraction.

These checks should be kept separate from the model implementation where practical. A model that verifies itself using the exact same calculation path can hide shared bugs.

## Regression strategy

Every new solver or model should ideally receive at least one of:

- an exact-reference test;
- an invariant or conservation test;
- a limiting-case test;
- a dimension/parameter validation test;
- a convergence or sensitivity regression;
- an event-localization regression when the model has meaningful thresholds.

This makes numerical changes reviewable and keeps the library from silently becoming less accurate as features are added.
