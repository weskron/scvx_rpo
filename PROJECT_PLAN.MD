# Rendezvous and Proximity Operation Simulation using SCVx

This is a rework plan to develop better infrastructure for doing convex optimization simulations. 

## Background

In the original implementation of SCVx for RPO the Field of View (FoV) constraint would not converge to the set point value. I could not determine why this was the case. This was due to a lack of utility testing and a lack of strong infrastructure. To correct these systemic problem I am proposing a multipart arcitecture update with the hope of creating something that works well. 

## Architecture Design

- `SpacecraftState` is the base class that can be initialized from Keplerian elements or a ECEF state vector. This class can internally convert state representations. This class will also hold the inertial properties and attitude information of the spacecraft. 

- `TrajectoryHistory` inherits `SpacecraftState` and store the time history of states in an easy to plot layout such that the time histories can easily be plotted for debugging. 

TODO: The dynamics are relative nonlinear dynamics for the orbits but not for the attitude data. I need to figure out how to best hold the state information for the reletive motion based on using these dynamics. 

TODO: Should the trajectory history include propagation? I think no but I need to figure out exactly why that is the case. how can I split up those functions and still get the desired result?

## Workflow Rules
- Branch name: `issue-<number>-description` 
- PR Description must contain `Closes #<issue-number>`.
- A Complete PR has: 
    1. Code merged
    2. `tests/` covering changes
    3. CI passes

## Epics and Issues
Each issue issue can be copied into a new github issue. 

### Epic A — Core State Classes
- [ ] A1. Define `SpacecraftState` — fields, constructors (Keplerian / ECEF), derived properties (orbital radius, specific angular momentum)
- [ ] A2. Keplerian to ECEF conversion + round-trip tests
- [ ] A3a. Target absolute propagation (two-body/perturbed dynamics, ECEF)
- [ ] A3b. Chaser nonlinear relative motion propagation (couples to target's instantaneous radius/rotation rate from A3a)
- [ ] A4. `RelativeState` class + absolute↔relative conversion functions (TODO: attitude included or translational-only — pending decision)
- [ ] A5. `TrajectoryHistory` class — efficient storage + inherits state
- [ ] A6. Plotting methods on `TrajectoryHistory`

### Epic B — Reference Trajectory Generation
- [ ] B1. Define reference trajectory generator interface
- [ ] B2. Implement reference trajectory generator
- [ ] B3. Tests: feasibility + boundary conditions

### Epic C — Convex Optimization Loop
- [ ] C1. Refactor opt loop to consume SpacecraftState/reference trajectories
- [ ] C2. Constraint diagnostics tool (the thing you actually needed last time)
- [ ] C3. Regression tests: known-good scenario stays satisfied

### Epic D — Testing & CI Infrastructure
- [ ] D1. tests/ structure + pytest config
- [ ] D2. GitHub Actions CI workflow
- [ ] D3. (optional) coverage reporting badge
