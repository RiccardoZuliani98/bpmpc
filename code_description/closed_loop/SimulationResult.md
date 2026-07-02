**Tags:** #class

## Overview

The `SimulationResult` class is a structured, JAX-compatible container implemented as a frozen dataclass for closed-loop simulation outputs.

## Purpose

After running a simulation, the resulting data is usually a mix of massive, compiled JAX arrays and simple native Python metadata. This class wraps the results of a simulation into a registered JAX PyTree. It seamlessly bridges the data gap by providing serialization utilities to easily save and load the nested data to and from disk while preserving the original PyTree structure.

## Key Attributes

* **`objective`**: The scalar target used for gradient-based optimization (e.g., via `jax.grad`).
* **`trajectory`**: The time-stacked history of the system (e.g., states, inputs, logs) accumulated over the simulation loop.
* **`final_state`**: The terminal carry state of the simulation at the last time step.
* **`metrics`**: Optional aggregated data, costs, or violation metrics computed during the finalize phase.

## Key Methods

* **`save(directory)`**: Iterates through the stored data, extracting arrays and compressing them into a single `trajectories.npz` file while preserving their nested structure via path-like keys (e.g., "trajectory/x"). Standard Python types (scalars, strings, nulls) and 0D scalars are saved to a `metadata.json` file in the same directory.
* **`load(directory)`**: A class method that reconstructs a `SimulationResult` from a saved directory. It loads standard arrays and scalar metadata, rebuilding the original PyTree structure. Arrays are loaded as standard NumPy arrays, which JAX can natively consume.
* **`tree_flatten()` & `tree_unflatten(aux_data, children)**`: JAX PyTree registration methods that flatten and reconstruct the dataclass to allow automatic differentiation and compilation tracking.

## Dependencies

*This class relies only on standard Python libraries (such as `json`, `pathlib`, and `dataclasses`), `numpy`, and `jax`.*