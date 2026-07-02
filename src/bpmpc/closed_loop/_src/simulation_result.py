import json
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Any, Optional

import numpy as np
import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimulationResult:
    """A structured, JAX-compatible container for closed-loop simulation outputs.
    
    This class wraps the results of a simulation (objective, trajectories, 
    terminal states, and optional metrics) into a registered JAX PyTree. It 
    includes utilities to easily serialize and deserialize the nested data 
    to and from disk while preserving the PyTree structure.
    
    Attributes
    ----------
    objective : Any
        The scalar target used for gradient-based optimization (e.g., via `jax.grad`).
    trajectory : Any
        The time-stacked history of the system (e.g., states, inputs, logs) 
        accumulated over the simulation loop.
    final_state : Any
        The terminal carry state of the simulation at the last time step.
    metrics : Any, optional
        Aggregated data, costs, or violation metrics computed during the 
        finalize phase. Defaults to None.
    """
    objective: Any       
    trajectory: Any      
    final_state: Any     
    metrics: Optional[Any] = None

    # =======================================================================
    # JAX PyTree Registration
    # =======================================================================
    def tree_flatten(self) -> tuple[tuple[Any, Any, Any, Any], None]:
        """Flattens the dataclass into a list of children for JAX."""
        # JAX tracks these fields for automatic differentiation and compilation
        children = (self.objective, self.trajectory, self.final_state, self.metrics)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "SimulationResult":
        """Reconstructs the dataclass from the flattened children."""
        return cls(*children)

    # =======================================================================
    # I/O Methods
    # =======================================================================
    def save(self, directory: Union[str, Path]) -> None:
        """Saves the simulation result to the specified directory.
        
        Arrays are extracted and compressed into a single `trajectories.npz` file, 
        preserving their nested structure via path-like keys (e.g., "trajectory/x"). 
        Standard Python types (scalars, strings, nulls) are saved to `metadata.json`.
        
        Parameters
        ----------
        directory : Union[str, Path]
            The directory where the files will be saved. Will be created if it 
            does not exist.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Package the dataclass fields into a standard dictionary
        data_dict = {
            "objective": self.objective,
            "trajectory": self.trajectory,
            "final_state": self.final_state,
            "metrics": self.metrics
        }
        
        arrays, metadata = self._flatten_for_storage(data_dict)
        
        # Save arrays efficiently using NumPy compressed archive
        if arrays:
            np.savez_compressed(dir_path / "trajectories.npz", **arrays)
            
        # Save Python primitives and 0D scalars as JSON
        if metadata:
            with open(dir_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "SimulationResult":
        """Reconstructs a SimulationResult from a saved directory.
        
        Parameters
        ----------
        directory : Union[str, Path]
            The directory containing `trajectories.npz` and/or `metadata.json`.
            
        Returns
        -------
        SimulationResult
            A new instance populated with the loaded data. Arrays are loaded 
            as standard NumPy arrays (which JAX can natively consume).
        """
        dir_path = Path(directory)
        arrays, metadata = {}, {}
        
        # Load standard arrays
        npz_file = dir_path / "trajectories.npz"
        if npz_file.exists():
            with np.load(npz_file) as npz:
                arrays = {k: v for k, v in npz.items()}
                
        # Load scalar metadata
        json_file = dir_path / "metadata.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                metadata = json.load(f)
                
        # Reconstruct the original PyTree structure
        data_dict = cls._unflatten_from_storage(arrays, metadata)
        return cls(**data_dict)

    # =======================================================================
    # Private Helpers for Nested Dictionary (PyTree) Serialization
    # =======================================================================
    @classmethod
    def _flatten_for_storage(cls, obj: Any, prefix: str = "") -> tuple[dict, dict]:
        """Recursively separates arrays and scalars, tracking paths as keys."""
        arrays, metadata = {}, {}
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Recursively flatten dictionaries, appending to the path prefix
                a, m = cls._flatten_for_storage(v, f"{prefix}{k}/")
                arrays.update(a)
                metadata.update(m)
        elif isinstance(obj, (np.ndarray, jax.Array)):
            # Store standard JAX/NumPy arrays
            arrays[prefix[:-1]] = np.asarray(obj)
        elif isinstance(obj, (list, tuple)):
            if any(isinstance(x, (np.ndarray, jax.Array)) for x in obj):
                # Lists containing arrays are stacked into a single NumPy array
                arrays[prefix[:-1]] = np.asarray(obj)
            else:
                # Pure Python lists go to JSON
                metadata[prefix[:-1]] = obj
        elif obj is None:
            metadata[prefix[:-1]] = None
        else:
            # Handle JAX/NumPy zero-dimensional scalars correctly for JSON
            if hasattr(obj, "item"):
                metadata[prefix[:-1]] = obj.item()
            else:
                # Catch-all for standard Python primitives (int, float, str)
                metadata[prefix[:-1]] = obj
                
        return arrays, metadata

    @classmethod
    def _unflatten_from_storage(cls, arrays: dict, metadata: dict) -> dict:
        """Reconstructs nested dictionaries from flat path-like keys."""
        result = {}
        
        for d in (arrays, metadata):
            for k, v in d.items():
                parts = k.split('/')
                current = result
                
                # Traverse and create nested dictionaries
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                    
                # Assign the leaf value at the end of the path
                current[parts[-1]] = v
                
        return result