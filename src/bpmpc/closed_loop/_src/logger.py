from typing import Any, Optional
import numpy as np

class RunLogger:
    """A dynamic terminal logger that formats runtime metrics into a clean table.
    Supports custom format strings per column.
    """
    
    def __init__(self, padding: int = 4, formats: Optional[dict[str, str]] = None):
        self.padding = padding
        self._last_keys = None  # Changed to None (was {})
        self._col_widths = {}
        self.formats = formats or {}

    def log(self, **metrics: Any) -> None:
        """Logs a row of raw metrics."""
        current_keys = list(metrics.keys())
        
        if current_keys != self._last_keys:
            self._update_layout(metrics)
            self._last_keys = current_keys
            self._print_header()
            
        self._print_row(metrics)

    def _update_layout(self, metrics: dict) -> None:
        self._col_widths = {
            key: max(len(str(key)), 10) + self.padding 
            for key in metrics.keys()
        }

    def _print_header(self) -> None:
        # Now self._last_keys is guaranteed to be updated before this runs
        header_cols = [f"{key:^{self._col_widths[key]}}" for key in self._last_keys]
        header_str = "|" + "|".join(header_cols) + "|"
        separator = "+" + "+".join("-" * self._col_widths[key] for key in self._last_keys) + "+"
        
        print("\n" + separator)
        print(header_str)
        print(separator)

    def _print_row(self, metrics: dict) -> None:
        row_cols = []
        for key, val in metrics.items():
            width = self._col_widths[key]
            
            # 1. Extract raw value if it's a 0D JAX/NumPy array
            if hasattr(val, "ndim") and val.ndim == 0:
                val = val.item()

            # 2. Apply formatting
            if key in self.formats:
                formatted_val = self.formats[key].format(val)
            elif isinstance(val, (int, np.integer)):
                formatted_val = f"{val:d}"
            elif isinstance(val, (float, np.floating)):
                formatted_val = f"{val:.4e}"  
            else:
                formatted_val = str(val)[:width-2]
                
            # Right-align the formatted string in the column
            row_cols.append(f"{formatted_val:>{width - 1}} ")
            
        row_str = "|" + "|".join(row_cols) + "|"
        print(row_str)