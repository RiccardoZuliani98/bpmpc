**Tags:** #class

## Overview

The `RunLogger` class is a dynamic terminal logging utility designed to format runtime metrics into a clean, dynamically sized ASCII table. It also supports custom format strings per column.

## Purpose

During iterative processes like control loops, optimization algorithms, or hyperparameter tuning, it is critical to track ongoing metrics. `RunLogger` abstracts away the formatting boilerplate. It calculates column widths automatically based on the length of the metric names. It applies custom formatting or defaults floating-point numbers into clean scientific notation, and smartly reprints the table header only if the set of tracked fields changes mid-run.

## Key Methods & Parameters

* **`__init__(padding: int = 4, formats: Optional[dict[str, str]] = None)`**: Initializes the logger, allowing customization of the column `padding` and providing a dictionary of `formats` for specific metric keys.
* **`log(metrics)`**: The primary user-facing method. Takes arbitrary keyword arguments (e.g., `log(iter=1, cost=0.5)`) and prints them as a formatted row. If the given keys differ from the previously logged keys, it automatically triggers a layout update and reprints the header.
* **`_update_layout(metrics)`**: Recalculates the required width for each column to ensure perfect alignment.
* **`_print_header()`**: An internal helper that formats and prints the header row and its separators.
* **`_print_row(metrics)`**: An internal helper that handles the actual string formatting and terminal output. Before formatting, it extracts the raw scalar value if a metric is a 0D JAX or NumPy array. It then prioritizes custom format strings from `self.formats`, falling back to standard integer formatting, scientific notation for floats, or truncated string representations.

## Dependencies

*This class is a pure utility leaf node and has no internal dependencies on other core framework classes. It relies only on standard Python libraries and `numpy`.*