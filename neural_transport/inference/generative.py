"""DEPRECATED: Use neural_transport.inference.generation instead.

This module re-exports all public names from generation.py for backward compatibility.
"""

from neural_transport.inference.generation import (  # noqa: F401
    GenerationPipeline,
    align_time,
    generate_for_distributional_eval,
    get_batches,
    get_zarrpath_obspath,
    is_bad_sample,
    iterative_generate,
    iterative_generate_oco2,
    parse_freq,
    remap_with_cdo,
)
