"""
Patch for vllm.v1.engine.utils module
Creates CoreEngineProcManager wrapper for compatibility with verl
"""

from vllm.v1.engine.core import EngineCoreProc
from typing import Any, Callable


class CoreEngineProcManager:
    """
    Simplified wrapper for EngineCoreProc to maintain compatibility with verl.
    """

    def __init__(
        self,
        target_fn: Callable,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: Any,
        **kwargs
    ):
        self.target_fn = target_fn
        self.local_engine_count = local_engine_count
        self.start_index = start_index
        self.local_start_index = local_start_index
        self.vllm_config = vllm_config
        self.engines = []

        for i in range(local_engine_count):
            engine_idx = start_index + local_start_index + i
            self.engines.append({
                "index": engine_idx,
                "config": vllm_config
            })

    def __del__(self):
        self.engines.clear()
