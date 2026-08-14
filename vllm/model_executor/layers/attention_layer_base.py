# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from vllm.v1.kv_cache_interface import KVCacheSpec


class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"
    supports_dcp: bool = True

    @property
    def kv_cache(self) -> Any:
        """Return the cache bound to the current dual-stream role.

        Dual runners share the model module and therefore its attention-layer
        objects, but their cache allocations must remain disjoint. Outside
        dual-stream execution this behaves like the original plain attribute.
        """
        from vllm.forward_context import get_dual_stream_role

        role = get_dual_stream_role()
        role_caches = getattr(self, "_dual_stream_kv_caches", None)
        if role is not None and role_caches is not None and role in role_caches:
            return role_caches[role]
        if hasattr(self, "_kv_cache"):
            return self._kv_cache
        raise AttributeError("KV cache has not been bound")

    @kv_cache.setter
    def kv_cache(self, value: Any) -> None:
        from vllm.forward_context import get_dual_stream_role

        role = get_dual_stream_role()
        if role is None:
            self._kv_cache = value
            return
        role_caches = getattr(self, "_dual_stream_kv_caches", None)
        if role_caches is None:
            role_caches = {}
            self._dual_stream_kv_caches = role_caches
        role_caches[role] = value

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        self.kv_cache = kv_cache

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """
        Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
