# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar
from weakref import ReferenceType, ref

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
    _dual_stream_kv_registry: ClassVar[
        dict[
            tuple[str, str],
            tuple[bool, ReferenceType[Any] | tuple[ReferenceType[Any], ...]],
        ]
    ] = {}

    def _dual_stream_kv_cache_key(self) -> str | None:
        """Return the stable name used by the model's forward context."""
        return getattr(self, "layer_name", None) or getattr(self, "prefix", None)

    @staticmethod
    def _make_dual_stream_kv_ref(
        value: Any,
    ) -> tuple[bool, ReferenceType[Any] | tuple[ReferenceType[Any], ...]] | None:
        if isinstance(value, torch.Tensor):
            return False, ref(value)
        if isinstance(value, tuple) and all(isinstance(v, torch.Tensor) for v in value):
            return True, tuple(ref(v) for v in value)
        return None

    @staticmethod
    def _resolve_dual_stream_kv_ref(
        cache_ref: tuple[bool, ReferenceType[Any] | tuple[ReferenceType[Any], ...]],
    ) -> Any | None:
        is_tuple, refs = cache_ref
        if not is_tuple:
            assert isinstance(refs, ReferenceType)
            return refs()
        assert isinstance(refs, tuple)
        values = tuple(cache_ref() for cache_ref in refs)
        if any(value is None for value in values):
            return None
        return values

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
        # Some pluggable hybrid layers execute through a runtime module instance
        # distinct from the object registered in static_forward_context. Resolve
        # those through the stable layer name without retaining cache tensors
        # after their owning model is released.
        layer_key = self._dual_stream_kv_cache_key()
        if role is not None and layer_key is not None:
            registry_key = (layer_key, role)
            cache_ref = AttentionLayerBase._dual_stream_kv_registry.get(registry_key)
            if cache_ref is not None:
                value = self._resolve_dual_stream_kv_ref(cache_ref)
                if value is not None:
                    return value
                AttentionLayerBase._dual_stream_kv_registry.pop(registry_key, None)
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
        layer_key = self._dual_stream_kv_cache_key()
        cache_ref = self._make_dual_stream_kv_ref(value)
        if layer_key is not None and cache_ref is not None:
            AttentionLayerBase._dual_stream_kv_registry[(layer_key, role)] = cache_ref

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
