# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import torch

import vllm.distributed.parallel_state as parallel_state
import vllm.utils.torch_utils as torch_utils
from vllm.forward_context import (
    get_dual_stream_role,
    get_forward_context,
    override_dual_stream_role,
    override_forward_context,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.v1.worker import dual_stream, workspace


class _FakeEvent:
    def record(self, stream):
        self.stream = stream


class _FakeStream:
    def wait_event(self, event):
        self.waited_for = event


class _FakeAttentionLayer(AttentionLayerBase):
    def get_attn_backend(self):
        return None

    def get_kv_cache_spec(self, vllm_config):
        return None


def _patch_fake_cuda(monkeypatch):
    stream = _FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)


def test_collective_sequencer_orders_roles(monkeypatch):
    _patch_fake_cuda(monkeypatch)
    sequencer = dual_stream._CollectiveSequencer()
    order = []

    def run(role):
        for _ in range(3):
            sequencer.run(role, lambda: order.append(role))

    throughput = threading.Thread(target=run, args=(dual_stream.THROUGHPUT,))
    latency = threading.Thread(target=run, args=(dual_stream.LATENCY,))
    throughput.start()
    latency.start()
    throughput.join(timeout=1)
    latency.join(timeout=1)

    assert not throughput.is_alive()
    assert not latency.is_alive()
    assert (
        order
        == [
            dual_stream.LATENCY,
            dual_stream.THROUGHPUT,
        ]
        * 3
    )


def test_collective_sequencer_drains_surviving_role(monkeypatch):
    _patch_fake_cuda(monkeypatch)
    sequencer = dual_stream._CollectiveSequencer()
    order = []

    def run_throughput():
        for _ in range(2):
            sequencer.run(
                dual_stream.THROUGHPUT,
                lambda: order.append(dual_stream.THROUGHPUT),
            )

    throughput = threading.Thread(target=run_throughput)
    throughput.start()
    sequencer.mark_done(dual_stream.LATENCY, 0)
    throughput.join(timeout=1)

    assert not throughput.is_alive()
    assert order == [dual_stream.THROUGHPUT] * 2


def test_tp_group_override_is_visible_through_existing_alias(monkeypatch):
    primary = object()
    override = object()
    get_tp_group_alias = parallel_state.get_tp_group
    monkeypatch.setattr(parallel_state, "_TP", primary)

    assert get_tp_group_alias() is primary
    with parallel_state.override_tp_group(override):
        assert get_tp_group_alias() is override
    assert get_tp_group_alias() is primary


def test_forward_context_is_isolated_between_threads():
    contexts = [object(), object()]
    observed = [None, None]
    barrier = threading.Barrier(2)

    def run(index):
        with override_forward_context(contexts[index]):
            barrier.wait()
            observed[index] = get_forward_context()

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert observed == contexts


def test_kv_cache_binding_is_isolated_by_role():
    layer = _FakeAttentionLayer()
    fallback = object()
    latency = object()
    throughput = object()
    layer.bind_kv_cache(fallback)

    with override_dual_stream_role(dual_stream.LATENCY):
        layer.bind_kv_cache(latency)
    with override_dual_stream_role(dual_stream.THROUGHPUT):
        layer.bind_kv_cache(throughput)

    assert layer.kv_cache is fallback
    with override_dual_stream_role(dual_stream.LATENCY):
        assert layer.kv_cache is latency
    with override_dual_stream_role(dual_stream.THROUGHPUT):
        assert layer.kv_cache is throughput


def test_role_context_binds_model_state_role():
    assert get_dual_stream_role() is None
    with dual_stream.role_context(dual_stream.LATENCY):
        assert get_dual_stream_role() == dual_stream.LATENCY
        assert dual_stream.current_role() == dual_stream.LATENCY
    assert get_dual_stream_role() is None


def test_aux_stream_is_disabled_in_dual_mode(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("VLLM_DUAL_STREAM", "1")
    monkeypatch.setattr(torch_utils, "_aux_stream", sentinel)

    assert torch_utils.aux_stream() is None


def test_shared_expert_transient_output_is_thread_local():
    shared_experts = object.__new__(SharedExperts)
    shared_experts._output_local = threading.local()
    observed = [None, None]
    barrier = threading.Barrier(2)

    def run(index):
        shared_experts._output[0] = index
        barrier.wait()
        observed[index] = shared_experts._output[0]

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert observed == [0, 1]


def test_latency_role_gets_a_distinct_workspace(monkeypatch):
    primary = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_managers", {None: primary})
    monkeypatch.setattr(dual_stream, "enabled", lambda: True)

    monkeypatch.setattr(dual_stream, "current_role", lambda: dual_stream.THROUGHPUT)
    assert workspace.current_workspace_manager() is primary

    monkeypatch.setattr(dual_stream, "current_role", lambda: dual_stream.LATENCY)
    latency = workspace.current_workspace_manager()
    assert latency is not primary
    assert workspace.current_workspace_manager() is latency
