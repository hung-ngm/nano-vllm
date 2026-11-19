"""Drive the original engine classes and write JSONL traces.

Run from the repo root:

    python3 rust/record/record_engine.py

Importing ``nanovllm`` itself pulls flash-attn. This loader registers empty
packages first so only the modules under test are executed.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from codec import enc
from load_units import ROOT, exc_text, install

install()

from nanovllm.config import Config  # noqa: E402
from nanovllm.engine.block_manager import BlockManager  # noqa: E402
from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence, SequenceStatus  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
TRACES = Path(__file__).resolve().parent / "traces"
PLACEHOLDER = "rust/record/fixtures"


def norm(text: str) -> str:
    return text.replace(str(FIX), PLACEHOLDER).replace(str(FIX.resolve()), PLACEHOLDER)


def call(fn) -> dict:
    try:
        return {"ok": fn()}
    except Exception as err:  # noqa: BLE001
        return {"error": norm(exc_text(err))}


def write(name: str, cases: list[dict]) -> None:
    TRACES.mkdir(parents=True, exist_ok=True)
    path = TRACES / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for case in cases:
            out.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{name}: {len(cases)} cases -> {path.relative_to(ROOT)}")


def snap_sampling(params: SamplingParams) -> dict:
    return {
        "temperature": enc(params.temperature),
        "max_tokens": enc(params.max_tokens),
        "ignore_eos": enc(params.ignore_eos),
    }


def record_sampling() -> None:
    cases = []
    specs = [
        ("default", {}),
        ("hot", {"temperature": 0.2, "max_tokens": 8, "ignore_eos": True}),
        ("boundary", {"temperature": 1e-9, "max_tokens": 1, "ignore_eos": False}),
        ("just_over", {"temperature": 1e-10 + 1e-18, "max_tokens": 0, "ignore_eos": False}),
        ("greedy", {"temperature": 1e-10}),
        ("zero", {"temperature": 0.0}),
        ("negative", {"temperature": -1.0, "max_tokens": 4}),
        ("nan", {"temperature": float("nan")}),
        ("pos_inf", {"temperature": float("inf"), "max_tokens": 2}),
        ("neg_inf", {"temperature": float("-inf")}),
        ("neg_zero", {"temperature": -0.0}),
        ("big_tokens", {"max_tokens": 2**40}),
        ("neg_tokens", {"max_tokens": -3}),
        ("bool_temp", {"temperature": True}),
        ("str_temp", {"temperature": "hot"}),
    ]
    for name, kwargs in specs:
        def make(kwargs=kwargs):
            params = SamplingParams(**kwargs)
            return snap_sampling(params)

        cases.append({"id": name, "args": enc(kwargs), "result": call(make)})
    write("sampling_params", cases)


def snap_config(config: Config) -> dict:
    hf = config.hf_config
    return {
        "max_num_batched_tokens": enc(config.max_num_batched_tokens),
        "max_num_seqs": enc(config.max_num_seqs),
        "max_model_len": enc(config.max_model_len),
        "gpu_memory_utilization": enc(config.gpu_memory_utilization),
        "tensor_parallel_size": enc(config.tensor_parallel_size),
        "enforce_eager": enc(config.enforce_eager),
        "eos": enc(config.eos),
        "kvcache_block_size": enc(config.kvcache_block_size),
        "num_kvcache_blocks": enc(config.num_kvcache_blocks),
        "hf_model_type": enc(hf.model_type),
        "hf_max_position_embeddings": enc(hf.max_position_embeddings),
        "hf_hidden_size": enc(hf.hidden_size),
        "hf_num_attention_heads": enc(hf.num_attention_heads),
        "hf_num_key_value_heads": enc(hf.num_key_value_heads),
        "hf_vocab_size": enc(hf.vocab_size),
        "hf_num_hidden_layers": enc(hf.num_hidden_layers),
        "hf_intermediate_size": enc(hf.intermediate_size),
        "hf_rms_norm_eps": enc(hf.rms_norm_eps),
        "hf_rope_theta": enc(getattr(hf, "rope_theta", None)),
        "hf_hidden_act": enc(hf.hidden_act),
        "hf_tie_word_embeddings": enc(hf.tie_word_embeddings),
        "hf_attention_bias": enc(getattr(hf, "attention_bias", None)),
        "hf_head_dim": enc(getattr(hf, "head_dim", None)),
    }


def record_config() -> None:
    tiny = FIX / "qwen3-tiny"
    specs = [
        ("defaults", tiny, {}),
        ("clamped", tiny, {"max_model_len": 50, "eos": 9, "kvcache_block_size": 512, "tensor_parallel_size": 8, "num_kvcache_blocks": 3, "enforce_eager": True, "max_num_seqs": 4, "max_num_batched_tokens": 32, "gpu_memory_utilization": 0.5}),
        ("missing_dir", FIX / "missing", {}),
        ("bad_block", tiny, {"kvcache_block_size": 128}),
        ("block_zero", tiny, {"kvcache_block_size": 0}),
        ("tp_zero", tiny, {"tensor_parallel_size": 0}),
        ("tp_nine", tiny, {"tensor_parallel_size": 9}),
        ("tp_one", tiny, {"tensor_parallel_size": 1}),
        ("no_file", FIX / "empty-dir", {}),
        ("bad_json", FIX / "bad-json", {}),
    ]
    cases = []
    for name, path, kwargs in specs:
        def make(path=path, kwargs=kwargs):
            return snap_config(Config(str(path), **kwargs))

        cases.append({
            "id": name,
            "model": norm(str(path)),
            "args": enc(kwargs),
            "result": call(make),
        })
    write("config", cases)


SEQ_FIELDS = [
    "seq_id",
    "status",
    "token_ids",
    "last_token",
    "num_tokens",
    "num_prompt_tokens",
    "num_cached_tokens",
    "num_scheduled_tokens",
    "is_prefill",
    "block_table",
    "temperature",
    "max_tokens",
    "ignore_eos",
    "num_completion_tokens",
    "prompt_token_ids",
    "completion_token_ids",
    "num_blocks",
    "last_block_num_tokens",
    "is_finished",
    "len",
]


def snap_seq(seq: Sequence) -> dict:
    out = {}
    for name in SEQ_FIELDS:
        try:
            if name == "len":
                value = len(seq)
            elif name == "status":
                value = seq.status.name
                out[name] = enc(value)
                continue
            else:
                value = getattr(seq, name)
            out[name] = enc(value)
        except Exception as err:  # noqa: BLE001
            out[name] = {"error": norm(exc_text(err))}
    return out


def sampling_from(raw: dict) -> SamplingParams:
    return SamplingParams(
        temperature=raw.get("temperature", 1.0),
        max_tokens=raw.get("max_tokens", 64),
        ignore_eos=raw.get("ignore_eos", False),
    )


def reset_ids() -> None:
    Sequence.counter = itertools.count()


def record_sequence() -> None:
    Sequence.block_size = 4
    cases = []

    def scenario(name: str, steps: list[dict]) -> None:
        reset_ids()
        seq = None
        traced = []
        for step in steps:
            def run_bound(seq_box=seq, step=step):
                op = step["op"]
                if op == "new":
                    return Sequence(list(step["token_ids"]), sampling_from(step.get("sampling", {})))
                if seq_box is None:
                    raise RuntimeError("sequence was not created")
                if op == "append":
                    seq_box.append_token(step["token_id"])
                    return None
                if op == "getitem":
                    return seq_box[step["index"]]
                if op == "slice":
                    return seq_box[step["start"] : step["stop"] : step.get("step", 1)]
                if op == "block":
                    return seq_box.block(step["index"])
                if op == "getstate":
                    return seq_box.__getstate__()
                if op == "set_prefill":
                    seq_box.is_prefill = step["value"]
                    return None
                if op == "setstate":
                    seq_box.__setstate__(tuple(step["state"]))
                    return None
                if op == "set_status":
                    seq_box.status = SequenceStatus[step["status"]]
                    return None
                raise AssertionError(op)

            try:
                value = run_bound()
                if step["op"] == "new":
                    seq = value
                    value = None
                result = {"ok": enc(value)}
            except Exception as err:  # noqa: BLE001
                result = {"error": norm(exc_text(err))}
            traced.append({"op": step["op"], "step": enc(step), "result": result, "state": None if seq is None else snap_seq(seq)})
        cases.append({"id": name, "block_size": 4, "steps": traced})

    scenario("prompt", [
        {"op": "new", "token_ids": [7, 8, 9], "sampling": {"temperature": 0.5, "max_tokens": 2, "ignore_eos": True}},
        {"op": "getitem", "index": 0},
        {"op": "getitem", "index": -1},
        {"op": "getitem", "index": 5},
        {"op": "slice", "start": 1, "stop": None, "step": 1},
        {"op": "block", "index": 0},
        {"op": "block", "index": 1},
        {"op": "append", "token_id": 4},
        {"op": "append", "token_id": 5},
        {"op": "block", "index": 1},
        {"op": "set_status", "status": "FINISHED"},
        {"op": "getstate"},
    ])
    scenario("decode_state", [
        {"op": "new", "token_ids": [1, 2, 3, 4]},
        {"op": "set_prefill", "value": False},
        {"op": "getstate"},
        {"op": "append", "token_id": 9},
        {"op": "setstate", "state": [2, 2, 0, 0, [1], 8]},
    ])
    scenario("empty", [{"op": "new", "token_ids": []}])
    scenario("negative_index", [
        {"op": "new", "token_ids": [3, 4]},
        {"op": "getitem", "index": -2},
        {"op": "slice", "start": None, "stop": None, "step": -1},
    ])
    write("sequence", cases)


def snap_block(block) -> dict:
    return {
        "id": enc(block.block_id),
        "ref": enc(block.ref_count),
        "hash": enc(block.hash),
        "tokens": enc(list(block.token_ids)),
    }


def snap_bm(manager: BlockManager) -> dict:
    return {
        "block_size": enc(manager.block_size),
        "free": enc(list(manager.free_block_ids)),
        "used": enc(sorted(manager.used_block_ids)),
        "hash_to_block": enc([[h, i] for h, i in sorted(manager.hash_to_block_id.items())]),
        "blocks": [snap_block(block) for block in manager.blocks],
    }


def record_blocks() -> None:
    cases = []
    samples = [
        ([], -1),
        ([1, 2, 3], -1),
        ([1, 2, 3], 0),
        ([1, 2, 3], 5),
        ([1, 2, 3], 9771088612715187706),
        (list(range(4)), -1),
        ([-1, 0, 2**40], -1),
        ([256] * 4, 99),
    ]
    for tokens, prefix in samples:
        def run(tokens=tokens, prefix=prefix):
            return BlockManager.compute_hash(list(tokens), prefix)

        cases.append({
            "id": f"hash-{len(tokens)}-{prefix}",
            "op": "compute_hash",
            "token_ids": tokens,
            "prefix": enc(prefix),
            "result": call(lambda run=run: enc(run())),
        })

    Sequence.block_size = 4
    reset_ids()

    def managed(name: str, steps: list[dict]) -> None:
        reset_ids()
        manager = BlockManager(4, 4)
        seqs: dict[str, Sequence] = {}
        traced = []
        for step in steps:
            def run(step=step):
                op = step["op"]
                if op == "seq":
                    seqs[step["name"]] = Sequence(list(step["token_ids"]), SamplingParams())
                    return step["name"]
                seq = seqs[step["seq"]]
                if op == "can_allocate":
                    return manager.can_allocate(seq)
                if op == "allocate":
                    manager.allocate(seq, step["cached"])
                    return None
                if op == "deallocate":
                    manager.deallocate(seq)
                    return None
                if op == "can_append":
                    return manager.can_append(seq)
                if op == "may_append":
                    manager.may_append(seq)
                    return None
                if op == "append":
                    seq.append_token(step["token_id"])
                    return None
                if op == "hash_blocks":
                    seq.num_cached_tokens = step["cached_tokens"]
                    seq.num_scheduled_tokens = step["scheduled_tokens"]
                    manager.hash_blocks(seq)
                    return None
                raise AssertionError(op)

            try:
                result = {"ok": enc(run())}
            except Exception as err:  # noqa: BLE001
                result = {"error": norm(exc_text(err))}
            traced.append({
                "op": step["op"],
                "step": step,
                "result": result,
                "blocks": snap_bm(manager),
                "seqs": {key: snap_seq(seq) for key, seq in seqs.items()},
            })
        cases.append({"id": name, "op": "script", "num_blocks": 4, "block_size": 4, "steps": traced})

    managed("allocate_two", [
        {"op": "seq", "name": "a", "token_ids": [1, 2, 3, 4, 5]},
        {"op": "can_allocate", "seq": "a"},
        {"op": "allocate", "seq": "a", "cached": 0},
        {"op": "hash_blocks", "seq": "a", "cached_tokens": 0, "scheduled_tokens": 4},
        {"op": "seq", "name": "b", "token_ids": [1, 2, 3, 4, 9, 9]},
        {"op": "can_allocate", "seq": "b"},
        {"op": "allocate", "seq": "b", "cached": 1},
        {"op": "deallocate", "seq": "a"},
        {"op": "can_append", "seq": "b"},
        {"op": "append", "seq": "b", "token_id": 7},
        {"op": "may_append", "seq": "b"},
        {"op": "deallocate", "seq": "b"},
    ])
    managed("full", [
        {"op": "seq", "name": "a", "token_ids": [1, 1, 1, 1]},
        {"op": "seq", "name": "b", "token_ids": [2, 2, 2, 2]},
        {"op": "seq", "name": "c", "token_ids": [3, 3, 3, 3]},
        {"op": "seq", "name": "d", "token_ids": [4, 4, 4, 4]},
        {"op": "seq", "name": "e", "token_ids": [5, 5, 5, 5]},
        {"op": "allocate", "seq": "a", "cached": 0},
        {"op": "allocate", "seq": "b", "cached": 0},
        {"op": "allocate", "seq": "c", "cached": 0},
        {"op": "allocate", "seq": "d", "cached": 0},
        {"op": "can_allocate", "seq": "e"},
        {"op": "deallocate", "seq": "b"},
        {"op": "can_allocate", "seq": "e"},
        {"op": "allocate", "seq": "e", "cached": 0},
    ])
    write("block_manager", cases)


def snap_sched(sched: Scheduler) -> dict:
    def lane(queue) -> list:
        return [snap_seq(seq) for seq in queue]

    return {
        "eos": enc(sched.eos),
        "max_num_seqs": enc(sched.max_num_seqs),
        "max_num_batched_tokens": enc(sched.max_num_batched_tokens),
        "block_size": enc(sched.block_size),
        "waiting": lane(sched.waiting),
        "running": lane(sched.running),
        "finished": enc(sched.is_finished()),
        "blocks": snap_bm(sched.block_manager),
    }


def make_config(**kwargs) -> Config:
    defaults = dict(num_kvcache_blocks=8, kvcache_block_size=256, max_num_seqs=4, max_num_batched_tokens=128, eos=11, max_model_len=128)
    defaults.update(kwargs)
    return Config(str(FIX / "qwen3-tiny"), **defaults)


def record_scheduler() -> None:
    cases = []

    def script(name: str, config_kwargs: dict, steps: list[dict]) -> None:
        reset_ids()
        config = make_config(**config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        sched = Scheduler(config)
        seqs: dict[str, Sequence] = {}
        traced = []
        for step in steps:
            def run(step=step):
                op = step["op"]
                if op == "add":
                    seq = Sequence(list(step["token_ids"]), sampling_from(step.get("sampling", {})))
                    seqs[step["name"]] = seq
                    sched.add(seq)
                    return step["name"]
                if op == "schedule":
                    chosen, is_prefill = sched.schedule()
                    return {
                        "seq_ids": [seq.seq_id for seq in chosen],
                        "is_prefill": is_prefill,
                        "scheduled": [seq.num_scheduled_tokens for seq in chosen],
                    }
                if op == "post":
                    sched.postprocess(list(sched.running if not step.get("use_waiting") else sched.waiting), list(step["token_ids"]), step["is_prefill"])
                    return None
                if op == "post_ids":
                    chosen = [seqs[n] for n in step["names"]]
                    sched.postprocess(chosen, list(step["token_ids"]), step["is_prefill"])
                    return None
                raise AssertionError(op)

            try:
                result = {"ok": enc(run())}
            except Exception as err:  # noqa: BLE001
                result = {"error": norm(exc_text(err))}
            traced.append({"op": step["op"], "step": step, "result": result, "state": snap_sched(sched)})
        cases.append({
            "id": name,
            "config": {
                "num_kvcache_blocks": config.num_kvcache_blocks,
                "kvcache_block_size": config.kvcache_block_size,
                "max_num_seqs": config.max_num_seqs,
                "max_num_batched_tokens": config.max_num_batched_tokens,
                "eos": config.eos,
                "max_model_len": config.max_model_len,
                "tensor_parallel_size": config.tensor_parallel_size,
                "enforce_eager": config.enforce_eager,
                "gpu_memory_utilization": config.gpu_memory_utilization,
            },
            "steps": traced,
        })

    script("one_prefill_decode", {"max_num_batched_tokens": 16, "max_num_seqs": 2, "num_kvcache_blocks": 4}, [
        {"op": "add", "name": "a", "token_ids": [1, 2, 3, 4], "sampling": {"max_tokens": 2, "temperature": 0.7}},
        {"op": "schedule"},
        {"op": "post_ids", "names": ["a"], "token_ids": [20], "is_prefill": True},
        {"op": "schedule"},
        {"op": "post_ids", "names": ["a"], "token_ids": [21], "is_prefill": False},
        {"op": "schedule"},
    ])
    script("chunked", {"max_num_batched_tokens": 4, "max_num_seqs": 2, "num_kvcache_blocks": 4}, [
        {"op": "add", "name": "a", "token_ids": list(range(10)), "sampling": {"max_tokens": 1}},
        {"op": "add", "name": "b", "token_ids": [7, 7, 7], "sampling": {"max_tokens": 1}},
        {"op": "schedule"},
        {"op": "post_ids", "names": ["a"], "token_ids": [1], "is_prefill": True},
        {"op": "schedule"},
        {"op": "post_ids", "names": ["a"], "token_ids": [2], "is_prefill": True},
        {"op": "schedule"},
    ])
    script("eos_and_ignore", {"max_num_batched_tokens": 32, "num_kvcache_blocks": 4}, [
        {"op": "add", "name": "stop", "token_ids": [5], "sampling": {"max_tokens": 4, "ignore_eos": False}},
        {"op": "add", "name": "keep", "token_ids": [6], "sampling": {"max_tokens": 4, "ignore_eos": True}},
        {"op": "schedule"},
        {"op": "post_ids", "names": ["stop", "keep"], "token_ids": [11, 11], "is_prefill": True},
    ])
    script("preempt", {"max_num_batched_tokens": 512, "max_num_seqs": 2, "num_kvcache_blocks": 2}, [
        {"op": "add", "name": "a", "token_ids": [1] * 256, "sampling": {"max_tokens": 3}},
        {"op": "add", "name": "b", "token_ids": [2] * 256, "sampling": {"max_tokens": 3}},
        {"op": "schedule"},
        {"op": "post_ids", "names": ["a", "b"], "token_ids": [3, 4], "is_prefill": True},
        {"op": "schedule"},
    ])
    write("scheduler", cases)


if __name__ == "__main__":
    record_sampling()
    record_config()
    record_sequence()
    record_blocks()
    record_scheduler()
