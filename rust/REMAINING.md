# Remaining units

`nanovllm.layers.divide` is the only export that gated on this machine. Everything else is still Python. No unverified Rust was added for it.

## This machine

No GPU. `nvidia-smi` is not installed. The torch that is installed is `2.15.0+cpu`, and `torch.cuda.is_available()` is false. Importing `nanovllm` also imports `flash_attn`. There is no flash-attn build here. Recording `divide` used a `PYTHONPATH` stub that raises if a flash-attn function is called. `divide` never calls it. The 215 recorded results are from the real function.

## What `analyze` produced

Seven units, folder-sized. `start_here` was `nanovllm/utils/` and `nanovllm/layers/`. Waves: utils (0), layers (1), models (2), `nanovllm/` and `nanovllm/engine/` together (3, import cycle), `example.py` and `bench.py` (4). Dependencies with no crate in the tool's table: torch, transformers, flash-attn, safetensors, triton, xxhash. numpy maps to `ndarray` (partial). tqdm maps to `indicatif`.

## Why the CPU classes did not gate

`rust-anything` 1.5.1 refuses a class that has a public attribute, a dataclass decorator, a special method, a property, a classmethod, or a base class. That refusal happens before record, and a GPU does not remove it.

`rust-anything bridge . nanovllm/` failed. `SamplingParams` and `Config` stay in Python (`@dataclass`, public fields, `__post_init__`). `LLM` stays because it subclasses `LLMEngine`.

`rust-anything bridge . nanovllm/engine/` failed. `Sequence`, `SequenceStatus`, `Block`, `BlockManager`, `Scheduler`, `LLMEngine`, and `ModelRunner` all stay in Python. `BlockManager.compute_hash` is a `@classmethod` on a refused class, so the prefix-cache hash was not recorded.

`rust-anything bridge . nanovllm/models/` failed. Every `Qwen3*` class subclasses `nn.Module` and sets public attributes.

## Layers and utils, besides `divide`

`nanovllm/layers/` did bridge. These exports were kept in Python, with the reason stored in `rust/layers-rs/rust-anything.json`:

- `store_kvcache` and `store_kvcache_kernel`: Triton. A CPU torch cannot run the kernel, and this machine has no CUDA device.
- `apply_rotary_emb`: arguments and the result are `torch.Tensor`. The recorder's object encoding stores a Tensor as `{"$object":"Tensor","attrs":{}}`. The numbers are not in the case, so a passing diff would not check them.
- `get_rope`: returns an `nn.Module`. The same encoding drops `cos_sin_cache`'s values.
- Every `nn.Module` (`Attention`, the linears, `RMSNorm`, `Sampler`, `RotaryEmbedding`, `SiluAndMul`): refused as a class.

`nanovllm/utils/` bridges `get_context`, `set_context`, `reset_context`, `default_weight_loader`, and `load_model`. They pass `torch.Tensor` and `nn.Module` values, and `Context` itself is a refused dataclass. They were not ported. A diff on the encoded shells would not check tensor contents.

## What a GPU machine has to record

Install a CUDA torch, real Triton, and real flash-attn. Do not use the import stub. Record on a copy of the repo, with a test model directory and no production credentials.

Once a rust-anything build can bridge the classes (this 1.5.1 CLI cannot), record at least:

- `BlockManager.compute_hash(token_ids, prefix=-1)`. It is xxh64. When `prefix != -1`, the first update is `prefix.to_bytes(8, "little")`. The token update is `numpy.array(token_ids).tobytes()` (NumPy's default integer dtype, int64, little-endian on this platform). Then `intdigest()`.
- `can_allocate`, `allocate`, `deallocate`, `may_append`, and `hash_blocks` on real `Sequence` values, including a prefix-cache hit and a full block pool.
- `Scheduler.schedule` and `postprocess` for prefill, chunked prefill, decode, and preemption. `Config` needs a real model directory: it asserts the path is a directory and calls `AutoConfig.from_pretrained`.
- `store_kvcache` on CUDA tensors, with the real Triton kernel, including `slot == -1`.
- `apply_rotary_emb` and `RotaryEmbedding.forward` with the actual cos/sin values, not the empty Tensor encoding.
- `Qwen3ForCausalLM` forward and `ModelRunner.run` for one short prefill and one decode step, against a local test checkpoint. Compare token ids. Do not treat a CPU stub as that recording.
- `load_model` against that checkpoint's safetensors, including a packed module name.

## Crates for that work

- Prefix hash: `xxhash-rust` (`xxh64`), checked against the bytes above.
- Token buffers: `ndarray` (dtype and layout have to match NumPy; the tool marks this fit partial).
- Model and attention: `candle` or `cudarc`. flash-attn has no drop-in Rust crate in the tool's table. The recorded CUDA outputs decide whether a candle kernel is enough or a `cudarc` kernel is required. Triton `store_kvcache` is a slot-wise copy; record it on device before rewriting it.
- `transformers` (`AutoConfig`, tokenizer, `Qwen3Config`) has no crate mapping here. Read the config JSON in the test checkpoint rather than reimplementing Hugging Face.
- Progress bars: `indicatif` for tqdm. Not on the behavior path.

`example.py` and `bench.py` load weights and generate. Leave them until the model recording above exists.
