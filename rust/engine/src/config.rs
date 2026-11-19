//! `Config`: the checks in `__post_init__`, then the Hugging Face fields this tree actually reads.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::pyval::PyVal;

#[derive(Clone, Debug)]
pub struct Config {
    pub max_num_batched_tokens: i64,
    pub max_num_seqs: i64,
    pub max_model_len: i64,
    pub gpu_memory_utilization: f64,
    pub tensor_parallel_size: i64,
    pub enforce_eager: bool,
    pub eos: i64,
    pub kvcache_block_size: i64,
    pub num_kvcache_blocks: i64,
    pub hf_model_type: String,
    pub hf_max_position_embeddings: i64,
    pub hf_hidden_size: i64,
    pub hf_num_attention_heads: i64,
    pub hf_num_key_value_heads: i64,
    pub hf_vocab_size: i64,
    pub hf_num_hidden_layers: i64,
    pub hf_intermediate_size: i64,
    pub hf_rms_norm_eps: f64,
    pub hf_hidden_act: String,
    pub hf_tie_word_embeddings: bool,
    pub hf_attention_bias: bool,
    pub hf_head_dim: i64,
}

impl Config {
    pub fn try_load(repo: &Path, model: &str, args: &PyVal) -> Result<Self, String> {
        let path = repo.join(model);
        if !path.is_dir() {
            return Err("AssertionError: ".into());
        }
        let block = i64_arg(args, "kvcache_block_size", 256);
        if block % 256 != 0 {
            return Err("AssertionError: ".into());
        }
        let tp = i64_arg(args, "tensor_parallel_size", 1);
        if !(1..=8).contains(&tp) {
            return Err("AssertionError: ".into());
        }
        let file = path.join("config.json");
        if !file.is_file() {
            return Err(format!(
                "ValueError: Unrecognized model in {model}. Should have a `model_type` key in its config.json."
            ));
        }
        let text = fs::read_to_string(&file).map_err(|err| format!("OSError: {err}"))?;
        let json: Value = serde_json::from_str(&text).map_err(|_| {
            format!("OSError: It looks like the config file at '{model}/config.json' is not a valid JSON file.")
        })?;
        let Some(model_type) = json.get("model_type").and_then(Value::as_str) else {
            return Err(format!(
                "ValueError: Unrecognized model in {model}. Should have a `model_type` key in its config.json."
            ));
        };
        let max_position = json_i64(&json, "max_position_embeddings")?;
        let mut max_model_len = i64_arg(args, "max_model_len", 4096);
        if max_position < max_model_len {
            max_model_len = max_position;
        }
        Ok(Self {
            max_num_batched_tokens: i64_arg(args, "max_num_batched_tokens", 16384),
            max_num_seqs: i64_arg(args, "max_num_seqs", 512),
            max_model_len,
            gpu_memory_utilization: f64_arg(args, "gpu_memory_utilization", 0.9),
            tensor_parallel_size: tp,
            enforce_eager: bool_arg(args, "enforce_eager", false),
            eos: i64_arg(args, "eos", -1),
            kvcache_block_size: block,
            num_kvcache_blocks: i64_arg(args, "num_kvcache_blocks", -1),
            hf_model_type: model_type.to_string(),
            hf_max_position_embeddings: max_position,
            hf_hidden_size: json_i64(&json, "hidden_size")?,
            hf_num_attention_heads: json_i64(&json, "num_attention_heads")?,
            hf_num_key_value_heads: json_i64(&json, "num_key_value_heads")?,
            hf_vocab_size: json_i64(&json, "vocab_size")?,
            hf_num_hidden_layers: json_i64(&json, "num_hidden_layers")?,
            hf_intermediate_size: json_i64(&json, "intermediate_size")?,
            hf_rms_norm_eps: json_f64(&json, "rms_norm_eps")?,
            hf_hidden_act: json
                .get("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or("silu")
                .to_string(),
            hf_tie_word_embeddings: json
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hf_attention_bias: json
                .get("attention_bias")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hf_head_dim: json_i64(&json, "head_dim")?,
        })
    }

    pub fn snapshot(&self) -> PyVal {
        PyVal::Map(vec![
            (
                "max_num_batched_tokens".into(),
                PyVal::Int(i128::from(self.max_num_batched_tokens)),
            ),
            (
                "max_num_seqs".into(),
                PyVal::Int(i128::from(self.max_num_seqs)),
            ),
            (
                "max_model_len".into(),
                PyVal::Int(i128::from(self.max_model_len)),
            ),
            (
                "gpu_memory_utilization".into(),
                PyVal::Float(self.gpu_memory_utilization),
            ),
            (
                "tensor_parallel_size".into(),
                PyVal::Int(i128::from(self.tensor_parallel_size)),
            ),
            ("enforce_eager".into(), PyVal::Bool(self.enforce_eager)),
            ("eos".into(), PyVal::Int(i128::from(self.eos))),
            (
                "kvcache_block_size".into(),
                PyVal::Int(i128::from(self.kvcache_block_size)),
            ),
            (
                "num_kvcache_blocks".into(),
                PyVal::Int(i128::from(self.num_kvcache_blocks)),
            ),
            (
                "hf_model_type".into(),
                PyVal::Str(self.hf_model_type.clone()),
            ),
            (
                "hf_max_position_embeddings".into(),
                PyVal::Int(i128::from(self.hf_max_position_embeddings)),
            ),
            (
                "hf_hidden_size".into(),
                PyVal::Int(i128::from(self.hf_hidden_size)),
            ),
            (
                "hf_num_attention_heads".into(),
                PyVal::Int(i128::from(self.hf_num_attention_heads)),
            ),
            (
                "hf_num_key_value_heads".into(),
                PyVal::Int(i128::from(self.hf_num_key_value_heads)),
            ),
            (
                "hf_vocab_size".into(),
                PyVal::Int(i128::from(self.hf_vocab_size)),
            ),
            (
                "hf_num_hidden_layers".into(),
                PyVal::Int(i128::from(self.hf_num_hidden_layers)),
            ),
            (
                "hf_intermediate_size".into(),
                PyVal::Int(i128::from(self.hf_intermediate_size)),
            ),
            ("hf_rms_norm_eps".into(), PyVal::Float(self.hf_rms_norm_eps)),
            // This transformers build does not surface `rope_theta` on Qwen3Config.
            ("hf_rope_theta".into(), PyVal::Null),
            (
                "hf_hidden_act".into(),
                PyVal::Str(self.hf_hidden_act.clone()),
            ),
            (
                "hf_tie_word_embeddings".into(),
                PyVal::Bool(self.hf_tie_word_embeddings),
            ),
            (
                "hf_attention_bias".into(),
                PyVal::Bool(self.hf_attention_bias),
            ),
            (
                "hf_head_dim".into(),
                PyVal::Int(i128::from(self.hf_head_dim)),
            ),
        ])
    }
}

fn i64_arg(args: &PyVal, key: &str, default: i64) -> i64 {
    args.get(key)
        .and_then(PyVal::as_i128)
        .and_then(|n| i64::try_from(n).ok())
        .unwrap_or(default)
}

fn f64_arg(args: &PyVal, key: &str, default: f64) -> f64 {
    match args.get(key) {
        Some(PyVal::Float(n)) => *n,
        Some(PyVal::Int(n)) => *n as f64,
        _ => default,
    }
}

fn bool_arg(args: &PyVal, key: &str, default: bool) -> bool {
    match args.get(key) {
        Some(PyVal::Bool(v)) => *v,
        _ => default,
    }
}

fn json_i64(json: &Value, key: &str) -> Result<i64, String> {
    json.get(key)
        .and_then(|v| {
            v.as_i64()
                .or_else(|| v.as_u64().and_then(|n| i64::try_from(n).ok()))
        })
        .ok_or_else(|| format!("ValueError: config.json is missing {key}"))
}

fn json_f64(json: &Value, key: &str) -> Result<f64, String> {
    json.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("ValueError: config.json is missing {key}"))
}
