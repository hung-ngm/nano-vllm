//! Replay `rust/record/traces/*.jsonl` against the Rust engine.

use std::fs;
use std::path::{Path, PathBuf};

use nanovllm_engine::config::Config;
use nanovllm_engine::pyval::{self, PyVal};
use nanovllm_engine::sampling::SamplingParams;
use serde_json::Value;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load(name: &str) -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../record/traces")
        .join(name);
    fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("trace line"))
        .collect()
}

fn check_result(id: &str, actual: Result<PyVal, String>, expected: &Value) {
    if let Some(err) = expected.get("error").and_then(Value::as_str) {
        match actual {
            Err(got) if got == err => {}
            other => panic!("{id}\n  got {other:?}\n  expected error {err}"),
        }
    } else {
        let want = pyval::decode(&expected["ok"]);
        match actual {
            Ok(got) if got == want => {}
            other => panic!("{id}\n  got {other:?}\n  expected {want:?}"),
        }
    }
}

#[test]
fn sampling_params_matches_python() {
    for case in load("sampling_params.jsonl") {
        let id = case["id"].as_str().unwrap();
        let args = pyval::decode(&case["args"]);
        let actual = SamplingParams::try_new(&args).map(|params| params.snapshot());
        check_result(id, actual, &case["result"]);
    }
}

#[test]
fn config_matches_python() {
    let root = repo();
    for case in load("config.jsonl") {
        let id = case["id"].as_str().unwrap();
        let model = case["model"].as_str().unwrap();
        let args = pyval::decode(&case["args"]);
        let actual = Config::try_load(&root, model, &args).map(|config| config.snapshot());
        check_result(id, actual, &case["result"]);
    }
}
