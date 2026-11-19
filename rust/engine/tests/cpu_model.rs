use std::fs;
use std::path::Path;

use nanovllm_engine::model::{self, parse_f32};
use serde_json::Value;

fn nums(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| parse_f32(v.as_str().unwrap()))
        .collect()
}

#[test]
fn cpu_model_matches_torch() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../record/traces/cpu_model.jsonl");
    let text = fs::read_to_string(&path).unwrap_or_else(|err| panic!("{} {err}", path.display()));
    for line in text.lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let id = case["id"].as_str().unwrap();
        let got = match case["op"].as_str().unwrap() {
            "rms" => {
                let shape = case["shape"].as_array().unwrap();
                let rows = shape[0].as_u64().unwrap() as usize;
                let cols = shape[1].as_u64().unwrap() as usize;
                model::rms_norm(
                    &nums(&case["x"]),
                    &nums(&case["weight"]),
                    rows,
                    cols,
                    case["eps"].as_f64().unwrap() as f32,
                )
            }
            "silu" => {
                let shape = case["shape"].as_array().unwrap();
                let rows = shape[0].as_u64().unwrap() as usize;
                let cols = shape[1].as_u64().unwrap() as usize;
                model::silu_and_mul(&nums(&case["x"]), rows, cols)
            }
            "rope" => {
                let shape: Vec<usize> = case["x_shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect();
                model::apply_rotary(
                    &nums(&case["x"]),
                    &nums(&case["cos"]),
                    &nums(&case["sin"]),
                    &shape,
                )
            }
            "rope_cache" => model::rope_cache(
                case["head_dim"].as_u64().unwrap() as usize,
                case["max_pos"].as_u64().unwrap() as usize,
                case["base"].as_f64().unwrap() as f32,
            ),
            "attn" => {
                let shape: Vec<usize> = case["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect();
                model::causal_attn(
                    &nums(&case["q"]),
                    &nums(&case["k"]),
                    &nums(&case["v"]),
                    shape[0],
                    shape[1],
                    shape[2],
                    case["scale"].as_f64().unwrap() as f32,
                )
            }
            "linear" => {
                let xs = case["x_shape"].as_array().unwrap();
                let ws = case["w_shape"].as_array().unwrap();
                model::linear(
                    &nums(&case["x"]),
                    &nums(&case["weight"]),
                    Some(&nums(&case["bias"])),
                    xs[0].as_u64().unwrap() as usize,
                    xs[1].as_u64().unwrap() as usize,
                    ws[0].as_u64().unwrap() as usize,
                )
            }
            "sample" => {
                let shape = case["shape"].as_array().unwrap();
                let tokens = model::sample(
                    &nums(&case["logits"]),
                    &nums(&case["noise"]),
                    shape[0].as_u64().unwrap() as usize,
                    shape[1].as_u64().unwrap() as usize,
                    case["temperature"].as_f64().unwrap() as f32,
                );
                let expect: Vec<i64> = case["out"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_i64().unwrap())
                    .collect();
                assert_eq!(tokens, expect, "{id}");
                continue;
            }
            other => panic!("unknown op {other}"),
        };
        let expect = nums(&case["out"]);
        model::close_enough(&got, &expect).unwrap_or_else(|err| panic!("{id}: {err}"));
    }
}
