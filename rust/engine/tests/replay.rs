//! Replay `rust/record/traces/*.jsonl` against the Rust engine.

use std::fs;
use std::path::{Path, PathBuf};

use nanovllm_engine::block::BlockManager;
use nanovllm_engine::config::Config;
use nanovllm_engine::pyval::{self, PyVal};
use nanovllm_engine::sampling::SamplingParams;
use nanovllm_engine::scheduler::{self, Scheduler};
use nanovllm_engine::sequence::{self, Sequence};
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

fn check_state(id: &str, actual: &PyVal, expected: &Value) {
    let want = pyval::decode(expected);
    if actual != &want {
        panic!("{id} state\n  got {actual:?}\n  expected {want:?}");
    }
}

fn opt_i64(value: Option<&PyVal>) -> Option<i64> {
    match value {
        None | Some(PyVal::Null) => None,
        Some(other) => Some(sequence::as_i64(other).expect("index")),
    }
}

fn i64s(value: &PyVal) -> Vec<i64> {
    let PyVal::List(items) = value else {
        panic!("expected a list, got {value:?}");
    };
    items
        .iter()
        .map(|v| sequence::as_i64(v).expect("int"))
        .collect()
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

#[test]
fn sequence_matches_python() {
    for case in load("sequence.jsonl") {
        let id = case["id"].as_str().unwrap();
        Sequence::reset_ids();
        Sequence::set_block_size(case["block_size"].as_u64().unwrap());
        let mut seq: Option<Sequence> = None;
        for (n, step_json) in case["steps"].as_array().unwrap().iter().enumerate() {
            let step = pyval::decode(&step_json["step"]);
            let op = step
                .get("op")
                .and_then(PyVal::as_string)
                .unwrap()
                .to_string();
            let label = format!("{id}#{n} {op}");
            let actual = match op.as_str() {
                "new" => {
                    let tokens = i64s(step.get("token_ids").unwrap());
                    let sampling = SamplingParams::try_new(
                        step.get("sampling").unwrap_or(&PyVal::Map(vec![])),
                    );
                    match sampling.and_then(|params| Sequence::try_new(&tokens, params)) {
                        Ok(created) => {
                            seq = Some(created);
                            Ok(PyVal::Null)
                        }
                        Err(err) => Err(err),
                    }
                }
                "append" => {
                    seq.as_mut()
                        .unwrap()
                        .append_token(opt_i64(step.get("token_id")).unwrap());
                    Ok(PyVal::Null)
                }
                "getitem" => seq
                    .as_ref()
                    .unwrap()
                    .get(opt_i64(step.get("index")).unwrap())
                    .map(py_i64),
                "slice" => seq
                    .as_ref()
                    .unwrap()
                    .slice(
                        opt_i64(step.get("start")),
                        opt_i64(step.get("stop")),
                        opt_i64(step.get("step")).unwrap_or(1),
                    )
                    .map(|xs| pyval::list_i64(&xs)),
                "block" => seq
                    .as_ref()
                    .unwrap()
                    .block(opt_i64(step.get("index")).unwrap())
                    .map(|xs| pyval::list_i64(&xs)),
                "getstate" => Ok(seq.as_ref().unwrap().getstate()),
                "set_prefill" => {
                    seq.as_mut().unwrap().is_prefill =
                        matches!(step.get("value"), Some(PyVal::Bool(true)));
                    Ok(PyVal::Null)
                }
                "setstate" => seq
                    .as_mut()
                    .unwrap()
                    .setstate(step.get("state").unwrap())
                    .map(|()| PyVal::Null),
                "set_status" => seq
                    .as_mut()
                    .unwrap()
                    .set_status(step.get("status").and_then(PyVal::as_string).unwrap())
                    .map(|()| PyVal::Null),
                other => panic!("unknown sequence op {other}"),
            };
            check_result(&label, actual, &step_json["result"]);
            if let Some(seq) = &seq {
                if !step_json["state"].is_null() {
                    check_state(&label, &seq.snapshot(), &step_json["state"]);
                }
            }
        }
    }
}

fn py_i64(value: i64) -> PyVal {
    PyVal::Int(i128::from(value))
}

#[test]
fn block_manager_matches_python() {
    for case in load("block_manager.jsonl") {
        let id = case["id"].as_str().unwrap().to_string();
        if case["op"] == "compute_hash" {
            let tokens: Vec<i64> = case["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_i64().unwrap())
                .collect();
            let prefix = pyval::decode(&case["prefix"]).as_i128().unwrap();
            let hash = i128::from(BlockManager::compute_hash(&tokens, prefix));
            check_result(&id, Ok(PyVal::Int(hash)), &case["result"]);
            continue;
        }
        Sequence::reset_ids();
        Sequence::set_block_size(case["block_size"].as_u64().unwrap());
        let mut manager = BlockManager::new(
            case["num_blocks"].as_i64().unwrap(),
            case["block_size"].as_i64().unwrap(),
        );
        let mut seqs: Vec<(String, Sequence)> = Vec::new();
        for (n, step_json) in case["steps"].as_array().unwrap().iter().enumerate() {
            let step = pyval::decode(&step_json["step"]);
            let op = step
                .get("op")
                .and_then(PyVal::as_string)
                .unwrap_or("")
                .to_string();
            let label = format!("{id}#{n} {op}");
            let actual = run_block_step(&mut manager, &mut seqs, &step, &op);
            check_result(&label, actual, &step_json["result"]);
            check_state(
                &format!("{label} blocks"),
                &manager.snapshot(),
                &step_json["blocks"],
            );
            let got_seqs = PyVal::Map(
                seqs.iter()
                    .map(|(name, seq)| (name.clone(), seq.snapshot()))
                    .collect(),
            );
            check_state(&format!("{label} seqs"), &got_seqs, &step_json["seqs"]);
        }
    }
}

fn run_block_step(
    manager: &mut BlockManager,
    seqs: &mut Vec<(String, Sequence)>,
    step: &PyVal,
    op: &str,
) -> Result<PyVal, String> {
    if op == "seq" {
        let name = step
            .get("name")
            .and_then(PyVal::as_string)
            .unwrap()
            .to_string();
        let tokens = i64s(step.get("token_ids").unwrap());
        let seq = BlockManager::fresh_seq(&tokens)?;
        seqs.push((name.clone(), seq));
        return Ok(PyVal::Str(name));
    }
    let name = step.get("seq").and_then(PyVal::as_string).unwrap();
    let seq = &mut seqs.iter_mut().find(|(n, _)| n == name).unwrap().1;
    match op {
        "can_allocate" => Ok(py_i64(manager.can_allocate(seq))),
        "allocate" => manager
            .allocate(seq, opt_i64(step.get("cached")).unwrap())
            .map(|()| PyVal::Null),
        "deallocate" => manager.deallocate(seq).map(|()| PyVal::Null),
        "can_append" => Ok(PyVal::Bool(manager.can_append(seq))),
        "may_append" => manager.may_append(seq).map(|()| PyVal::Null),
        "append" => {
            seq.append_token(opt_i64(step.get("token_id")).unwrap());
            Ok(PyVal::Null)
        }
        "hash_blocks" => {
            seq.num_cached_tokens = opt_i64(step.get("cached_tokens")).unwrap();
            seq.num_scheduled_tokens = opt_i64(step.get("scheduled_tokens")).unwrap();
            manager.hash_blocks(seq).map(|()| PyVal::Null)
        }
        other => panic!("unknown block op {other}"),
    }
}

#[test]
fn scheduler_matches_python() {
    let root = repo();
    for case in load("scheduler.jsonl") {
        let id = case["id"].as_str().unwrap().to_string();
        Sequence::reset_ids();
        let config = pyval::decode(&case["config"]);
        let mut sched = Scheduler::from_fixture(&root, &config)
            .unwrap_or_else(|err| panic!("{id} config {err}"));
        let mut names: Vec<(String, u64)> = Vec::new();
        for (n, step_json) in case["steps"].as_array().unwrap().iter().enumerate() {
            let step = pyval::decode(&step_json["step"]);
            let op = step
                .get("op")
                .and_then(PyVal::as_string)
                .unwrap()
                .to_string();
            let label = format!("{id}#{n} {op}");
            let actual = match op.as_str() {
                "add" => {
                    let tokens = i64s(step.get("token_ids").unwrap());
                    match scheduler::sampling_from(&step)
                        .and_then(|sampling| Sequence::try_new(&tokens, sampling))
                    {
                        Ok(seq) => {
                            let seq_id = seq.seq_id;
                            let name = step
                                .get("name")
                                .and_then(PyVal::as_string)
                                .unwrap()
                                .to_string();
                            names.push((name.clone(), seq_id));
                            sched.add(seq);
                            Ok(PyVal::Str(name))
                        }
                        Err(err) => Err(err),
                    }
                }
                "schedule" => sched.schedule().map(|(ids, prefill, counts)| {
                    PyVal::Map(vec![
                        (
                            "seq_ids".into(),
                            PyVal::List(
                                ids.into_iter()
                                    .map(|id| PyVal::Int(i128::from(id)))
                                    .collect(),
                            ),
                        ),
                        ("is_prefill".into(), PyVal::Bool(prefill)),
                        (
                            "scheduled".into(),
                            PyVal::List(
                                counts
                                    .into_iter()
                                    .map(|n| PyVal::Int(i128::from(n)))
                                    .collect(),
                            ),
                        ),
                    ])
                }),
                "post_ids" => {
                    let who = step.get("names").map(i64_names).unwrap_or_default();
                    let ids: Vec<u64> = who
                        .iter()
                        .map(|name| names.iter().find(|(n, _)| n == name).unwrap().1)
                        .collect();
                    let tokens = i64s(step.get("token_ids").unwrap());
                    let is_prefill = matches!(step.get("is_prefill"), Some(PyVal::Bool(true)));
                    sched
                        .postprocess(&ids, &tokens, is_prefill)
                        .map(|()| PyVal::Null)
                }
                other => panic!("unknown scheduler op {other}"),
            };
            check_result(&label, actual, &step_json["result"]);
            check_state(&label, &sched.snapshot(), &step_json["state"]);
        }
    }
}

fn i64_names(value: &PyVal) -> Vec<String> {
    let PyVal::List(items) = value else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|v| v.as_string().map(str::to_string))
        .collect()
}
