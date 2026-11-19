//! Plain Rust model of nano-vllm's CPU engine.
//!
//! Behavior is checked by `tests/replay.rs` against JSONL traces produced by
//! `rust/record/record_engine.py` from the original classes.

pub mod config;
pub mod pyval;
pub mod sampling;
pub mod sequence;
