//! `Scheduler`: prefill, including chunked prefill, then decode with preemption.

use std::collections::VecDeque;
use std::path::Path;

use crate::block::BlockManager;
use crate::config::Config;
use crate::pyval::PyVal;
use crate::sampling::SamplingParams;
use crate::sequence::{Sequence, Status};

pub struct Scheduler {
    pub max_num_seqs: i64,
    pub max_num_batched_tokens: i64,
    pub eos: i64,
    pub block_size: i64,
    pub block_manager: BlockManager,
    pub waiting: VecDeque<Sequence>,
    pub running: VecDeque<Sequence>,
}

impl Scheduler {
    pub fn new(config: &Config) -> Self {
        Sequence::set_block_size(u64::try_from(config.kvcache_block_size).unwrap_or(0));
        Self {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            eos: config.eos,
            block_size: config.kvcache_block_size,
            block_manager: BlockManager::new(config.num_kvcache_blocks, config.kvcache_block_size),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
        }
    }

    pub fn from_fixture(repo: &Path, kwargs: &PyVal) -> Result<Self, String> {
        let config = Config::try_load(repo, "rust/record/fixtures/qwen3-tiny", kwargs)?;
        Ok(Self::new(&config))
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn add(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn schedule(&mut self) -> Result<(Vec<u64>, bool, Vec<i64>), String> {
        let mut scheduled_ids = Vec::new();
        let mut num_batched_tokens = 0_i64;
        while !self.waiting.is_empty() && (scheduled_ids.len() as i64) < self.max_num_seqs {
            let remaining = self.max_num_batched_tokens - num_batched_tokens;
            if remaining == 0 {
                break;
            }
            let empty_table = self.waiting[0].block_table.is_empty();
            let (cached, num_tokens) = if empty_table {
                let cached = self.block_manager.can_allocate(&self.waiting[0]);
                if cached == -1 {
                    break;
                }
                (
                    Some(cached),
                    self.waiting[0].num_tokens - cached * self.block_size,
                )
            } else {
                (
                    None,
                    self.waiting[0].num_tokens - self.waiting[0].num_cached_tokens,
                )
            };
            if remaining < num_tokens && !scheduled_ids.is_empty() {
                break;
            }
            if let Some(cached) = cached {
                let mut seq = self.waiting.pop_front().expect("waiting head");
                self.block_manager.allocate(&mut seq, cached)?;
                self.waiting.push_front(seq);
            }
            let scheduled_tokens = num_tokens.min(remaining);
            num_batched_tokens += scheduled_tokens;
            let complete =
                self.waiting[0].num_cached_tokens + scheduled_tokens == self.waiting[0].num_tokens;
            if complete {
                let mut seq = self.waiting.pop_front().expect("waiting head");
                seq.num_scheduled_tokens = scheduled_tokens;
                seq.status = Status::Running;
                scheduled_ids.push(seq.seq_id);
                self.running.push_back(seq);
            } else {
                self.waiting[0].num_scheduled_tokens = scheduled_tokens;
                scheduled_ids.push(self.waiting[0].seq_id);
            }
        }
        if !scheduled_ids.is_empty() {
            let counts = scheduled_ids.iter().map(|id| self.tokens_of(*id)).collect();
            return Ok((scheduled_ids, true, counts));
        }

        let mut chosen = Vec::new();
        while !self.running.is_empty() && (chosen.len() as i64) < self.max_num_seqs {
            let mut seq = self.running.pop_front().expect("running head");
            loop {
                if self.block_manager.can_append(&seq) {
                    seq.num_scheduled_tokens = 1;
                    seq.is_prefill = false;
                    self.block_manager.may_append(&mut seq)?;
                    chosen.push(seq);
                    break;
                }
                if let Some(mut victim) = self.running.pop_back() {
                    self.preempt(&mut victim);
                    self.waiting.push_front(victim);
                } else {
                    self.preempt(&mut seq);
                    self.waiting.push_front(seq);
                    break;
                }
            }
        }
        if chosen.is_empty() {
            return Err("AssertionError: ".into());
        }
        let ids = chosen.iter().map(|seq| seq.seq_id).collect();
        let counts = chosen.iter().map(|seq| seq.num_scheduled_tokens).collect();
        for seq in chosen.into_iter().rev() {
            self.running.push_front(seq);
        }
        Ok((ids, false, counts))
    }

    fn tokens_of(&self, seq_id: u64) -> i64 {
        self.waiting
            .iter()
            .chain(self.running.iter())
            .find(|seq| seq.seq_id == seq_id)
            .map(|seq| seq.num_scheduled_tokens)
            .unwrap_or(0)
    }

    fn preempt(&mut self, seq: &mut Sequence) {
        seq.status = Status::Waiting;
        seq.is_prefill = true;
        let _ = self.block_manager.deallocate(seq);
    }

    pub fn postprocess(
        &mut self,
        names: &[u64],
        token_ids: &[i64],
        is_prefill: bool,
    ) -> Result<(), String> {
        let n = names.len().min(token_ids.len());
        for i in 0..n {
            let seq_id = names[i];
            let token_id = token_ids[i];
            let Some(place) = self.find_mut(seq_id) else {
                return Err("ValueError: sequence is not queued".into());
            };
            match place {
                Place::Waiting(index) => {
                    self.block_manager.hash_blocks(&self.waiting[index])?;
                    let seq = &mut self.waiting[index];
                    seq.num_cached_tokens += seq.num_scheduled_tokens;
                    seq.num_scheduled_tokens = 0;
                    if is_prefill && seq.num_cached_tokens < seq.num_tokens {
                        continue;
                    }
                    seq.append_token(token_id);
                    let done = (!seq.ignore_eos && token_id == self.eos)
                        || i128::from(seq.num_tokens - seq.num_prompt_tokens) == seq.max_tokens;
                    if done {
                        seq.status = Status::Finished;
                        let mut seq = self.waiting.remove(index).expect("waiting");
                        self.block_manager.deallocate(&mut seq)?;
                    }
                }
                Place::Running(index) => {
                    self.block_manager.hash_blocks(&self.running[index])?;
                    let seq = &mut self.running[index];
                    seq.num_cached_tokens += seq.num_scheduled_tokens;
                    seq.num_scheduled_tokens = 0;
                    if is_prefill && seq.num_cached_tokens < seq.num_tokens {
                        continue;
                    }
                    seq.append_token(token_id);
                    let done = (!seq.ignore_eos && token_id == self.eos)
                        || i128::from(seq.num_tokens - seq.num_prompt_tokens) == seq.max_tokens;
                    if done {
                        seq.status = Status::Finished;
                        let mut seq = self.running.remove(index).expect("running");
                        self.block_manager.deallocate(&mut seq)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn find_mut(&self, seq_id: u64) -> Option<Place> {
        if let Some(index) = self.waiting.iter().position(|seq| seq.seq_id == seq_id) {
            return Some(Place::Waiting(index));
        }
        self.running
            .iter()
            .position(|seq| seq.seq_id == seq_id)
            .map(Place::Running)
    }

    pub fn snapshot(&self) -> PyVal {
        PyVal::Map(vec![
            ("eos".into(), PyVal::Int(i128::from(self.eos))),
            (
                "max_num_seqs".into(),
                PyVal::Int(i128::from(self.max_num_seqs)),
            ),
            (
                "max_num_batched_tokens".into(),
                PyVal::Int(i128::from(self.max_num_batched_tokens)),
            ),
            ("block_size".into(), PyVal::Int(i128::from(self.block_size))),
            (
                "waiting".into(),
                PyVal::List(self.waiting.iter().map(Sequence::snapshot).collect()),
            ),
            (
                "running".into(),
                PyVal::List(self.running.iter().map(Sequence::snapshot).collect()),
            ),
            ("finished".into(), PyVal::Bool(self.is_finished())),
            ("blocks".into(), self.block_manager.snapshot()),
        ])
    }
}

enum Place {
    Waiting(usize),
    Running(usize),
}

pub fn sampling_from(value: &PyVal) -> Result<SamplingParams, String> {
    match value.get("sampling") {
        Some(args) => SamplingParams::try_new(args),
        None => SamplingParams::try_new(&PyVal::Map(vec![])),
    }
}
