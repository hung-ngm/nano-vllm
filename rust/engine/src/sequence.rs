//! `Sequence`: token buffer, block geometry, and the pickle state pair.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::pyval::{self, PyVal};
use crate::sampling::{SamplingParams, Scalar};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);
static BLOCK_SIZE: AtomicU64 = AtomicU64::new(4);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Waiting,
    Running,
    Finished,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Self::Waiting => "WAITING",
            Self::Running => "RUNNING",
            Self::Finished => "FINISHED",
        }
    }

    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "WAITING" => Ok(Self::Waiting),
            "RUNNING" => Ok(Self::Running),
            "FINISHED" => Ok(Self::Finished),
            _ => Err(format!("KeyError: {name}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sequence {
    pub seq_id: u64,
    pub status: Status,
    pub token_ids: Vec<i64>,
    pub last_token: i64,
    pub num_tokens: i64,
    pub num_prompt_tokens: i64,
    pub num_cached_tokens: i64,
    pub num_scheduled_tokens: i64,
    pub is_prefill: bool,
    pub block_table: Vec<i64>,
    pub temperature: Scalar,
    pub max_tokens: i128,
    pub ignore_eos: bool,
    block_size: usize,
}

impl Sequence {
    pub fn reset_ids() {
        NEXT_ID.store(0, Ordering::Relaxed);
    }

    pub fn set_block_size(size: u64) {
        BLOCK_SIZE.store(size, Ordering::Relaxed);
    }

    pub fn try_new(token_ids: &[i64], sampling: SamplingParams) -> Result<Self, String> {
        let last_token = *token_ids
            .last()
            .ok_or("IndexError: list index out of range")?;
        let n = i64::try_from(token_ids.len()).unwrap_or(i64::MAX);
        Ok(Self {
            seq_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            status: Status::Waiting,
            token_ids: token_ids.to_vec(),
            last_token,
            num_tokens: n,
            num_prompt_tokens: n,
            num_cached_tokens: 0,
            num_scheduled_tokens: 0,
            is_prefill: true,
            block_table: Vec::new(),
            temperature: sampling.temperature,
            max_tokens: sampling.max_tokens,
            ignore_eos: sampling.ignore_eos,
            block_size: usize::try_from(BLOCK_SIZE.load(Ordering::Relaxed)).unwrap_or(1),
        })
    }

    pub fn len(&self) -> i64 {
        self.num_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    pub fn num_blocks(&self) -> i64 {
        let size = self.block_size as i64;
        (self.num_tokens + size - 1) / size
    }

    pub fn last_block_num_tokens(&self) -> i64 {
        self.num_tokens - (self.num_blocks() - 1) * self.block_size as i64
    }

    pub fn get(&self, index: i64) -> Result<i64, String> {
        let n = self.token_ids.len() as i64;
        let i = if index < 0 { index + n } else { index };
        if i < 0 || i >= n {
            return Err("IndexError: list index out of range".into());
        }
        Ok(self.token_ids[usize::try_from(i).unwrap_or(0)])
    }

    pub fn slice(
        &self,
        start: Option<i64>,
        stop: Option<i64>,
        step: i64,
    ) -> Result<Vec<i64>, String> {
        if step == 0 {
            return Err("ValueError: slice step cannot be zero".into());
        }
        let len = self.token_ids.len() as i64;
        let (start, stop, step) = slice_indices(len, start, stop, step);
        let mut out = Vec::new();
        let mut i = start;
        if step > 0 {
            while i < stop {
                out.push(self.token_ids[i as usize]);
                i += step;
            }
        } else {
            while i > stop {
                out.push(self.token_ids[i as usize]);
                i += step;
            }
        }
        Ok(out)
    }

    pub fn block(&self, index: i64) -> Result<Vec<i64>, String> {
        if index < 0 || index >= self.num_blocks() {
            return Err("AssertionError: ".into());
        }
        let start = (index as usize) * self.block_size;
        let end = (start + self.block_size).min(self.token_ids.len());
        Ok(self.token_ids[start..end].to_vec())
    }

    pub fn append_token(&mut self, token_id: i64) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
    }

    pub fn getstate(&self) -> PyVal {
        let last = if self.is_prefill {
            pyval::list_i64(&self.token_ids)
        } else {
            PyVal::Int(i128::from(self.last_token))
        };
        PyVal::Tuple(vec![
            PyVal::Int(i128::from(self.num_tokens)),
            PyVal::Int(i128::from(self.num_prompt_tokens)),
            PyVal::Int(i128::from(self.num_cached_tokens)),
            PyVal::Int(i128::from(self.num_scheduled_tokens)),
            pyval::list_i64(&self.block_table),
            last,
        ])
    }

    pub fn setstate(&mut self, state: &PyVal) -> Result<(), String> {
        let (PyVal::List(items) | PyVal::Tuple(items)) = state else {
            return Err("ValueError: setstate expected a tuple".into());
        };
        if items.len() != 6 {
            return Err("ValueError: not enough values to unpack".into());
        }
        self.num_tokens = as_i64(&items[0])?;
        self.num_prompt_tokens = as_i64(&items[1])?;
        self.num_cached_tokens = as_i64(&items[2])?;
        self.num_scheduled_tokens = as_i64(&items[3])?;
        self.block_table = as_i64_list(&items[4])?;
        match &items[5] {
            PyVal::List(tokens) => {
                self.token_ids = tokens.iter().map(as_i64).collect::<Result<Vec<_>, _>>()?;
                self.last_token = *self
                    .token_ids
                    .last()
                    .ok_or("IndexError: list index out of range")?;
            }
            other => {
                self.token_ids.clear();
                self.last_token = as_i64(other)?;
            }
        }
        Ok(())
    }

    pub fn set_status(&mut self, name: &str) -> Result<(), String> {
        self.status = Status::parse(name)?;
        Ok(())
    }

    pub fn snapshot(&self) -> PyVal {
        PyVal::Map(vec![
            ("seq_id".into(), PyVal::Int(i128::from(self.seq_id))),
            ("status".into(), PyVal::Str(self.status.name().into())),
            ("token_ids".into(), pyval::list_i64(&self.token_ids)),
            ("last_token".into(), PyVal::Int(i128::from(self.last_token))),
            ("num_tokens".into(), PyVal::Int(i128::from(self.num_tokens))),
            (
                "num_prompt_tokens".into(),
                PyVal::Int(i128::from(self.num_prompt_tokens)),
            ),
            (
                "num_cached_tokens".into(),
                PyVal::Int(i128::from(self.num_cached_tokens)),
            ),
            (
                "num_scheduled_tokens".into(),
                PyVal::Int(i128::from(self.num_scheduled_tokens)),
            ),
            ("is_prefill".into(), PyVal::Bool(self.is_prefill)),
            ("block_table".into(), pyval::list_i64(&self.block_table)),
            ("temperature".into(), scalar_py(&self.temperature)),
            ("max_tokens".into(), PyVal::Int(self.max_tokens)),
            ("ignore_eos".into(), PyVal::Bool(self.ignore_eos)),
            (
                "num_completion_tokens".into(),
                PyVal::Int(i128::from(self.num_tokens - self.num_prompt_tokens)),
            ),
            (
                "prompt_token_ids".into(),
                take(&self.token_ids, self.num_prompt_tokens, true),
            ),
            (
                "completion_token_ids".into(),
                take(&self.token_ids, self.num_prompt_tokens, false),
            ),
            (
                "num_blocks".into(),
                PyVal::Int(i128::from(self.num_blocks())),
            ),
            (
                "last_block_num_tokens".into(),
                PyVal::Int(i128::from(self.last_block_num_tokens())),
            ),
            (
                "is_finished".into(),
                PyVal::Bool(self.status == Status::Finished),
            ),
            ("len".into(), PyVal::Int(i128::from(self.len()))),
        ])
    }
}

fn take(tokens: &[i64], at: i64, prefix: bool) -> PyVal {
    let n = usize::try_from(at).unwrap_or(0).min(tokens.len());
    if prefix {
        pyval::list_i64(&tokens[..n])
    } else {
        pyval::list_i64(&tokens[n..])
    }
}

fn scalar_py(value: &Scalar) -> PyVal {
    match *value {
        Scalar::Bool(v) => PyVal::Bool(v),
        Scalar::Int(v) => PyVal::Int(v),
        Scalar::Float(v) => PyVal::Float(v),
    }
}

pub fn as_i64(value: &PyVal) -> Result<i64, String> {
    value
        .as_i128()
        .and_then(|n| i64::try_from(n).ok())
        .ok_or_else(|| "TypeError: expected an int".into())
}

fn as_i64_list(value: &PyVal) -> Result<Vec<i64>, String> {
    let PyVal::List(items) = value else {
        return Err("TypeError: block_table is not a list".into());
    };
    items.iter().map(as_i64).collect()
}

fn adjust(index: i64, len: i64, step: i64) -> i64 {
    let mut index = index;
    if index < 0 {
        index += len;
        if index < 0 {
            return if step < 0 { -1 } else { 0 };
        }
    }
    if index >= len {
        return if step < 0 { len - 1 } else { len };
    }
    index
}

/// Python `slice.indices` for a step that is not zero.
fn slice_indices(len: i64, start: Option<i64>, stop: Option<i64>, step: i64) -> (i64, i64, i64) {
    let start = match start {
        None => {
            if step < 0 {
                len - 1
            } else {
                0
            }
        }
        Some(index) => adjust(index, len, step),
    };
    let stop = match stop {
        None => {
            if step < 0 {
                -1
            } else {
                len
            }
        }
        Some(index) => adjust(index, len, step),
    };
    (start, stop, step)
}
