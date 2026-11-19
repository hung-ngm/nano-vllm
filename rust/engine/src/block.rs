//! `Block` and `BlockManager`, including the xxh64 prefix-cache hash.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use xxhash_rust::xxh64::Xxh64;

use crate::pyval::{self, PyVal};
use crate::sampling::SamplingParams;
use crate::sequence::Sequence;

#[derive(Clone, Debug)]
pub struct Block {
    pub block_id: i64,
    pub ref_count: i64,
    pub hash: i128,
    pub token_ids: Vec<i64>,
}

impl Block {
    fn new(block_id: i64) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: -1,
            token_ids: Vec::new(),
        }
    }

    fn update(&mut self, hash: i128, token_ids: Vec<i64>) {
        self.hash = hash;
        self.token_ids = token_ids;
    }

    fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = -1;
        self.token_ids.clear();
    }

    fn snapshot(&self) -> PyVal {
        PyVal::Map(vec![
            ("id".into(), PyVal::Int(i128::from(self.block_id))),
            ("ref".into(), PyVal::Int(i128::from(self.ref_count))),
            ("hash".into(), PyVal::Int(self.hash)),
            ("tokens".into(), pyval::list_i64(&self.token_ids)),
        ])
    }
}

#[derive(Clone, Debug)]
pub struct BlockManager {
    pub block_size: i64,
    pub blocks: Vec<Block>,
    hash_to_block_id: BTreeMap<i128, i64>,
    free_block_ids: VecDeque<i64>,
    used_block_ids: BTreeSet<i64>,
}

impl BlockManager {
    pub fn new(num_blocks: i64, block_size: i64) -> Self {
        let n = usize::try_from(num_blocks).unwrap_or(0);
        Self {
            block_size,
            blocks: (0..n)
                .map(|i| Block::new(i64::try_from(i).unwrap_or(0)))
                .collect(),
            hash_to_block_id: BTreeMap::new(),
            free_block_ids: (0..n).map(|i| i64::try_from(i).unwrap_or(0)).collect(),
            used_block_ids: BTreeSet::new(),
        }
    }

    /// xxh64 of an optional 8-byte little-endian prefix, then int64 token bytes.
    pub fn compute_hash(token_ids: &[i64], prefix: i128) -> u64 {
        let mut hasher = Xxh64::new(0);
        if prefix != -1 {
            hasher.update(&(prefix as u64).to_le_bytes());
        }
        let mut bytes = Vec::with_capacity(token_ids.len() * 8);
        for id in token_ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        hasher.update(&bytes);
        hasher.digest()
    }

    pub fn can_allocate(&self, seq: &Sequence) -> i64 {
        let mut hash = -1_i128;
        let mut num_cached_blocks = 0_i64;
        let mut num_new_blocks = seq.num_blocks();
        for i in 0..seq.num_blocks().saturating_sub(1) {
            let Ok(token_ids) = seq.block(i) else {
                break;
            };
            hash = i128::from(Self::compute_hash(&token_ids, hash));
            let block_id = self.hash_to_block_id.get(&hash).copied().unwrap_or(-1);
            if block_id == -1 || self.blocks[block_id as usize].token_ids != token_ids {
                break;
            }
            num_cached_blocks += 1;
            if self.used_block_ids.contains(&block_id) {
                num_new_blocks -= 1;
            }
        }
        if (self.free_block_ids.len() as i64) < num_new_blocks {
            -1
        } else {
            num_cached_blocks
        }
    }

    pub fn allocate(&mut self, seq: &mut Sequence, num_cached_blocks: i64) -> Result<(), String> {
        if !seq.block_table.is_empty() {
            return Err("AssertionError: ".into());
        }
        let mut hash = -1_i128;
        for i in 0..num_cached_blocks {
            let token_ids = seq.block(i)?;
            hash = i128::from(Self::compute_hash(&token_ids, hash));
            let block_id = *self.hash_to_block_id.get(&hash).ok_or("KeyError: hash")?;
            if self.used_block_ids.contains(&block_id) {
                self.blocks[block_id as usize].ref_count += 1;
            } else {
                self.blocks[block_id as usize].ref_count = 1;
                let pos = self
                    .free_block_ids
                    .iter()
                    .position(|id| *id == block_id)
                    .ok_or("ValueError: block is not free")?;
                self.free_block_ids.remove(pos);
                self.used_block_ids.insert(block_id);
            }
            seq.block_table.push(block_id);
        }
        for _ in num_cached_blocks..seq.num_blocks() {
            seq.block_table.push(self.allocate_block()?);
        }
        seq.num_cached_tokens = num_cached_blocks * self.block_size;
        Ok(())
    }

    pub fn deallocate(&mut self, seq: &mut Sequence) -> Result<(), String> {
        for block_id in seq.block_table.iter().rev().copied() {
            let block = &mut self.blocks[block_id as usize];
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.deallocate_block(block_id)?;
            }
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
        Ok(())
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        let need = i64::from(seq.len() % self.block_size == 1);
        (self.free_block_ids.len() as i64) >= need
    }

    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<(), String> {
        if seq.len() % self.block_size == 1 {
            let id = self.allocate_block()?;
            seq.block_table.push(id);
        }
        Ok(())
    }

    pub fn hash_blocks(&mut self, seq: &Sequence) -> Result<(), String> {
        let start = seq.num_cached_tokens / self.block_size;
        let end = (seq.num_cached_tokens + seq.num_scheduled_tokens) / self.block_size;
        if start == end {
            return Ok(());
        }
        let mut hash = if start > 0 {
            let id = seq.block_table[(start as usize) - 1];
            self.blocks[id as usize].hash
        } else {
            -1
        };
        for i in start..end {
            let block_id = seq.block_table[i as usize];
            let token_ids = seq.block(i)?;
            hash = i128::from(Self::compute_hash(&token_ids, hash));
            self.blocks[block_id as usize].update(hash, token_ids);
            self.hash_to_block_id.insert(hash, block_id);
        }
        Ok(())
    }

    fn allocate_block(&mut self) -> Result<i64, String> {
        let block_id = self
            .free_block_ids
            .pop_front()
            .ok_or("IndexError: pop from empty deque")?;
        let block = &mut self.blocks[block_id as usize];
        if block.ref_count != 0 {
            return Err("AssertionError: ".into());
        }
        if block.hash != -1 && self.hash_to_block_id.get(&block.hash) == Some(&block_id) {
            self.hash_to_block_id.remove(&block.hash);
        }
        block.reset();
        self.used_block_ids.insert(block_id);
        Ok(block_id)
    }

    fn deallocate_block(&mut self, block_id: i64) -> Result<(), String> {
        if self.blocks[block_id as usize].ref_count != 0 {
            return Err("AssertionError: ".into());
        }
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push_back(block_id);
        Ok(())
    }

    pub fn snapshot(&self) -> PyVal {
        let hashes = self
            .hash_to_block_id
            .iter()
            .map(|(hash, id)| PyVal::List(vec![PyVal::Int(*hash), PyVal::Int(i128::from(*id))]))
            .collect();
        PyVal::Map(vec![
            ("block_size".into(), PyVal::Int(i128::from(self.block_size))),
            (
                "free".into(),
                pyval::list_i64(&self.free_block_ids.iter().copied().collect::<Vec<_>>()),
            ),
            (
                "used".into(),
                pyval::list_i64(&self.used_block_ids.iter().copied().collect::<Vec<_>>()),
            ),
            ("hash_to_block".into(), PyVal::List(hashes)),
            (
                "blocks".into(),
                PyVal::List(self.blocks.iter().map(Block::snapshot).collect()),
            ),
        ])
    }

    pub fn fresh_seq(token_ids: &[i64]) -> Result<Sequence, String> {
        Sequence::try_new(token_ids, SamplingParams::try_new(&PyVal::Map(vec![]))?)
    }
}
