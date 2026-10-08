use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub const TOKENS_PER_BLOCK: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    pub id: BlockId,
    pub ref_count: usize,
    pub last_accessed_step: usize,
}

#[derive(Default)]
pub struct RadixNode {
    pub token_prefix: Vec<u32>,
    pub block_ids: Vec<BlockId>,
    pub children: HashMap<u32, Arc<RwLock<RadixNode>>>,
}

pub struct RadixPrefixCache {
    root: Arc<RwLock<RadixNode>>,
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
}

impl Default for RadixPrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixPrefixCache {
    pub fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(RadixNode::default())),
            hit_count: AtomicUsize::new(0),
            miss_count: AtomicUsize::new(0),
        }
    }

    /// Match prefix tokens to existing cached physical KV block IDs
    pub fn match_prefix(&self, tokens: &[u32]) -> (Vec<BlockId>, usize) {
        let mut matched_blocks = Vec::new();
        let mut matched_tokens = 0;
        let mut current_node = self.root.clone();

        let mut token_idx = 0;
        while token_idx < tokens.len() {
            let token = tokens[token_idx];
            let next_node = {
                let node = current_node.read();
                node.children.get(&token).cloned()
            };

            if let Some(child_arc) = next_node {
                let child = child_arc.read();
                let prefix_len = child.token_prefix.len();
                if token_idx + prefix_len <= tokens.len()
                    && tokens[token_idx..token_idx + prefix_len] == child.token_prefix[..]
                {
                    matched_blocks.extend_from_slice(&child.block_ids);
                    matched_tokens += prefix_len;
                    token_idx += prefix_len;
                    drop(child);
                    current_node = child_arc;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if matched_tokens > 0 {
            self.hit_count.fetch_add(matched_tokens, Ordering::Relaxed);
        } else {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
        }

        (matched_blocks, matched_tokens)
    }

    pub fn insert_prefix(&self, tokens: &[u32], blocks: &[BlockId]) {
        if tokens.is_empty() {
            return;
        }

        let current_node = self.root.clone();
        let first_token = tokens[0];

        let mut root_guard = current_node.write();
        let child = root_guard
            .children
            .entry(first_token)
            .or_insert_with(|| Arc::new(RwLock::new(RadixNode::default())));

        let mut child_guard = child.write();
        child_guard.token_prefix = tokens.to_vec();
        child_guard.block_ids = blocks.to_vec();
    }

    pub fn metrics(&self) -> (usize, usize) {
        (
            self.hit_count.load(Ordering::Relaxed),
            self.miss_count.load(Ordering::Relaxed),
        )
    }
}

pub struct PagedKVPool {
    pub total_blocks: usize,
    free_blocks: RwLock<Vec<BlockId>>,
    active_blocks: RwLock<HashMap<BlockId, PhysicalBlock>>,
    pub radix_cache: Arc<RadixPrefixCache>,
}

impl PagedKVPool {
    pub fn new(total_blocks: usize) -> Self {
        let free = (0..total_blocks).map(BlockId).collect();
        Self {
            total_blocks,
            free_blocks: RwLock::new(free),
            active_blocks: RwLock::new(HashMap::new()),
            radix_cache: Arc::new(RadixPrefixCache::new()),
        }
    }

    pub fn allocate_block(&self, current_step: usize) -> Option<BlockId> {
        let mut free = self.free_blocks.write();
        let block_id = free.pop()?;
        
        let mut active = self.active_blocks.write();
        active.insert(
            block_id,
            PhysicalBlock {
                id: block_id,
                ref_count: 1,
                last_accessed_step: current_step,
            },
        );

        Some(block_id)
    }

    pub fn release_block(&self, block_id: BlockId) {
        let mut active = self.active_blocks.write();
        if let Some(block) = active.get_mut(&block_id) {
            block.ref_count = block.ref_count.saturating_sub(1);
            if block.ref_count == 0 {
                active.remove(&block_id);
                self.free_blocks.write().push(block_id);
            }
        }
    }

    pub fn utilization(&self) -> f32 {
        let free_count = self.free_blocks.read().len();
        (self.total_blocks - free_count) as f32 / self.total_blocks as f32
    }
}
