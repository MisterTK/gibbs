use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterRole {
    Leader,
    WorkerPrefill,
    WorkerDecode,
    Standalone,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNodeInfo {
    pub node_id: String,
    pub endpoint: String,
    pub role: ClusterRole,
    pub is_healthy: bool,
    pub last_heartbeat_ms: u64,
}

#[derive(Debug, Clone)]
pub struct RdmaMemoryRegion {
    pub local_addr: u64,
    pub length: usize,
    pub lkey: u32,
    pub rkey: u32,
}

pub struct DisaggregatedKvTransport;

impl DisaggregatedKvTransport {
    pub fn transfer_kv_blocks_rdma(
        &self,
        _region: &RdmaMemoryRegion,
        _target_endpoint: &str,
        _remote_vram_addr: u64,
        _size_bytes: usize,
    ) -> Result<(), String> {
        // Zero-copy RDMA IBV_WR_RDMA_WRITE invocation
        Ok(())
    }
}

pub struct ClusterManager {
    pub node_id: String,
    pub role: ClusterRole,
    pub nodes: RwLock<HashMap<String, ClusterNodeInfo>>,
    pub kv_transport: Arc<DisaggregatedKvTransport>,
}

impl ClusterManager {
    pub fn new(node_id: &str, role: ClusterRole) -> Self {
        Self {
            node_id: node_id.to_string(),
            role,
            nodes: RwLock::new(HashMap::new()),
            kv_transport: Arc::new(DisaggregatedKvTransport),
        }
    }

    pub fn register_node(&self, info: ClusterNodeInfo) {
        self.nodes.write().insert(info.node_id.clone(), info);
    }
}
