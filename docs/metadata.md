# ModelExpress Metadata Architecture

This document describes the metadata storage and coordination layer for ModelExpress P2P transfers.

## Overview

ModelExpress P2P transfers require coordination between source and target GPU workers:

1. **Source** loads model weights, registers tensors with a transfer backend (NIXL or Mooncake), and publishes metadata to the MX server
2. **Target** queries the MX server for available sources, fetches tensor metadata on demand, and executes RDMA transfers
3. **Status** transitions (`INITIALIZING` -> `READY` -> `STALE`) signal when sources are available for transfers

## Key Concepts

### Source Identity and Content-Addressed Keys

Every source is identified by a `SourceIdentity` proto containing all fields that affect tensor layout compatibility:

| Field | Example | Purpose |
|-------|---------|---------|
| `mx_version` | `"0.3.0"` | Format compatibility across upgrades |
| `mx_source_type` | `WEIGHTS`, `LORA`, `CUDA_GRAPH` | Type of tensors being served |
| `model_name` | `"deepseek-ai/DeepSeek-V3"` | Model identifier |
| `backend_framework` | `VLLM`, `SGLANG`, `TRT_LLM` | Inference framework |
| `tensor_parallel_size` | `8` | TP degree |
| `pipeline_parallel_size` | `2` | PP degree |
| `expert_parallel_size` | `4` | EP degree (MoE models) |
| `dtype` | `"bfloat16"` | Weight data type |
| `quantization` | `"fp8"`, `""` | Quantization method |
| `extra_parameters` | `{}` | Framework-specific config |

The server computes `mx_source_id = SHA256(canonical_json(identity))[:16]` -- a 16-char hex key used to address all metadata for sources with identical configuration. This is content-addressed: two sources with the same identity hash to the same `mx_source_id`, enabling automatic peer discovery.

### Multi-Instance Support

Multiple replicas of the same model (same `SourceIdentity`) can coexist. Each GPU worker process generates a unique `worker_id` (`uuid4().hex[:8]`) at startup. The combination `(mx_source_id, worker_id)` uniquely identifies one worker's metadata.

Each worker publishes independently -- no inter-worker coordination or barriers required.

### Worker Rank

Workers use `torch.distributed.get_rank()` as their global rank, which captures both tensor-parallel and pipeline-parallel position. This is stored as `worker_rank` in metadata so targets can find a peer with a matching rank.

## gRPC API

```protobuf
service P2pService {
  rpc PublishMetadata(PublishMetadataRequest) returns (PublishMetadataResponse);
  rpc ListSources(ListSourcesRequest) returns (ListSourcesResponse);
  rpc GetMetadata(GetMetadataRequest) returns (GetMetadataResponse);
  rpc UpdateStatus(UpdateStatusRequest) returns (UpdateStatusResponse);
}
```

### PublishMetadata

Called once per GPU worker after loading weights and registering with the transfer backend. The server computes `mx_source_id` from `identity` and returns it to the client.

```protobuf
PublishMetadataRequest {
  identity: SourceIdentity    // Server computes mx_source_id from this
  worker: WorkerMetadata       // One worker per call (rank, backend metadata, tensors)
  worker_id: string            // Unique per GPU process (uuid4 hex[:8])
}
```

### ListSources

Lightweight listing -- returns `SourceInstanceRef` entries (no tensor data). Clients filter by `worker_rank` to find matching peers, then call `GetMetadata` for the chosen one.

```protobuf
ListSourcesRequest {
  identity: SourceIdentity         // Optional: filter by source identity
  status_filter: SourceStatus      // Optional: e.g., SOURCE_STATUS_READY
}

ListSourcesResponse {
  instances: [SourceInstanceRef]   // One entry per worker
}

SourceInstanceRef {
  mx_source_id: string    // 16-char hex
  worker_id: string       // Unique worker identifier
  model_name: string      // Human-readable
  worker_rank: uint32     // Global rank for peer matching
}
```

### GetMetadata

Fetches full tensor metadata (MB-scale) for one specific worker. Called on demand after filtering `ListSources` results.

```protobuf
GetMetadataRequest {
  mx_source_id: string   // From ListSources or PublishMetadata response
  worker_id: string      // From ListSources or PublishMetadata response
}
```

### UpdateStatus

Transitions a worker's lifecycle status. Called periodically by the client-side heartbeat thread to refresh `updated_at`, and on shutdown to mark `STALE`.

```protobuf
UpdateStatusRequest {
  mx_source_id: string
  worker_id: string
  worker_rank: uint32
  status: SourceStatus   // INITIALIZING -> READY -> STALE
}
```

## Source Lifecycle

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING : PublishMetadata
    INITIALIZING --> READY : Heartbeat (NIXL healthy)
    READY --> READY : Heartbeat refreshes updated_at
    READY --> STALE : atexit or reaper timeout
    INITIALIZING --> STALE : Reaper timeout
    STALE --> Deleted : Reaper GC
```

- **INITIALIZING**: Worker has published metadata but heartbeat hasn't confirmed NIXL health yet
- **READY**: Worker is healthy and accepting RDMA connections. Heartbeat refreshes `updated_at` every `MX_HEARTBEAT_INTERVAL_SECS` (default 30s)
- **STALE**: Worker is no longer available. Set by the client `atexit` handler on clean shutdown (SIGTERM), or by the server-side reaper when `updated_at` exceeds `MX_HEARTBEAT_TIMEOUT_SECS` (default 90s)
- **Deleted**: Reaper garbage-collects stale entries after `MX_GC_TIMEOUT_SECS` (default 3600s)

## Backend Implementations

Configured via `MX_METADATA_BACKEND` environment variable:

| Value | Backend | Use Case |
|-------|---------|----------|
| `redis` | Redis | Production with Redis |
| `kubernetes` / `k8s` / `crd` | Kubernetes CRDs | K8s-native deployments |

### Redis Backend

#### Storage Layout

Two types of Redis keys per source:

**Source index key** -- `mx:source:{source_id}` (Redis Hash)

| Field | Value | Purpose |
|-------|-------|---------|
| `__attributes__` | JSON of all `SourceIdentity` fields | Stored once per source, avoids duplication |
| `{worker_id}` | `"{global_rank}"` | Presence marker with rank for fast listing |

**Worker data key** -- `mx:source:{source_id}:{worker_id}` (Redis Hash)

| Field | Value | Purpose |
|-------|-------|---------|
| `"{worker_rank}"` | JSON `WorkerRecordJson` | Full tensor metadata for one rank |

Global listing uses `SCAN` with pattern `mx:source:????????????????` (exactly 16 hex chars) to enumerate source index keys without a secondary index.

No Redis TTL is applied to keys. Stale detection and cleanup are handled by the server-side reaper (see Source Lifecycle above).

#### Example Redis State

```
# Source index -- identity stored once, workers as presence markers
mx:source:a1b2c3d4e5f67890
  __attributes__  ->  {"model_name":"deepseek-ai/DeepSeek-V3","mx_version":"0.3.0",...}
  f3a2b1c4        ->  "0"    # worker_id f3a2b1c4, global rank 0
  e7d6c5b8        ->  "1"    # worker_id e7d6c5b8, global rank 1

# Worker data -- full tensor metadata
mx:source:a1b2c3d4e5f67890:f3a2b1c4
  "0"  ->  {"worker_rank":0,"backend_type":"nixl","nixl_metadata":[...],"tensors":[...],"status":2,...}

mx:source:a1b2c3d4e5f67890:e7d6c5b8
  "1"  ->  {"worker_rank":1,"backend_type":"nixl","nixl_metadata":[...],"tensors":[...],"status":2,...}
```

#### JSON Schemas

**WorkerRecordJson** (stored per rank in worker data hash):
```json
{
  "worker_rank": 0,
  "backend_type": "nixl",
  "nixl_metadata": [222, 173, 190, 239],
  "transfer_engine_session_id": null,
  "tensors": [
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "addr": "139948187451390",
      "size": "134217728",
      "device_id": 0,
      "dtype": "bfloat16"
    }
  ],
  "status": 2,
  "updated_at": 1700000000000
}
```

`addr` and `size` are serialized as strings to avoid JSON precision loss with large u64 values.

### Kubernetes CRD Backend

Uses `ModelMetadata` CRDs for metadata and `ConfigMap`s for tensor descriptors (to avoid etcd size limits).

**CRD name format**: `mx-source-{source_id}-{worker_id}`

**ConfigMap name format**: `mx-source-{source_id}-{worker_id}-tensors-worker-{rank}`

ConfigMaps use `ownerReferences` pointing to the parent CRD so they are garbage-collected automatically.

#### Example CRD

```bash
kubectl get modelmetadatas -n <namespace>
kubectl get modelmetadata mx-source-a1b2c3d4e5f67890-f3a2b1c4 -n <namespace> -o yaml
```

```yaml
apiVersion: modelexpress.nvidia.com/v1alpha1
kind: ModelMetadata
metadata:
  name: mx-source-a1b2c3d4e5f67890-f3a2b1c4
  labels:
    modelexpress.nvidia.com/mx-source-id: a1b2c3d4e5f67890
    modelexpress.nvidia.com/mx-worker-id: f3a2b1c4
spec:
  modelName: deepseek-ai/DeepSeek-V3
status:
  worker:
    workerRank: 0
    backendType: nixl
    nixlMetadata: <base64>
    tensorCount: 1327
    tensorConfigMap: mx-source-a1b2c3d4e5f67890-f3a2b1c4-tensors-worker-0
    status: Ready
    updatedAt: "2025-11-14T22:13:20Z"
  conditions:
    - type: Ready
      status: "True"
      reason: WorkerReady
      message: Worker is ready
      lastTransitionTime: "2025-11-14T22:13:20Z"
  observedGeneration: 1
  publishedAt: "2025-11-14T22:13:20Z"
```

## Client Workflow

The `MxModelLoader` evaluates strategies in priority order (`RdmaStrategy` -> `ModelStreamerStrategy` -> `GdsStrategy` -> `DefaultStrategy`) and runs the first available one. The four diagrams below describe the per-strategy flow. Every diagram shares the same five participants so the boundaries between modules are explicit:

1. **Inference Engine** -- vLLM (or other host) calling `load_model`
2. **ModelExpress Client** -- in-process Python loader, NIXL agent, gRPC client
3. **ModelExpress Server** -- central metadata service (Redis or Kubernetes backend)
4. **Remote ModelExpress Worker** -- another worker that already published metadata as an RDMA source
5. **Storage Origin** -- HuggingFace Hub, S3, GCS, Azure Blob, or local disk

When a strategy does not interact with a given module, that lifeline simply has no messages on it.

### DefaultStrategy (vLLM `DefaultModelLoader`, CPU-staged + HF Hub download)

Activated as the always-available fallback. Uses vLLM's stock `DefaultModelLoader` to download safetensors from HuggingFace Hub (or read them from the local HF cache), stage them through CPU, copy to GPU, then publish metadata so this worker can serve future targets via RDMA.

```mermaid
sequenceDiagram
    participant Engine as Inference Engine
    participant Client as ModelExpress Client
    participant Server as ModelExpress Server
    participant Remote as Remote ModelExpress Worker
    participant Storage as Storage Origin (HuggingFace / S3)

    Engine->>Client: load_model(model_name, identity)
    Client->>Storage: HuggingFace Hub download (snapshot_download)
    Storage-->>Client: safetensors files (local cache)
    Client->>Client: DefaultModelLoader.load_weights() -- read files, CPU stage, copy to GPU
    Client->>Client: process_weights_after_loading() (capture tensor attrs)
    Client->>Client: register_tensors() -- init NIXL agent, register GPU buffers
    Client->>Server: PublishMetadata(identity, worker, worker_id)
    Server-->>Client: mx_source_id (status=INITIALIZING)
    Client->>Client: Start HeartbeatThread
    loop Every MX_HEARTBEAT_INTERVAL_SECS
        Client->>Server: UpdateStatus(mx_source_id, worker_id, rank, READY)
    end
    Client-->>Engine: Model ready for inference
    Note over Remote: Idle for this load -- this worker may later serve as a source<br/>for other workers' RdmaStrategy.
```

### GdsStrategy (GPUDirect Storage, file -> GPU direct read)

Activated when `is_gds_available()` returns true. `MxGdsLoader` reads safetensors directly from disk into GPU memory, bypassing CPU staging. Falls through to the next strategy on failure.

```mermaid
sequenceDiagram
    participant Engine as Inference Engine
    participant Client as ModelExpress Client
    participant Server as ModelExpress Server
    participant Remote as Remote ModelExpress Worker
    participant Storage as Storage Origin (HuggingFace / S3)

    Engine->>Client: load_model(model_name, identity)
    Client->>Client: is_gds_available() -- check GDS hardware/driver
    Note over Storage: Files must already be on a GDS-capable filesystem.<br/>If not present, the operator pre-stages them out of band<br/>(this strategy does not download).
    Client->>Storage: GDS read (cuFile) -- file -> GPU memory directly
    Storage-->>Client: tensor bytes streamed into GPU buffers
    Client->>Client: model.load_weights(weights_iter)
    Client->>Client: process_weights_after_loading()
    Client->>Client: register_tensors() -- init NIXL agent, register GPU buffers
    Client->>Server: PublishMetadata(identity, worker, worker_id)
    Server-->>Client: mx_source_id (status=INITIALIZING)
    loop Every MX_HEARTBEAT_INTERVAL_SECS
        Client->>Server: UpdateStatus(... READY)
    end
    Client-->>Engine: Model ready for inference
    Note over Remote: Not contacted -- this worker will become a future RDMA source.
```

### ModelStreamerStrategy (`runai-model-streamer`, object storage / local)

Activated when `MX_MODEL_URI` is set and `runai_model_streamer` is installed. Streams safetensors concurrently from `s3://`, `gs://`, `az://`, an absolute local path, or an HF model ID resolved via `HF_HUB_CACHE`.

```mermaid
sequenceDiagram
    participant Engine as Inference Engine
    participant Client as ModelExpress Client
    participant Server as ModelExpress Server
    participant Remote as Remote ModelExpress Worker
    participant Storage as Storage Origin (HuggingFace / S3)

    Engine->>Client: load_model(model_name, identity)
    Client->>Client: _resolve_model_uri(MX_MODEL_URI) -- s3://, gs://, az://, /path, or HF id
    Client->>Storage: list_safetensors(model_uri)
    Storage-->>Client: file URI list
    Client->>Storage: SafetensorsStreamer.stream_files(file_uris) -- concurrent reads
    Storage-->>Client: tensor bytes (streamed, CPU-staged then GPU copy)
    Client->>Client: model.load_weights(weights_iter) (clones each tensor)
    Client->>Client: process_weights_after_loading()
    Client->>Client: register_tensors() -- init NIXL agent, register GPU buffers
    Client->>Server: PublishMetadata(identity, worker, worker_id)
    Server-->>Client: mx_source_id (status=INITIALIZING)
    loop Every MX_HEARTBEAT_INTERVAL_SECS
        Client->>Server: UpdateStatus(... READY)
    end
    Client-->>Engine: Model ready for inference
    Note over Remote: Not contacted -- this worker will become a future RDMA source.
```

### RdmaStrategy (NIXL P2P, receive from a READY same-rank source)

Highest-priority strategy. Discovers an existing source via `ListSources`, fetches per-worker metadata, transfers fully-processed weights over RDMA (NIXL, optionally with a lightweight P2P manifest fetched directly from the source worker), then republishes its own metadata so it can serve later workers.

```mermaid
sequenceDiagram
    participant Engine as Inference Engine
    participant Client as ModelExpress Client
    participant Server as ModelExpress Server
    participant Remote as Remote ModelExpress Worker
    participant Storage as Storage Origin (HuggingFace / S3)

    Engine->>Client: load_model(model_name, identity)
    Client->>Client: is_nixl_available() and check_transfer_allowed() (skip MLA)
    Client->>Server: ListSources(identity, status=READY)
    Server-->>Client: [SourceInstanceRef, ...]
    Client->>Client: Filter by worker_rank, shuffle, take up to MAX_SOURCE_RETRIES
    Client->>Client: DummyModelLoader.load_weights() + process_weights_after_loading()
    Client->>Client: register_tensors() -- init NIXL agent, register GPU buffers
    loop For each candidate
        Client->>Server: GetMetadata(mx_source_id, worker_id)
        Server-->>Client: WorkerMetadata (tensors or worker_grpc_endpoint, nixl_metadata, agent_name)
        opt P2P mode (worker_grpc_endpoint set)
            Client->>Remote: WorkerService.GetTensorManifest(mx_source_id)
            Remote-->>Client: tensor descriptors
            Client->>Remote: NIXL fetch_remote_metadata(host, port)
            Remote-->>Client: NIXL agent metadata blob
        end
        Client->>Remote: NIXL RDMA read (GPU -> GPU, fully-processed tensors)
        Remote-->>Client: tensor bytes via RDMA
        alt SourceTransferError or ManifestMismatchError
            Client->>Client: rollback NIXL state, try next candidate
        else Success
            Client->>Client: torch.cuda.synchronize()
        end
    end
    Note over Storage: Not contacted on the RDMA happy path.<br/>If all candidates fail, MxModelLoader falls through to<br/>ModelStreamer / GDS / Default which read from Storage.
    Client->>Server: PublishMetadata(identity, worker, worker_id) -- become a source
    Server-->>Client: mx_source_id (status=INITIALIZING)
    loop Every MX_HEARTBEAT_INTERVAL_SECS
        Client->>Server: UpdateStatus(... READY)
    end
    Client-->>Engine: Model ready for inference
```

### Three-Tier Loading Strategy

The `MxModelLoader` (`--load-format mx`) auto-detects the best loading strategy:

1. **RDMA** -- If `ListSources` returns READY instances with matching rank, receive weights via NIXL/Mooncake
2. **GDS** -- If no source available and GPUDirect Storage is available, load directly from file to GPU
3. **Disk** -- Standard vLLM `DefaultModelLoader` as final fallback

After loading by any path, the worker registers its tensors and publishes metadata so future workers can discover it as an RDMA source.

## Transfer Backends

`WorkerMetadata` uses a `oneof backend_metadata` field supporting multiple transfer backends:

| Backend | Field | Description |
|---------|-------|-------------|
| NIXL | `nixl_metadata` (bytes) | Serialized NIXL agent blob for RDMA connections |
| Mooncake | `transfer_engine_session_id` (string) | TransferEngine session ID (`"ip:port"`) |

The `backend_type` discriminator is persisted in storage for unambiguous deserialization.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_METADATA_BACKEND` | (required) | `redis` or `kubernetes` |
| `MX_SERVER_ADDRESS` | `modelexpress-server:8001` | gRPC server address |
| `MX_REDIS_HOST` / `REDIS_HOST` | `localhost` | Redis host |
| `MX_REDIS_PORT` / `REDIS_PORT` | `6379` | Redis port |
| `REDIS_URL` | (computed) | Full Redis URL (overrides host/port) |
| `MX_METADATA_NAMESPACE` / `POD_NAMESPACE` | `default` | K8s namespace for CRD backend |
| `MX_HEARTBEAT_INTERVAL_SECS` | `30` | Client heartbeat frequency |
| `MX_HEARTBEAT_TIMEOUT_SECS` | `90` | Server reaper staleness threshold |
| `MX_REAPER_SCAN_INTERVAL_SECS` | `30` | Server reaper scan frequency |
| `MX_GC_TIMEOUT_SECS` | `3600` | Time before stale entries are deleted |
| `MX_CONTIGUOUS_REG` | `0` | Use contiguous region registration (experimental) |

## Debugging

### Verify server connectivity

```bash
grpcurl -plaintext <server_host>:8001 list
grpcurl -plaintext -d '{}' <server_host>:8001 model_express.p2p.P2pService/ListSources
```

### Inspect Redis state

```bash
redis-cli KEYS "mx:source:*"
redis-cli HGETALL "mx:source:<source_id>"
redis-cli HGETALL "mx:source:<source_id>:<worker_id>"
```

### Inspect K8s state

```bash
kubectl get modelmetadatas -n <namespace>
kubectl get configmaps -l modelexpress.nvidia.com/mx-source-id=<source_id> -n <namespace>
```

### Common failures

| Symptom | Likely Cause |
|---------|-------------|
| `ListSources` returns empty | No source has published + updated status to READY yet |
| `GetMetadata` returns `found: false` | Worker was garbage-collected by reaper, or wrong `mx_source_id`/`worker_id` |
| Target stuck waiting | Source still loading (check source pod logs for progress) |
| K8s CRs missing | RBAC issue -- check source logs and service account permissions |
| Stale metadata after redeploy | Reaper marks stale within 90s. For immediate cleanup: `FLUSHDB` |
| Transfer failure with address errors | Source pod restarted -- GPU addresses are invalid. Target retries next candidate (max 3) |
