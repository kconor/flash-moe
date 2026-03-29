# Thunderbolt 4 Dual-Machine Expert Storage

## Context

A single NVMe SSD delivers ~6.1 GB/s for expert reads. Expert I/O is 63% of layer time (4.0ms/layer for the 122B model). A second MacBook Pro over Thunderbolt 4 could double effective storage bandwidth.

Thunderbolt 4 provides 40 Gbps = ~5 GB/s theoretical, ~3-4 GB/s practical after protocol overhead. Not as fast as local NVMe, but additive — the two SSDs can be read in parallel.

## Hardware Setup

- **Machine A** (primary): runs inference, has GPU, reads local experts
- **Machine B** (remote): serves expert data over Thunderbolt, acts as network-attached storage
- **Connection**: Thunderbolt 4 cable, IP networking via Thunderbolt bridge (macOS auto-configures this)

Both machines see each other as a network peer at a 169.254.x.x link-local address or via Thunderbolt Bridge in Network preferences. Achievable throughput: ~3.5 GB/s (measured by others for TB4 IP).

## Version 1: Split Experts (Simple)

### Concept
Even layers (0,2,4,...) on Machine A's SSD. Odd layers (1,3,5,...) on Machine B, accessed via NFS or a custom TCP server. Each layer's expert reads go to one machine only — no cross-machine coordination per layer.

### Setup
```
Machine B:
  # Share the packed_experts directory via NFS
  sudo echo "/path/to/packed_experts -network 169.254.0.0 -mask 255.255.0.0" >> /etc/exports
  sudo nfsd restart

Machine A:
  mkdir -p /Volumes/remote_experts
  mount -t nfs 169.254.x.x:/path/to/packed_experts /Volumes/remote_experts
  # Symlink odd layers to remote, even to local
  for i in $(seq 1 2 47); do
    ln -sf /Volumes/remote_experts/layer_$(printf '%02d' $i).bin packed_experts/layer_$(printf '%02d' $i).bin
  done
```

### Code Changes
None. The engine opens `packed_experts/layer_XX.bin` via `open()` + `pread()`. NFS mounts are transparent to the application — pread on an NFS file issues a network read. The OS handles caching and prefetch.

### Expected Performance
- Local layers: 4.0ms (unchanged)
- Remote layers: ~12ms (5.3MB × 8 experts / 3.5 GB/s ≈ 12ms, plus NFS protocol overhead)
- Average: ~8ms/layer (slower than single-machine because remote layers are 3x slower)

**Verdict: Version 1 is likely SLOWER because NFS per-read latency kills it.** NFS adds ~0.5ms per RPC round-trip on top of bulk transfer. With 8 parallel preads per layer, each one incurs this overhead. The small read size (5.3MB) doesn't amortize the protocol cost well.

### Potential NFS Optimization
- `mount -o rsize=1048576,wsize=1048576` — 1MB NFS block size (default is 32KB, causing many round-trips for 5.3MB)
- `mount -o nconnect=8` — multiple TCP connections for parallel I/O (Linux; macOS support unclear)
- `mount -o async,noac` — disable attribute caching, async writes

## Version 2: Mirrored Experts + Load Balancing (Full)

### Concept
Both machines have ALL expert data. The inference engine reads each expert from whichever source is faster — local SSD or remote machine. A simple load balancer tracks which source is busy and routes requests accordingly.

### Architecture
```
Machine A (inference):                    Machine B (expert server):
  infer.m                                   expert_server (new)
    │                                          │
    ├─ local pread (6.1 GB/s)                  ├─ pread from local SSD
    │                                          │
    └─ TCP request ──── TB4 ────────────────── └─ send expert data back
         (~3.5 GB/s)                              (~3.5 GB/s)
```

### New Component: expert_server (runs on Machine B)

Simple TCP server. Listens on a port. Protocol:
```
Request:  [uint16 layer_idx] [uint16 expert_idx]   (4 bytes)
Response: [uint8 status] [expert_data...]           (1 + EXPERT_SIZE bytes)
```

Single binary, ~200 lines of C. Uses `pread` + `send` with `TCP_NODELAY`. Pre-opens all layer files at startup.

### New Component: Load Balancer (in infer.m)

Replace `async_pread_start` with a smarter dispatcher:

```c
typedef struct {
    int local_inflight;       // number of preads currently running locally
    int remote_inflight;      // number of requests currently on TB4
    double local_avg_us;      // rolling average local read time (microseconds)
    double remote_avg_us;     // rolling average remote read time
    int remote_fd;            // TCP socket to Machine B
} ExpertLoadBalancer;
```

**Decision logic per expert:**
1. If this (layer, expert) is in local page cache → local (instant)
2. If `local_inflight < 4` and `remote_inflight < 4` → send to whichever has lower avg_us
3. If one is fully loaded → send to the other
4. Tie-break: local (lower latency)

**Expected steady-state:** ~5 experts read locally, ~3 remotely per layer (adapts based on cache state). Both SSDs contribute bandwidth in parallel.

### Code Changes

**Machine B — new file: `metal_infer/expert_server.c` (~200 lines)**
- TCP listener on configurable port
- Pre-opens all packed_experts/layer_XX.bin files
- Per-connection: read 4-byte request → pread expert → send response
- Multi-threaded (pthread per connection) or GCD dispatch

**Machine A — changes to `metal_infer/infer.m`:**
- New `--remote HOST:PORT` CLI flag
- `ExpertLoadBalancer` struct and init
- Modified `async_pread_start`: for each expert k, decide local vs remote
  - Local: existing pread path
  - Remote: `send(request)` + `recv(response)` into buf_multi_expert_data[k]
  - Both set `ready[k]` when done (per-expert CMD submits immediately)
- Rolling average update after each completion

**Machine A — no changes to:**
- Metal shaders, config, weight extraction, repacking
- All other inference logic (the expert data arrives in the same Metal buffer regardless of source)

### Expected Performance

Local SSD: 6.1 GB/s, ~0.87ms per 5.3MB expert
Remote TB4: ~3.5 GB/s, ~1.5ms per 5.3MB expert + ~0.1ms protocol overhead

With 8 experts per layer, optimally split 5 local + 3 remote:
- Local bottleneck: 5 × 5.3MB / 6.1 GB/s = 4.3ms (parallel, so ~0.87ms if fully parallel)
- Remote bottleneck: 3 × 5.3MB / 3.5 GB/s = 4.5ms (parallel, so ~1.5ms)
- Combined: max(0.87, 1.5) = ~1.5ms

vs current: 4.0ms (all 8 on local SSD, 5 cold + 3 cached)

**Potential improvement: 4.0ms → 1.5ms expert_io = 2.5ms saved = 39% faster total layer time.**

But this assumes both SSDs are reading in parallel and the TB4 link isn't saturated. Real-world: probably 2.0-2.5ms → 25-35% improvement.

### For the 122B model specifically

Current: 6.4ms/layer, 3.1 tok/s
With dual storage: ~4.0-4.5ms/layer → 4.4-5.0 tok/s (+40-60%)

### Risks
- TB4 link saturation at 3.5 GB/s with 3 parallel 5.3MB reads (15.9MB/layer × 48 layers = 763MB/s sustained — well within TB4 bandwidth)
- TCP overhead per read: connect once, reuse connection, use sendmsg/recvmsg batching
- macOS Thunderbolt networking quirks (may need manual IP config)
- Page cache on Machine B doesn't help Machine A (each has its own cache)

### Testing Plan
1. Benchmark raw TB4 throughput: `iperf3 -c 169.254.x.x`
2. Build expert_server, test with manual TCP client
3. Benchmark single expert fetch time over TB4
4. Integrate into infer.m, benchmark with --timing
5. Compare: local-only vs split vs mirrored+balanced
