# CARRIED-DEBT

Appended at every slice merge: what the slice settled → deferred, with rulings → process lessons. Resolved items are struck through, never deleted.

## S1 (in progress)
### Settled
- (fill at merge)
### Deferred, with rulings
- **Sandbox isolation is Python-level, not OS-level** (ruling R-T2-3, Task 2). `crucible/sandbox/exec.py` blocks outbound sockets with a `sitecustomize.py` shim: it stops *accidental* network use by generated single-function code, which is the failure mode S1 has, but it is not an adversary barrier -- a unit that shells out to `curl` or calls `connect` through `ctypes` still reaches the network. No S1 code path produces such a unit (units are single functions the proposer writes, run by pytest). OS-level isolation (network namespace / bubblewrap) is deferred past S1. The same ruling covers the escaped-`setsid` grandchild: file-backed capture means it can no longer stall the wall cap, but reaping it would need a cgroup or PID namespace.
### Process lessons
- (fill at merge)
