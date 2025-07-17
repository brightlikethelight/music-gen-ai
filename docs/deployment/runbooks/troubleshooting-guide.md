# Deployment Troubleshooting Guide

This guide provides step-by-step troubleshooting procedures for common deployment issues.

## Quick Reference

| Symptom | Possible Cause | Page |
|---------|---------------|------|
| API returns 503 | Service not ready, health check failing | [Service Unavailable](#1-service-unavailable-503) |
| High error rate | Code bug, dependency issue | [High Error Rate](#2-high-error-rate) |
| Slow response times | Resource constraints, DB issues | [Performance Degradation](#3-performance-degradation) |
| Memory leaks | Code issue, cache growth | [Memory Issues](#4-memory-issues) |
| Database errors | Connection pool, slow queries | [Database Problems](#5-database-problems) |
| GPU errors | Driver issues, OOM | [GPU Issues](#6-gpu-issues) |
| Deploy stuck | Resource limits, image pull | [Deployment Stuck](#7-deployment-stuck) |

---

## 1. Service Unavailable (503)

### Symptoms
- API returns 503 Service Unavailable
- Health checks failing
- Pods in CrashLoopBackOff

### Diagnostic Steps

```bash
# 1. Check pod status
kubectl get pods -n production -l app=musicgen-api
kubectl describe pod <pod-name> -n production

# 2. Check recent events
kubectl get events -n production --sort-by='.lastTimestamp' | tail -20

# 3. Check logs
kubectl logs <pod-name> -n production --tail=100
kubectl logs <pod-name> -n production --previous  # If pod restarted

# 4. Check health endpoint directly
kubectl exec <pod-name> -n production -- curl -s localhost:8000/health
```

### Common Causes & Solutions

#### A. Configuration Error
```bash
# Check environment variables
kubectl exec <pod-name> -n production -- env | grep -E "DATABASE|REDIS|API"

# Solution: Fix ConfigMap/Secret
kubectl edit configmap musicgen-config -n production
kubectl rollout restart deployment musicgen-api -n production
```

#### B. Database Connection Failed
```bash
# Test database connection
kubectl exec <pod-name> -n production -- nc -zv postgres.musicgen.ai 5432

# Check connection string
kubectl get secret musicgen-secrets -n production -o jsonpath='{.data.DATABASE_URL}' | base64 -d

# Solution: Fix connection string or network policy
kubectl edit secret musicgen-secrets -n production
```

#### C. Dependency Service Down
```bash
# Check Redis
kubectl exec <pod-name> -n production -- redis-cli -h redis.musicgen.ai ping

# Check external services
kubectl exec <pod-name> -n production -- curl -s https://api.external-service.com/health

# Solution: Wait for dependencies or use fallback
```

#### D. Resource Limits Hit
```bash
# Check resource usage
kubectl top pod <pod-name> -n production

# Solution: Increase limits
kubectl edit deployment musicgen-api -n production
# Update resources.limits.memory and resources.limits.cpu
```

### Recovery Steps
1. **Immediate**: Scale to 0 and back to force fresh pods
   ```bash
   kubectl scale deployment musicgen-api -n production --replicas=0
   kubectl scale deployment musicgen-api -n production --replicas=3
   ```

2. **If persists**: Rollback to previous version
   ```bash
   kubectl rollout undo deployment musicgen-api -n production
   ```

---

## 2. High Error Rate

### Symptoms
- Error rate >5%
- 500 errors in logs
- Customer complaints

### Diagnostic Steps

```bash
# 1. Check error metrics
curl -s http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])

# 2. Analyze error logs
kubectl logs -n production -l app=musicgen-api --tail=1000 | grep -E "ERROR|CRITICAL|Exception"

# 3. Check for patterns
kubectl logs -n production -l app=musicgen-api --tail=1000 | grep ERROR | awk '{print $8}' | sort | uniq -c | sort -rn

# 4. Trace specific errors
kubectl logs -n production -l app=musicgen-api | grep -A 10 -B 10 "request_id=xxx"
```

### Common Causes & Solutions

#### A. New Code Bug
```python
# Identify error location
kubectl logs -n production -l app=musicgen-api | grep -E "Traceback|File.*line" | tail -20

# Solution: Hot fix or rollback
# Option 1: Deploy hotfix
git checkout -b hotfix/error-fix
# Make fix
git commit -m "Fix: Handle null case in generation"
./scripts/emergency_deploy.sh hotfix/error-fix

# Option 2: Rollback
kubectl rollout undo deployment musicgen-api -n production
```

#### B. Rate Limiting Hit
```bash
# Check rate limit metrics
curl -s http://prometheus:9090/api/v1/query?query=rate_limit_exceeded_total

# Solution: Increase limits temporarily
kubectl exec <pod-name> -n production -- redis-cli SET rate_limit_multiplier 2 EX 3600
```

#### C. External API Failures
```bash
# Check external service calls
kubectl logs -n production -l app=musicgen-api | grep -E "external_api|timeout|connection"

# Solution: Enable circuit breaker
kubectl exec <pod-name> -n production -- python -c "
import redis
r = redis.Redis(host='redis.musicgen.ai')
r.set('circuit_breaker:external_api', 'open', ex=300)
"
```

### Recovery Steps
1. **Identify scope**: Determine affected endpoints
2. **Mitigate**: Enable feature flags to disable problematic features
3. **Fix**: Deploy patch or rollback
4. **Verify**: Monitor error rates return to normal

---

## 3. Performance Degradation

### Symptoms
- Response times >2s p95
- Timeouts increasing
- Queue backlog growing

### Diagnostic Steps

```bash
# 1. Check response times
curl -s http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))

# 2. Identify slow endpoints
curl -s http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m])) | jq -r '.data.result[] | "\(.metric.handler): \(.value[1])"' | sort -k2 -rn

# 3. Check database performance
kubectl exec postgres-primary -- psql -U musicgen -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# 4. Check CPU/Memory pressure
kubectl top nodes
kubectl top pods -n production
```

### Common Causes & Solutions

#### A. Database Slow Queries
```sql
-- Identify slow queries
SELECT 
    query,
    calls,
    mean_exec_time,
    total_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC;

-- Solution: Add missing index
CREATE INDEX CONCURRENTLY idx_generations_user_created 
ON generations(user_id, created_at);

-- Kill long-running queries
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE query_start < NOW() - INTERVAL '5 minutes'
AND state = 'active';
```

#### B. Cache Miss Storm
```bash
# Check cache hit rate
kubectl exec <pod-name> -n production -- redis-cli INFO stats | grep hit

# Solution: Warm cache
kubectl exec <pod-name> -n production -- python scripts/warm_cache.py

# Increase cache size
kubectl exec <pod-name> -n production -- redis-cli CONFIG SET maxmemory 4gb
```

#### C. Resource Saturation
```bash
# Check for CPU throttling
kubectl describe pod <pod-name> -n production | grep -A 5 "Limits:"

# Solution: Increase resources or scale out
kubectl scale deployment musicgen-api -n production --replicas=5

# Or increase limits
kubectl patch deployment musicgen-api -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
```

### Recovery Steps
1. **Quick win**: Clear caches and restart pods
2. **Scale out**: Add more replicas
3. **Optimize**: Fix slow queries, add caches
4. **Monitor**: Watch metrics return to baseline

---

## 4. Memory Issues

### Symptoms
- OOMKilled pods
- Increasing memory usage
- Gradual performance degradation

### Diagnostic Steps

```bash
# 1. Check for OOM kills
kubectl describe pod <pod-name> -n production | grep -i "OOMKilled"
dmesg | grep -i "killed process"

# 2. Monitor memory usage over time
kubectl top pod <pod-name> -n production --containers

# 3. Heap dump analysis (if configured)
kubectl exec <pod-name> -n production -- python -c "
import tracemalloc
import gc
gc.collect()
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"

# 4. Check for large objects
kubectl exec <pod-name> -n production -- python scripts/memory_profiler.py
```

### Common Causes & Solutions

#### A. Memory Leak in Code
```python
# Identify leaking objects
# Add to application
import weakref
import gc

class LeakDetector:
    def __init__(self):
        self.objects = weakref.WeakSet()
    
    def track(self, obj):
        self.objects.add(obj)
    
    def report(self):
        gc.collect()
        print(f"Tracked objects: {len(self.objects)}")
        
# Solution: Fix leak and redeploy
# Common patterns to fix:
# - Close files/connections properly
# - Clear caches periodically
# - Use weak references for caches
```

#### B. Cache Growing Unbounded
```bash
# Check Redis memory
kubectl exec <pod-name> -n production -- redis-cli INFO memory

# Solution: Set TTL on cache entries
kubectl exec <pod-name> -n production -- redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Clear specific cache pattern
kubectl exec <pod-name> -n production -- redis-cli --scan --pattern "cache:*" | xargs redis-cli DEL
```

#### C. Large Model in Memory
```python
# Check model memory usage
kubectl exec <pod-name> -n production -- python -c "
import torch
import psutil
import os

process = psutil.Process(os.getpid())
print(f'Memory before: {process.memory_info().rss / 1024 / 1024:.2f} MB')

# Force garbage collection
import gc
gc.collect()
torch.cuda.empty_cache()

print(f'Memory after: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Solution: Use model quantization or offloading
```

### Recovery Steps
1. **Immediate**: Restart affected pods
2. **Temporary**: Increase memory limits
3. **Permanent**: Fix memory leak and optimize usage
4. **Prevent**: Add memory monitoring alerts

---

## 5. Database Problems

### Symptoms
- Connection refused errors
- Slow queries
- Lock timeouts
- Replication lag

### Diagnostic Steps

```bash
# 1. Check connectivity
kubectl exec <pod-name> -n production -- pg_isready -h postgres.musicgen.ai

# 2. Check connection pool
kubectl exec <pod-name> -n production -- python -c "
import psycopg2
conn = psycopg2.connect('postgresql://...')
cur = conn.cursor()
cur.execute('SELECT count(*) FROM pg_stat_activity')
print(f'Active connections: {cur.fetchone()[0]}')
"

# 3. Check replication status
kubectl exec postgres-primary -- psql -U musicgen -c "SELECT * FROM pg_stat_replication;"

# 4. Check for locks
kubectl exec postgres-primary -- psql -U musicgen -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_query,
       blocking_activity.query AS blocking_query
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
"
```

### Common Causes & Solutions

#### A. Connection Pool Exhausted
```python
# Check pool settings
kubectl exec <pod-name> -n production -- python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://...')
print(f'Pool size: {engine.pool.size()}')
print(f'Overflow: {engine.pool._overflow}')
"

# Solution: Increase pool size
# In application config:
SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_POOL_OVERFLOW = 40
SQLALCHEMY_POOL_TIMEOUT = 30

# Or restart with larger pool
kubectl set env deployment/musicgen-api -n production DB_POOL_SIZE=50
kubectl rollout restart deployment/musicgen-api -n production
```

#### B. Long-Running Queries
```sql
-- Find and kill long queries
SELECT 
    pid,
    now() - query_start AS duration,
    query
FROM pg_stat_activity
WHERE state = 'active'
AND now() - query_start > interval '5 minutes'
ORDER BY duration DESC;

-- Kill specific query
SELECT pg_terminate_backend(12345);

-- Kill all long queries
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active'
AND now() - query_start > interval '5 minutes';
```

#### C. Replication Lag
```bash
# Check lag on replica
kubectl exec postgres-replica-1 -- psql -U musicgen -c "
SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int as lag_seconds;
"

# Solution: Pause batch jobs
kubectl scale deployment musicgen-batch-worker --replicas=0

# Increase wal_keep_size if needed
kubectl exec postgres-primary -- psql -U musicgen -c "ALTER SYSTEM SET wal_keep_size = '2GB';"
kubectl exec postgres-primary -- psql -U musicgen -c "SELECT pg_reload_conf();"
```

### Recovery Steps
1. **Identify blockers**: Kill blocking queries
2. **Free connections**: Restart app pods if needed
3. **Scale reads**: Route read traffic to replicas
4. **Optimize**: Add indexes, tune queries

---

## 6. GPU Issues

### Symptoms
- CUDA out of memory errors
- GPU not detected
- Training/inference hanging
- GPU utilization at 0%

### Diagnostic Steps

```bash
# 1. Check GPU availability
kubectl exec <gpu-pod> -n production -- nvidia-smi

# 2. Check CUDA errors in logs
kubectl logs <gpu-pod> -n production | grep -i "cuda\|gpu"

# 3. Monitor GPU memory
watch -n 1 'kubectl exec <gpu-pod> -n production -- nvidia-smi --query-gpu=memory.used,memory.free --format=csv'

# 4. Check GPU allocation
kubectl describe node <gpu-node> | grep -A5 "Allocated resources"
```

### Common Causes & Solutions

#### A. CUDA Out of Memory
```python
# Clear GPU memory
kubectl exec <gpu-pod> -n production -- python -c "
import torch
torch.cuda.empty_cache()
print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
"

# Solution: Reduce batch size
kubectl set env deployment/musicgen-worker -n production BATCH_SIZE=8
kubectl rollout restart deployment/musicgen-worker -n production

# Or enable gradient checkpointing
kubectl set env deployment/musicgen-worker -n production GRADIENT_CHECKPOINTING=true
```

#### B. GPU Not Detected
```bash
# Check node has GPU
kubectl get nodes -l nvidia.com/gpu=true

# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Solution: Restart device plugin
kubectl delete pod -n kube-system -l name=nvidia-device-plugin-ds

# Verify GPU runtime
kubectl exec <gpu-pod> -n production -- python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### C. GPU Memory Fragmentation
```python
# Check fragmentation
kubectl exec <gpu-pod> -n production -- python -c "
import torch
import numpy as np

# Try allocating large tensor
try:
    large_tensor = torch.cuda.FloatTensor(1000, 1000, 1000)
    print('Large allocation successful')
    del large_tensor
except RuntimeError as e:
    print(f'Large allocation failed: {e}')
    
# Check memory stats
print(torch.cuda.memory_summary())
"

# Solution: Restart pod to defragment
kubectl delete pod <gpu-pod> -n production
```

### Recovery Steps
1. **Clear memory**: Empty CUDA cache
2. **Reduce load**: Lower batch size or model size
3. **Restart pod**: Force fresh GPU allocation
4. **Scale out**: Add more GPU nodes if needed

---

## 7. Deployment Stuck

### Symptoms
- Deployment not progressing
- Pods stuck in Pending/ContainerCreating
- Rollout timeout

### Diagnostic Steps

```bash
# 1. Check rollout status
kubectl rollout status deployment/musicgen-api -n production

# 2. Check deployment conditions
kubectl describe deployment musicgen-api -n production | grep -A10 Conditions

# 3. Check pod events
kubectl describe pod <stuck-pod> -n production | tail -20

# 4. Check resource availability
kubectl describe nodes | grep -A5 "Allocated resources"
```

### Common Causes & Solutions

#### A. Insufficient Resources
```bash
# Check resource requests vs available
kubectl describe nodes | grep -E "Allocatable|Allocated" -A5

# Solution: Scale down other deployments or add nodes
kubectl scale deployment non-critical-app --replicas=1

# Or add node
eksctl scale nodegroup --cluster=musicgen-prod --nodes=4 --name=workers
```

#### B. Image Pull Errors
```bash
# Check for image pull errors
kubectl describe pod <pod-name> -n production | grep -i "pull"

# Solution: Fix image reference or credentials
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=user \
  --docker-password=pass

kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "regcred"}]}'
```

#### C. Failing Init Containers
```bash
# Check init container logs
kubectl logs <pod-name> -c <init-container> -n production

# Common issue: Database not ready
# Solution: Increase init container timeout
kubectl patch deployment musicgen-api -n production --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/initContainers/0/env/0/value",
    "value": "60"
  }
]'
```

### Recovery Steps
1. **Force progress**: Delete stuck pods
   ```bash
   kubectl delete pod <stuck-pod> -n production --force --grace-period=0
   ```

2. **Rollback if needed**: 
   ```bash
   kubectl rollout undo deployment/musicgen-api -n production
   ```

3. **Clean up and retry**:
   ```bash
   kubectl rollout restart deployment/musicgen-api -n production
   ```

---

## Emergency Contacts

### Escalation Path

1. **Level 1**: On-call Engineer
   - Phone: +1-xxx-xxx-xxxx
   - Slack: #oncall-alerts

2. **Level 2**: Platform Team Lead
   - Phone: +1-xxx-xxx-xxxx
   - Email: platform-lead@musicgen.ai

3. **Level 3**: VP of Engineering
   - Phone: +1-xxx-xxx-xxxx
   - Email: vp-eng@musicgen.ai

### External Support

- **AWS Support**: 1-800-xxx-xxxx (Enterprise Support)
- **Database Vendor**: support@database-vendor.com
- **CDN Support**: support@cdn-provider.com

---

## Quick Commands Reference

```bash
# Get all pods with issues
kubectl get pods -A | grep -v Running | grep -v Completed

# Force restart all API pods
kubectl rollout restart deployment/musicgen-api -n production

# Emergency scale down
kubectl scale deployment --all --replicas=1 -n production

# Clear all Redis cache
kubectl exec -it redis-master-0 -- redis-cli FLUSHALL

# Emergency database backup
kubectl exec postgres-primary -- pg_dump -Fc musicgen_prod > emergency_backup.dump

# Check all service endpoints
kubectl get endpoints -n production

# View recent errors across all pods
kubectl logs -n production -l app=musicgen --tail=100 | grep ERROR

# Get resource usage summary
kubectl top pods -n production --sort-by=memory
```

---

**Remember**: When in doubt, preserve data integrity over availability. It's better to have a brief outage than data corruption.