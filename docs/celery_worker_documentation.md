# Celery Worker System Documentation

This guide explains the Celery-based background worker system for Music Gen AI, including configuration, deployment, and monitoring.

## Overview

The Celery worker system provides:
- **Distributed task processing** across multiple workers
- **Priority-based task routing** with dedicated queues
- **Automatic retry logic** with exponential backoff
- **Dead letter queue** for failed tasks
- **Worker health monitoring** and auto-restart
- **Horizontal scaling** based on queue length
- **GPU support** for music generation tasks

## Architecture

### Components

1. **Redis** - Message broker and result backend
2. **Celery Beat** - Periodic task scheduler
3. **Worker Types**:
   - Critical Priority Workers
   - High Priority Generation Workers
   - Normal Generation Workers
   - Audio Processing Workers
   - Batch Processing Workers
   - Monitoring Workers
4. **Flower** - Web-based monitoring interface
5. **Worker Manager** - Autoscaling and health checks

### Queue Structure

```
Priority Queues:
├── critical (priority: 10)
├── generation-high (priority: 7)
├── generation (priority: 5)
├── processing (priority: 3)
├── batch (priority: 1)
├── monitoring (priority: 0)
└── dead-letter (failed tasks)
```

## Quick Start

### Local Development

1. Start Redis:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

2. Start a worker:
```bash
celery -A music_gen.workers.celery_app worker --loglevel=info
```

3. Start Celery Beat (for periodic tasks):
```bash
celery -A music_gen.workers.celery_app beat --loglevel=info
```

4. Start Flower (monitoring):
```bash
celery -A music_gen.workers.celery_app flower
```

### Docker Deployment

```bash
# Start all services
docker-compose -f docker-compose.workers.yml up -d

# Scale workers
docker-compose -f docker-compose.workers.yml up -d --scale worker-generation=5

# View logs
docker-compose -f docker-compose.workers.yml logs -f worker-generation
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/workers-deployment.yaml

# Scale workers
kubectl scale deployment worker-generation --replicas=10 -n musicgen-workers

# View worker pods
kubectl get pods -n musicgen-workers -l worker-type=generation
```

## Configuration

### Environment Variables

```bash
# Redis connection
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Worker configuration
CELERY_WORKER_CONCURRENCY=2
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_TASK_SOFT_TIME_LIMIT=3300
CELERY_TASK_TIME_LIMIT=3600

# Enable Celery workers in API
USE_CELERY_WORKERS=true

# Model cache directory
MODEL_CACHE_DIR=/app/models
```

### Worker Types Configuration

#### Critical Priority Worker
- Queues: `critical`
- Concurrency: 2
- Pool: prefork
- GPU: Required
- Use case: Urgent user requests

#### Generation Workers
- Queues: `generation-high`, `generation`
- Concurrency: 2-4 (autoscale)
- Pool: prefork
- GPU: Required
- Use case: Music generation tasks

#### Processing Workers
- Queues: `processing`
- Concurrency: 4
- Pool: prefork
- GPU: Not required
- Use case: Audio post-processing

#### Monitoring Workers
- Queues: `monitoring`, `dead-letter`
- Concurrency: 1
- Pool: eventlet
- GPU: Not required
- Use case: System monitoring, failed task handling

## Task Management

### Creating Tasks

```python
from music_gen.workers import generate_music_task

# High priority task
result = generate_music_task.apply_async(
    args=[task_id, request_data],
    priority=10,
    queue='generation-high'
)

# Normal priority with delay
result = generate_music_task.apply_async(
    args=[task_id, request_data],
    countdown=60  # Start after 60 seconds
)

# With retry configuration
result = generate_music_task.apply_async(
    args=[task_id, request_data],
    retry=True,
    retry_policy={
        'max_retries': 5,
        'interval_start': 60,
        'interval_step': 120,
        'interval_max': 3600,
    }
)
```

### Monitoring Tasks

```python
# Check task status
result = celery_app.AsyncResult(task_id)
print(result.status)  # PENDING, STARTED, SUCCESS, FAILURE

# Get task info
info = result.info  # Custom task progress/metadata

# Wait for completion
result.get(timeout=300)  # Wait up to 5 minutes

# Revoke task
celery_app.control.revoke(task_id, terminate=True)
```

### Batch Processing

```python
from celery import group

# Create batch of tasks
job = group(
    generate_music_task.s(f"task_{i}", data)
    for i, data in enumerate(batch_data)
)

# Execute batch
result = job.apply_async()

# Check batch progress
completed = sum(1 for r in result.results if r.ready())
```

## Monitoring

### Flower Web Interface

Access at `http://localhost:5555`

Features:
- Real-time worker status
- Task history and details
- Queue lengths
- Worker resource usage
- Task execution graphs

### Command Line Monitoring

```bash
# Active tasks
celery -A music_gen.workers.celery_app inspect active

# Worker stats
celery -A music_gen.workers.celery_app inspect stats

# Queue lengths
celery -A music_gen.workers.celery_app inspect reserved

# Registered tasks
celery -A music_gen.workers.celery_app inspect registered
```

### API Monitoring Endpoints

```bash
# Queue status
GET /api/v1/generate/queue/status

# Task monitoring
GET /api/v1/monitoring/tasks/health
GET /api/v1/monitoring/tasks/metrics
GET /api/v1/monitoring/tasks/distribution
GET /api/v1/monitoring/tasks/performance
```

## Scaling

### Manual Scaling

```bash
# Docker Compose
docker-compose -f docker-compose.workers.yml up -d --scale worker-generation=10

# Kubernetes
kubectl scale deployment worker-generation --replicas=10 -n musicgen-workers

# Worker Manager
python -m music_gen.workers.worker_manager --scale generation 10
```

### Auto-scaling Rules

The system automatically scales based on:

1. **Queue Length**:
   - < 20 tasks: 2 workers
   - 20-50 tasks: 3 workers
   - 50-100 tasks: 5 workers
   - > 100 tasks: 10 workers (max)

2. **CPU/Memory Usage**:
   - Scale up at 70% CPU
   - Scale up at 80% memory
   - Scale down at 30% usage

3. **Task Processing Time**:
   - Scale up if avg time > 60s
   - Scale down if avg time < 20s

### Kubernetes HPA Configuration

```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
- type: Pods
  pods:
    metric:
      name: celery_queue_length
    target:
      type: AverageValue
      averageValue: "30"
```

## Error Handling

### Retry Logic

Tasks automatically retry with exponential backoff:
- 1st retry: 60 seconds
- 2nd retry: 180 seconds
- 3rd retry: 540 seconds

### Dead Letter Queue

Failed tasks after max retries go to dead letter queue:
- Stored for 7 days
- Can be manually reprocessed
- Notifications sent for critical failures

### Common Issues

#### Worker Memory Leaks
```bash
# Restart workers after N tasks
CELERY_WORKER_MAX_TASKS_PER_CHILD=50

# Monitor memory usage
watch 'docker stats --no-stream | grep worker'
```

#### Stuck Tasks
```bash
# Find stalled tasks
GET /api/v1/monitoring/tasks/stalled

# Manual cleanup
POST /api/v1/monitoring/tasks/cleanup
```

#### Redis Connection Issues
```bash
# Check Redis connectivity
redis-cli -h redis-host ping

# Monitor Redis memory
redis-cli info memory
```

## Performance Tuning

### Redis Optimization

```conf
# redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
tcp-keepalive 60
timeout 300
```

### Worker Optimization

```python
# Prefetch optimization for long tasks
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# Connection pooling
CELERY_REDIS_MAX_CONNECTIONS = 20

# Task compression
CELERY_MESSAGE_COMPRESSION = 'gzip'
```

### GPU Utilization

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Set GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Limit GPU memory
export CUDA_VISIBLE_DEVICES=0,1
```

## Security

### Redis Security

```bash
# Enable Redis AUTH
requirepass your-strong-password

# Enable SSL/TLS
tls-port 6380
tls-cert-file /path/to/cert.pem
tls-key-file /path/to/key.pem
```

### Worker Security

```python
# Task serialization security
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'

# Disable pickle for security
CELERY_ENABLE_PICKLE = False
```

### Network Security

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: worker-network-policy
spec:
  podSelector:
    matchLabels:
      app: worker
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api
  - ports:
    - port: 6379
      protocol: TCP
```

## Troubleshooting

### Debug Mode

```bash
# Verbose logging
celery -A music_gen.workers.celery_app worker --loglevel=debug

# Single threaded for debugging
celery -A music_gen.workers.celery_app worker --concurrency=1 --pool=solo
```

### Common Errors

1. **Import errors**: Ensure PYTHONPATH includes project root
2. **GPU errors**: Check CUDA installation and drivers
3. **Memory errors**: Reduce concurrency or batch size
4. **Network timeouts**: Increase Redis timeout settings

### Health Checks

```python
# Worker health check task
@celery_app.task
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'worker': current_task.request.hostname,
    }

# Run health check
result = health_check.delay()
print(result.get(timeout=10))
```

## Best Practices

1. **Task Design**:
   - Keep tasks idempotent
   - Use task IDs for deduplication
   - Store results externally (not in Celery)
   - Handle partial failures gracefully

2. **Resource Management**:
   - Set appropriate time limits
   - Use memory limits for workers
   - Monitor GPU memory usage
   - Clean up temporary files

3. **Monitoring**:
   - Set up alerts for queue backlogs
   - Monitor worker health regularly
   - Track task failure rates
   - Log task execution times

4. **Deployment**:
   - Use separate workers for different task types
   - Deploy workers close to Redis
   - Use connection pooling
   - Enable task compression for large payloads