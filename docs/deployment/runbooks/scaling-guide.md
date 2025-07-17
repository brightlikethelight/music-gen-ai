# Scaling Guide and Runbook

This guide provides comprehensive procedures for scaling the Music Gen AI system to handle increased load, growth, and performance requirements.

## Table of Contents

1. [Scaling Overview](#scaling-overview)
2. [Horizontal Scaling](#horizontal-scaling)
3. [Vertical Scaling](#vertical-scaling)
4. [Database Scaling](#database-scaling)
5. [Auto-Scaling Configuration](#auto-scaling-configuration)
6. [Performance Monitoring](#performance-monitoring)
7. [Capacity Planning](#capacity-planning)
8. [Emergency Scaling](#emergency-scaling)

---

## Scaling Overview

### Scaling Triggers

| Metric | Scale Up Threshold | Scale Down Threshold | Action |
|--------|-------------------|---------------------|--------|
| CPU Utilization | >70% for 5 minutes | <30% for 15 minutes | Add/Remove instances |
| Memory Usage | >80% for 5 minutes | <40% for 15 minutes | Vertical scale or add instances |
| Request Rate | >1000 RPS | <100 RPS | Scale API tier |
| Queue Length | >100 tasks | <10 tasks | Scale workers |
| Response Time | P95 >2s | P95 <500ms | Add instances or optimize |
| Error Rate | >5% | <0.1% | Immediate scaling |

### Service Tiers

```yaml
scaling_tiers:
  api_gateway:
    min_replicas: 3
    max_replicas: 50
    target_cpu: 70
    target_memory: 80
  
  application_server:
    min_replicas: 5
    max_replicas: 100
    target_cpu: 70
    target_memory: 80
  
  generation_workers:
    min_replicas: 2
    max_replicas: 20
    target_gpu: 80
    queue_threshold: 50
  
  database:
    connection_pool_size: 100
    read_replicas: 3
    max_read_replicas: 10
```

---

## Horizontal Scaling

### API Tier Scaling

#### Manual Scaling
```bash
#!/bin/bash
# Manual horizontal scaling for API tier

# 1. Check current capacity
kubectl get deployment musicgen-api -n production -o yaml | grep replicas

# 2. Scale up API servers
kubectl scale deployment musicgen-api -n production --replicas=10

# 3. Verify scaling
kubectl get pods -n production -l app=musicgen-api
kubectl wait --for=condition=ready pod -l app=musicgen-api -n production --timeout=300s

# 4. Monitor during scale-up
watch -n 5 'kubectl get pods -n production -l app=musicgen-api | grep -E "Running|Pending"'

# 5. Update load balancer if needed
kubectl patch service musicgen-api -n production -p '{"spec":{"type":"LoadBalancer"}}'
```

#### Automated Scaling Configuration
```yaml
# hpa-api-advanced.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: musicgen-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: musicgen-api
  minReplicas: 5
  maxReplicas: 50
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # Custom metrics scaling
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
  
  # External metrics (queue length)
  - type: External
    external:
      metric:
        name: sqs_messages_visible
        selector:
          matchLabels:
            queue: generation-tasks
      target:
        type: Value
        value: "100"

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 10
        periodSeconds: 60
      selectPolicy: Max
    
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min
```

### Worker Scaling

#### GPU Worker Scaling
```python
#!/usr/bin/env python3
"""
GPU worker scaling based on queue depth and GPU utilization
"""
import kubernetes
import boto3
import psutil
import time
from datetime import datetime

class GPUWorkerScaler:
    def __init__(self):
        kubernetes.config.load_incluster_config()
        self.k8s_apps = kubernetes.client.AppsV1Api()
        self.k8s_core = kubernetes.client.CoreV1Api()
        self.namespace = "production"
        
    def get_queue_depth(self):
        """Get current generation queue depth"""
        import redis
        r = redis.Redis(host='redis.musicgen.ai', port=6379)
        return r.llen('generation_queue')
    
    def get_gpu_utilization(self):
        """Get average GPU utilization across workers"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            total_util = 0
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                total_util += util.gpu
            
            return total_util / device_count if device_count > 0 else 0
        except Exception as e:
            print(f"Error getting GPU utilization: {e}")
            return 0
    
    def get_current_worker_count(self):
        """Get current number of GPU worker pods"""
        deployment = self.k8s_apps.read_namespaced_deployment(
            name="musicgen-gpu-worker",
            namespace=self.namespace
        )
        return deployment.spec.replicas
    
    def scale_workers(self, target_replicas):
        """Scale GPU workers to target replica count"""
        # Update deployment
        body = {'spec': {'replicas': target_replicas}}
        
        self.k8s_apps.patch_namespaced_deployment_scale(
            name="musicgen-gpu-worker",
            namespace=self.namespace,
            body=body
        )
        
        print(f"Scaled GPU workers to {target_replicas} replicas")
        
        # Log scaling event
        self.log_scaling_event(target_replicas)
    
    def calculate_optimal_replicas(self):
        """Calculate optimal number of worker replicas"""
        queue_depth = self.get_queue_depth()
        gpu_util = self.get_gpu_utilization()
        current_replicas = self.get_current_worker_count()
        
        # Scaling algorithm
        if queue_depth > 50 and gpu_util > 80:
            # High queue + high utilization = scale up aggressively
            target_replicas = min(current_replicas * 2, 20)
        elif queue_depth > 100:
            # Very high queue = scale up regardless of utilization
            target_replicas = min(current_replicas + 5, 20)
        elif queue_depth < 5 and gpu_util < 30:
            # Low queue + low utilization = scale down
            target_replicas = max(current_replicas // 2, 2)
        elif queue_depth < 1:
            # No queue = minimum workers
            target_replicas = 2
        else:
            # Maintain current level
            target_replicas = current_replicas
        
        return target_replicas
    
    def auto_scale(self):
        """Execute auto-scaling logic"""
        current_replicas = self.get_current_worker_count()
        target_replicas = self.calculate_optimal_replicas()
        
        if target_replicas != current_replicas:
            print(f"Scaling from {current_replicas} to {target_replicas} workers")
            self.scale_workers(target_replicas)
        else:
            print(f"No scaling needed. Current: {current_replicas} workers")
    
    def log_scaling_event(self, new_replica_count):
        """Log scaling event for analysis"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'scale',
            'service': 'gpu-workers',
            'new_replica_count': new_replica_count,
            'queue_depth': self.get_queue_depth(),
            'gpu_utilization': self.get_gpu_utilization()
        }
        
        # Send to logging service
        import json
        print(json.dumps(event))

# Usage as a continuous scaler
if __name__ == "__main__":
    scaler = GPUWorkerScaler()
    
    while True:
        try:
            scaler.auto_scale()
        except Exception as e:
            print(f"Error in auto-scaling: {e}")
        
        time.sleep(60)  # Check every minute
```

### Load Balancer Scaling

#### AWS ALB Configuration
```yaml
# alb-scaling-config.yaml
apiVersion: v1
kind: Service
metadata:
  name: musicgen-api-alb
  namespace: production
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-interval: "10"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-timeout: "5"
    service.beta.kubernetes.io/aws-load-balancer-healthy-threshold: "2"
    service.beta.kubernetes.io/aws-load-balancer-unhealthy-threshold: "3"
spec:
  type: LoadBalancer
  selector:
    app: musicgen-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  - port: 443
    targetPort: 8000
    protocol: TCP
```

---

## Vertical Scaling

### Instance Type Scaling

#### AWS Instance Upgrade Script
```bash
#!/bin/bash
# Vertical scaling for AWS instances

# Configuration
INSTANCE_ID="i-1234567890abcdef0"
CURRENT_TYPE="c6i.2xlarge"
TARGET_TYPE="c6i.4xlarge"
REGION="us-east-1"

echo "=== Vertical Scaling Instance $INSTANCE_ID ==="
echo "From: $CURRENT_TYPE"
echo "To: $TARGET_TYPE"

# 1. Create AMI snapshot
echo "Creating AMI snapshot..."
AMI_ID=$(aws ec2 create-image \
    --instance-id $INSTANCE_ID \
    --name "scaling-snapshot-$(date +%Y%m%d-%H%M%S)" \
    --description "Snapshot before vertical scaling" \
    --region $REGION \
    --query 'ImageId' --output text)

echo "AMI snapshot created: $AMI_ID"

# 2. Wait for AMI to be available
echo "Waiting for AMI to be available..."
aws ec2 wait image-available --image-ids $AMI_ID --region $REGION

# 3. Stop instance
echo "Stopping instance..."
aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION
aws ec2 wait instance-stopped --instance-ids $INSTANCE_ID --region $REGION

# 4. Modify instance type
echo "Changing instance type to $TARGET_TYPE..."
aws ec2 modify-instance-attribute \
    --instance-id $INSTANCE_ID \
    --instance-type $TARGET_TYPE \
    --region $REGION

# 5. Start instance
echo "Starting instance..."
aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# 6. Verify new instance type
NEW_TYPE=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].InstanceType' \
    --output text)

echo "✅ Instance type changed to: $NEW_TYPE"

# 7. Health check
echo "Performing health check..."
sleep 30  # Wait for services to start

INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' \
    --output text)

if curl -sf http://$INSTANCE_IP:8000/health; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed - manual intervention required"
fi
```

### Container Resource Scaling

#### Kubernetes Resource Updates
```python
#!/usr/bin/env python3
"""
Container resource scaling for Kubernetes deployments
"""
import kubernetes
import json
from datetime import datetime

class ContainerResourceScaler:
    def __init__(self, namespace="production"):
        kubernetes.config.load_incluster_config()
        self.k8s_apps = kubernetes.client.AppsV1Api()
        self.namespace = namespace
    
    def get_current_resources(self, deployment_name):
        """Get current resource limits and requests"""
        deployment = self.k8s_apps.read_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace
        )
        
        container = deployment.spec.template.spec.containers[0]
        
        return {
            'requests': {
                'cpu': container.resources.requests.get('cpu', ''),
                'memory': container.resources.requests.get('memory', '')
            },
            'limits': {
                'cpu': container.resources.limits.get('cpu', ''),
                'memory': container.resources.limits.get('memory', '')
            }
        }
    
    def scale_resources(self, deployment_name, cpu_limit=None, memory_limit=None, 
                       cpu_request=None, memory_request=None):
        """Scale container resources"""
        
        # Prepare patch
        patch = {
            'spec': {
                'template': {
                    'spec': {
                        'containers': [{
                            'name': deployment_name.replace('-', ''),
                            'resources': {}
                        }]
                    }
                }
            }
        }
        
        resources = patch['spec']['template']['spec']['containers'][0]['resources']
        
        if cpu_limit or memory_limit:
            resources['limits'] = {}
            if cpu_limit:
                resources['limits']['cpu'] = cpu_limit
            if memory_limit:
                resources['limits']['memory'] = memory_limit
        
        if cpu_request or memory_request:
            resources['requests'] = {}
            if cpu_request:
                resources['requests']['cpu'] = cpu_request
            if memory_request:
                resources['requests']['memory'] = memory_request
        
        # Apply patch
        self.k8s_apps.patch_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace,
            body=patch
        )
        
        print(f"✅ Scaled resources for {deployment_name}")
        
        # Log scaling event
        self.log_resource_scaling(deployment_name, patch)
    
    def auto_scale_based_on_usage(self, deployment_name):
        """Auto-scale resources based on historical usage"""
        
        # Get metrics (this would typically come from Prometheus)
        metrics = self.get_resource_metrics(deployment_name)
        
        current_resources = self.get_current_resources(deployment_name)
        
        # Calculate new resources based on usage
        new_resources = self.calculate_optimal_resources(metrics, current_resources)
        
        if new_resources != current_resources:
            self.scale_resources(
                deployment_name,
                cpu_limit=new_resources['limits']['cpu'],
                memory_limit=new_resources['limits']['memory'],
                cpu_request=new_resources['requests']['cpu'],
                memory_request=new_resources['requests']['memory']
            )
    
    def calculate_optimal_resources(self, metrics, current_resources):
        """Calculate optimal resource allocation"""
        # Get 95th percentile usage
        cpu_p95 = metrics.get('cpu_p95', 0)
        memory_p95 = metrics.get('memory_p95', 0)
        
        # Add 20% buffer for CPU, 30% for memory
        optimal_cpu = cpu_p95 * 1.2
        optimal_memory = memory_p95 * 1.3
        
        # Convert to Kubernetes resource format
        new_resources = {
            'requests': {
                'cpu': f"{optimal_cpu:.1f}",
                'memory': f"{int(optimal_memory)}Mi"
            },
            'limits': {
                'cpu': f"{optimal_cpu * 1.5:.1f}",  # 50% above request
                'memory': f"{int(optimal_memory * 1.2)}Mi"  # 20% above request
            }
        }
        
        return new_resources
    
    def get_resource_metrics(self, deployment_name):
        """Get resource usage metrics (mock implementation)"""
        # In reality, this would query Prometheus or metrics API
        return {
            'cpu_p95': 2.5,  # 2.5 CPU cores
            'memory_p95': 4096,  # 4GB
            'cpu_avg': 1.8,
            'memory_avg': 3072
        }
    
    def log_resource_scaling(self, deployment_name, patch):
        """Log resource scaling event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'resource_scale',
            'deployment': deployment_name,
            'namespace': self.namespace,
            'patch': patch
        }
        
        print(json.dumps(event))

# Usage
if __name__ == "__main__":
    scaler = ContainerResourceScaler()
    
    # Example: Scale API deployment
    scaler.scale_resources(
        'musicgen-api',
        cpu_limit='4',
        memory_limit='8Gi',
        cpu_request='2',
        memory_request='4Gi'
    )
```

---

## Database Scaling

### Read Replica Management

#### Automated Read Replica Scaling
```python
#!/usr/bin/env python3
"""
Automated database read replica scaling
"""
import boto3
import psycopg2
from datetime import datetime, timedelta

class DatabaseScaler:
    def __init__(self, region='us-east-1'):
        self.rds = boto3.client('rds', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
    def get_read_replica_count(self, primary_instance):
        """Get current number of read replicas"""
        response = self.rds.describe_db_instances(
            DBInstanceIdentifier=primary_instance
        )
        
        primary_info = response['DBInstances'][0]
        return len(primary_info.get('ReadReplicaDBInstanceIdentifiers', []))
    
    def get_database_metrics(self, instance_id, hours=1):
        """Get database performance metrics"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = {}
        
        # CPU Utilization
        cpu_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average', 'Maximum']
        )
        
        if cpu_response['Datapoints']:
            metrics['cpu_avg'] = sum(d['Average'] for d in cpu_response['Datapoints']) / len(cpu_response['Datapoints'])
            metrics['cpu_max'] = max(d['Maximum'] for d in cpu_response['Datapoints'])
        
        # Connection Count
        conn_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName='DatabaseConnections',
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average', 'Maximum']
        )
        
        if conn_response['Datapoints']:
            metrics['connections_avg'] = sum(d['Average'] for d in conn_response['Datapoints']) / len(conn_response['Datapoints'])
            metrics['connections_max'] = max(d['Maximum'] for d in conn_response['Datapoints'])
        
        return metrics
    
    def create_read_replica(self, primary_instance, replica_suffix=None):
        """Create a new read replica"""
        if not replica_suffix:
            replica_suffix = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        
        replica_identifier = f"{primary_instance}-replica-{replica_suffix}"
        
        try:
            response = self.rds.create_db_instance_read_replica(
                DBInstanceIdentifier=replica_identifier,
                SourceDBInstanceIdentifier=primary_instance,
                DBInstanceClass='db.r6i.2xlarge',
                PubliclyAccessible=False,
                MultiAZ=False,
                StorageEncrypted=True,
                CopyTagsToSnapshot=True,
                Tags=[
                    {'Key': 'Purpose', 'Value': 'ReadReplica'},
                    {'Key': 'AutoScaled', 'Value': 'true'},
                    {'Key': 'CreatedAt', 'Value': datetime.utcnow().isoformat()}
                ]
            )
            
            print(f"✅ Creating read replica: {replica_identifier}")
            return replica_identifier
            
        except Exception as e:
            print(f"❌ Failed to create read replica: {e}")
            return None
    
    def delete_read_replica(self, replica_identifier):
        """Delete a read replica"""
        try:
            self.rds.delete_db_instance(
                DBInstanceIdentifier=replica_identifier,
                SkipFinalSnapshot=True
            )
            print(f"✅ Deleting read replica: {replica_identifier}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete read replica: {e}")
            return False
    
    def auto_scale_read_replicas(self, primary_instance, min_replicas=1, max_replicas=5):
        """Auto-scale read replicas based on load"""
        current_count = self.get_read_replica_count(primary_instance)
        metrics = self.get_database_metrics(primary_instance)
        
        # Scaling decision logic
        cpu_avg = metrics.get('cpu_avg', 0)
        connections_avg = metrics.get('connections_avg', 0)
        
        target_replicas = current_count
        
        # Scale up conditions
        if cpu_avg > 70 or connections_avg > 80:
            target_replicas = min(current_count + 1, max_replicas)
            print(f"High load detected (CPU: {cpu_avg:.1f}%, Connections: {connections_avg:.1f})")
        
        # Scale down conditions
        elif cpu_avg < 30 and connections_avg < 40 and current_count > min_replicas:
            target_replicas = max(current_count - 1, min_replicas)
            print(f"Low load detected (CPU: {cpu_avg:.1f}%, Connections: {connections_avg:.1f})")
        
        # Execute scaling
        if target_replicas > current_count:
            # Scale up
            for _ in range(target_replicas - current_count):
                self.create_read_replica(primary_instance)
        elif target_replicas < current_count:
            # Scale down - delete newest replica
            replicas = self.get_read_replica_list(primary_instance)
            if replicas:
                # Delete the most recently created replica
                newest_replica = sorted(replicas, key=lambda x: x['InstanceCreateTime'])[-1]
                self.delete_read_replica(newest_replica['DBInstanceIdentifier'])
    
    def get_read_replica_list(self, primary_instance):
        """Get list of read replicas for primary instance"""
        response = self.rds.describe_db_instances()
        
        replicas = []
        for db in response['DBInstances']:
            if (db.get('ReadReplicaSourceDBInstanceIdentifier') == primary_instance and
                db['DBInstanceStatus'] == 'available'):
                replicas.append(db)
        
        return replicas

# Usage
if __name__ == "__main__":
    scaler = DatabaseScaler()
    scaler.auto_scale_read_replicas('musicgen-primary', min_replicas=2, max_replicas=8)
```

### Connection Pool Scaling

```python
#!/usr/bin/env python3
"""
Database connection pool scaling
"""
import psycopg2
from psycopg2 import pool
import threading
import time

class ConnectionPoolScaler:
    def __init__(self, database_url, min_conn=5, max_conn=50):
        self.database_url = database_url
        self.min_conn = min_conn
        self.max_conn = max_conn
        self.current_pool = None
        self.lock = threading.Lock()
        self.metrics = {
            'active_connections': 0,
            'waiting_requests': 0,
            'failed_requests': 0
        }
        
    def create_pool(self, min_conn, max_conn):
        """Create a new connection pool"""
        return psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn, self.database_url
        )
    
    def scale_pool(self, target_size):
        """Scale connection pool to target size"""
        with self.lock:
            if self.current_pool:
                self.current_pool.closeall()
            
            self.current_pool = self.create_pool(
                min(self.min_conn, target_size),
                target_size
            )
            
            print(f"✅ Scaled connection pool to {target_size} connections")
    
    def get_connection_metrics(self):
        """Get current connection pool metrics"""
        with self.lock:
            if not self.current_pool:
                return self.metrics
                
            # This is a simplified example - real metrics would come from monitoring
            self.metrics.update({
                'pool_size': self.current_pool.maxconn,
                'available_connections': len(self.current_pool._pool),
                'busy_connections': self.current_pool.maxconn - len(self.current_pool._pool)
            })
            
            return self.metrics
    
    def auto_scale_pool(self):
        """Automatically scale pool based on usage"""
        metrics = self.get_connection_metrics()
        
        utilization = metrics.get('busy_connections', 0) / metrics.get('pool_size', 1)
        
        if utilization > 0.8:
            # Scale up
            new_size = min(metrics['pool_size'] * 2, self.max_conn)
            self.scale_pool(new_size)
        elif utilization < 0.3 and metrics['pool_size'] > self.min_conn:
            # Scale down
            new_size = max(metrics['pool_size'] // 2, self.min_conn)
            self.scale_pool(new_size)

# Usage
if __name__ == "__main__":
    pool_scaler = ConnectionPoolScaler(
        "postgresql://user:pass@host:5432/db",
        min_conn=10,
        max_conn=100
    )
    
    # Initialize pool
    pool_scaler.scale_pool(20)
    
    # Auto-scale periodically
    while True:
        pool_scaler.auto_scale_pool()
        time.sleep(60)  # Check every minute
```

---

## Auto-Scaling Configuration

### Kubernetes Vertical Pod Autoscaler (VPA)

```yaml
# vpa-musicgen-api.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: musicgen-api-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: musicgen-api
  updatePolicy:
    updateMode: "Auto"  # Can be "Auto", "Initial", or "Off"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      minAllowed:
        cpu: "500m"
        memory: "1Gi"
      maxAllowed:
        cpu: "8"
        memory: "16Gi"
      controlledResources:
      - cpu
      - memory
      controlledValues: RequestsAndLimits
```

### Custom Metrics Auto-Scaling

```python
#!/usr/bin/env python3
"""
Custom metrics-based auto-scaling
"""
import kubernetes
import requests
import time
from datetime import datetime

class CustomMetricsScaler:
    def __init__(self):
        kubernetes.config.load_incluster_config()
        self.k8s_apps = kubernetes.client.AppsV1Api()
        self.prometheus_url = "http://prometheus.musicgen.ai:9090"
        
    def get_prometheus_metric(self, query):
        """Query Prometheus for custom metrics"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query}
            )
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                return float(data['data']['result'][0]['value'][1])
            return 0
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return 0
    
    def get_scaling_metrics(self):
        """Get all metrics needed for scaling decisions"""
        metrics = {}
        
        # Queue depth
        metrics['queue_depth'] = self.get_prometheus_metric(
            'redis_list_length{key="generation_queue"}'
        )
        
        # Average response time
        metrics['avg_response_time'] = self.get_prometheus_metric(
            'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
        )
        
        # Error rate
        metrics['error_rate'] = self.get_prometheus_metric(
            'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])'
        )
        
        # GPU utilization
        metrics['gpu_utilization'] = self.get_prometheus_metric(
            'avg(nvidia_gpu_duty_cycle)'
        )
        
        # Business metrics
        metrics['generations_per_minute'] = self.get_prometheus_metric(
            'rate(generations_completed_total[1m]) * 60'
        )
        
        return metrics
    
    def calculate_scaling_decision(self, metrics):
        """Calculate scaling decision based on multiple metrics"""
        decisions = {}
        
        # API scaling
        if metrics['avg_response_time'] > 2.0 or metrics['error_rate'] > 0.05:
            decisions['api'] = 'scale_up'
        elif metrics['avg_response_time'] < 0.5 and metrics['error_rate'] < 0.01:
            decisions['api'] = 'scale_down'
        else:
            decisions['api'] = 'maintain'
        
        # Worker scaling
        if metrics['queue_depth'] > 50 or metrics['gpu_utilization'] > 80:
            decisions['workers'] = 'scale_up'
        elif metrics['queue_depth'] < 5 and metrics['gpu_utilization'] < 30:
            decisions['workers'] = 'scale_down'
        else:
            decisions['workers'] = 'maintain'
        
        # Emergency scaling
        if metrics['error_rate'] > 0.1 or metrics['avg_response_time'] > 5.0:
            decisions['emergency'] = True
            decisions['api'] = 'scale_up_aggressive'
        
        return decisions
    
    def execute_scaling_decisions(self, decisions):
        """Execute scaling decisions"""
        for service, action in decisions.items():
            if service == 'emergency':
                continue
                
            current_replicas = self.get_current_replicas(f"musicgen-{service}")
            target_replicas = current_replicas
            
            if action == 'scale_up':
                target_replicas = min(current_replicas + 2, 20)
            elif action == 'scale_up_aggressive':
                target_replicas = min(current_replicas * 2, 30)
            elif action == 'scale_down':
                target_replicas = max(current_replicas - 1, 2)
            
            if target_replicas != current_replicas:
                self.scale_deployment(f"musicgen-{service}", target_replicas)
    
    def get_current_replicas(self, deployment_name):
        """Get current replica count for deployment"""
        try:
            deployment = self.k8s_apps.read_namespaced_deployment(
                name=deployment_name,
                namespace="production"
            )
            return deployment.spec.replicas
        except Exception:
            return 3  # Default
    
    def scale_deployment(self, deployment_name, replicas):
        """Scale deployment to target replica count"""
        body = {'spec': {'replicas': replicas}}
        
        self.k8s_apps.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace="production",
            body=body
        )
        
        print(f"✅ Scaled {deployment_name} to {replicas} replicas")
    
    def run_scaling_loop(self):
        """Main scaling loop"""
        while True:
            try:
                metrics = self.get_scaling_metrics()
                decisions = self.calculate_scaling_decision(metrics)
                self.execute_scaling_decisions(decisions)
                
                # Log metrics and decisions
                print(f"[{datetime.utcnow()}] Metrics: {metrics}")
                print(f"[{datetime.utcnow()}] Decisions: {decisions}")
                
            except Exception as e:
                print(f"Error in scaling loop: {e}")
            
            time.sleep(60)  # Check every minute

# Usage
if __name__ == "__main__":
    scaler = CustomMetricsScaler()
    scaler.run_scaling_loop()
```

---

## Performance Monitoring

### Real-time Scaling Dashboard

```python
#!/usr/bin/env python3
"""
Real-time scaling dashboard with Grafana integration
"""
import requests
import json
from datetime import datetime, timedelta

class ScalingDashboard:
    def __init__(self, grafana_url, api_key):
        self.grafana_url = grafana_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def create_scaling_dashboard(self):
        """Create Grafana dashboard for scaling metrics"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Music Gen AI - Scaling Metrics",
                "tags": ["scaling", "performance"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "API Replicas",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "kube_deployment_spec_replicas{deployment=\"musicgen-api\"}",
                                "legendFormat": "Current Replicas"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "sum(rate(http_requests_total[5m]))",
                                "legendFormat": "Requests/sec"
                            }
                        ],
                        "gridPos": {"h": 6, "w": 12, "x": 6, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Response Time Percentiles",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "P50"
                            },
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "P95"
                            },
                            {
                                "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "P99"
                            }
                        ],
                        "gridPos": {"h": 6, "w": 12, "x": 0, "y": 6}
                    },
                    {
                        "id": 4,
                        "title": "GPU Worker Replicas",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "kube_deployment_spec_replicas{deployment=\"musicgen-gpu-worker\"}",
                                "legendFormat": "GPU Workers"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 6}
                    },
                    {
                        "id": 5,
                        "title": "Generation Queue Depth",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "redis_list_length{key=\"generation_queue\"}",
                                "legendFormat": "Queue Length"
                            }
                        ],
                        "gridPos": {"h": 6, "w": 12, "x": 0, "y": 12}
                    },
                    {
                        "id": 6,
                        "title": "Database Connections",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "sum(pg_stat_database_numbackends)",
                                "legendFormat": "Active Connections"
                            }
                        ],
                        "gridPos": {"h": 6, "w": 12, "x": 12, "y": 12}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "10s"
            }
        }
        
        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            json=dashboard
        )
        
        if response.status_code == 200:
            print("✅ Scaling dashboard created successfully")
            return response.json()
        else:
            print(f"❌ Failed to create dashboard: {response.text}")
            return None

# Usage
if __name__ == "__main__":
    dashboard = ScalingDashboard(
        "https://grafana.musicgen.ai",
        "your-grafana-api-key"
    )
    dashboard.create_scaling_dashboard()
```

---

## Capacity Planning

### Predictive Scaling

```python
#!/usr/bin/env python3
"""
Predictive scaling based on historical data and trends
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import requests

class PredictiveScaler:
    def __init__(self, prometheus_url):
        self.prometheus_url = prometheus_url
        self.models = {}
        
    def get_historical_metrics(self, metric_query, days=30):
        """Get historical metrics from Prometheus"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query_range",
            params={
                'query': metric_query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': '1h'  # 1 hour intervals
            }
        )
        
        data = response.json()
        if data['status'] == 'success' and data['data']['result']:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['timestamp', 'value'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['value'] = df['value'].astype(float)
            return df
        
        return pd.DataFrame()
    
    def train_prediction_model(self, metric_name, metric_query):
        """Train prediction model for a specific metric"""
        df = self.get_historical_metrics(metric_query)
        
        if df.empty:
            print(f"No data available for {metric_name}")
            return None
        
        # Feature engineering
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create lag features
        for lag in [1, 24, 168]:  # 1h, 1d, 1w
            df[f'value_lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling averages
        df['value_ma_24'] = df['value'].rolling(window=24).mean()
        df['value_ma_168'] = df['value'].rolling(window=168).mean()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 100:  # Need sufficient data
            print(f"Insufficient data for {metric_name}")
            return None
        
        # Prepare features and target
        feature_cols = ['hour', 'day_of_week', 'day_of_month', 'is_weekend',
                       'value_lag_1', 'value_lag_24', 'value_lag_168',
                       'value_ma_24', 'value_ma_168']
        
        X = df[feature_cols]
        y = df['value']
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Store model and scaler
        self.models[metric_name] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'last_data': df.tail(200)  # Keep recent data for predictions
        }
        
        print(f"✅ Trained model for {metric_name}")
        return model
    
    def predict_future_load(self, metric_name, hours_ahead=24):
        """Predict future load for a metric"""
        if metric_name not in self.models:
            print(f"No model available for {metric_name}")
            return None
        
        model_info = self.models[metric_name]
        model = model_info['model']
        scaler = model_info['scaler']
        feature_cols = model_info['feature_cols']
        last_data = model_info['last_data']
        
        predictions = []
        current_time = datetime.utcnow()
        
        for i in range(hours_ahead):
            future_time = current_time + timedelta(hours=i)
            
            # Create features for future time
            features = {
                'hour': future_time.hour,
                'day_of_week': future_time.weekday(),
                'day_of_month': future_time.day,
                'is_weekend': 1 if future_time.weekday() >= 5 else 0
            }
            
            # Add lag features (use recent actual data + predictions)
            if i == 0:
                features['value_lag_1'] = last_data['value'].iloc[-1]
            else:
                features['value_lag_1'] = predictions[i-1] if i > 0 else last_data['value'].iloc[-1]
            
            features['value_lag_24'] = last_data['value'].iloc[-24] if len(last_data) >= 24 else last_data['value'].mean()
            features['value_lag_168'] = last_data['value'].iloc[-168] if len(last_data) >= 168 else last_data['value'].mean()
            features['value_ma_24'] = last_data['value'].tail(24).mean()
            features['value_ma_168'] = last_data['value'].tail(168).mean() if len(last_data) >= 168 else last_data['value'].mean()
            
            # Make prediction
            X = np.array([[features[col] for col in feature_cols]])
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            
            predictions.append(max(0, prediction))  # Ensure non-negative
        
        return predictions
    
    def generate_scaling_recommendations(self):
        """Generate scaling recommendations based on predictions"""
        recommendations = {}
        
        # Train/update models
        self.train_prediction_model('request_rate', 'sum(rate(http_requests_total[5m]))')
        self.train_prediction_model('queue_depth', 'redis_list_length{key="generation_queue"}')
        self.train_prediction_model('cpu_usage', 'avg(rate(cpu_usage_seconds_total[5m]))')
        
        # Get predictions
        request_predictions = self.predict_future_load('request_rate', 24)
        queue_predictions = self.predict_future_load('queue_depth', 24)
        cpu_predictions = self.predict_future_load('cpu_usage', 24)
        
        if not all([request_predictions, queue_predictions, cpu_predictions]):
            return recommendations
        
        # Analyze predictions for scaling recommendations
        for hour in range(24):
            time_slot = f"{hour:02d}:00"
            
            # Predict required capacity
            request_rate = request_predictions[hour]
            queue_depth = queue_predictions[hour]
            cpu_usage = cpu_predictions[hour]
            
            # Calculate required replicas (simplified)
            # Assume each API pod can handle 50 RPS
            required_api_replicas = max(3, int(request_rate / 50) + 1)
            
            # GPU workers based on queue
            required_gpu_workers = max(2, int(queue_depth / 10) + 1)
            
            recommendations[time_slot] = {
                'api_replicas': required_api_replicas,
                'gpu_workers': required_gpu_workers,
                'predicted_metrics': {
                    'request_rate': request_rate,
                    'queue_depth': queue_depth,
                    'cpu_usage': cpu_usage
                }
            }
        
        return recommendations
    
    def create_scaling_schedule(self, recommendations):
        """Create a proactive scaling schedule"""
        schedule = []
        
        current_time = datetime.utcnow()
        
        for hour_offset, (time_slot, rec) in enumerate(recommendations.items()):
            schedule_time = current_time + timedelta(hours=hour_offset)
            
            schedule.append({
                'time': schedule_time,
                'actions': [
                    {
                        'service': 'musicgen-api',
                        'replicas': rec['api_replicas']
                    },
                    {
                        'service': 'musicgen-gpu-worker',
                        'replicas': rec['gpu_workers']
                    }
                ],
                'reason': f"Predicted load: {rec['predicted_metrics']['request_rate']:.1f} RPS"
            })
        
        return schedule

# Usage
if __name__ == "__main__":
    scaler = PredictiveScaler("http://prometheus.musicgen.ai:9090")
    
    # Generate recommendations
    recommendations = scaler.generate_scaling_recommendations()
    
    if recommendations:
        print("Predictive Scaling Recommendations (next 24 hours):")
        for time_slot, rec in recommendations.items():
            print(f"{time_slot}: API={rec['api_replicas']}, Workers={rec['gpu_workers']}")
        
        # Create schedule
        schedule = scaler.create_scaling_schedule(recommendations)
        print(f"\nGenerated {len(schedule)} scheduled scaling actions")
```

---

## Emergency Scaling

### Rapid Response Scaling

```bash
#!/bin/bash
# Emergency scaling for system overload

set -euo pipefail

echo "=== EMERGENCY SCALING ACTIVATED ==="
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Immediate API scaling
echo "Scaling API servers to maximum..."
kubectl scale deployment musicgen-api -n production --replicas=30
kubectl scale deployment musicgen-worker -n production --replicas=15

# 2. Database connection scaling
echo "Increasing database connections..."
kubectl exec postgres-primary -- psql -U musicgen -c "
    ALTER SYSTEM SET max_connections = '500';
    SELECT pg_reload_conf();
"

# 3. Add more nodes if on cloud
if command -v eksctl &> /dev/null; then
    echo "Adding emergency compute nodes..."
    eksctl scale nodegroup --cluster=musicgen-prod --nodes=20 --name=api-workers
    eksctl scale nodegroup --cluster=musicgen-prod --nodes=10 --name=gpu-workers
fi

# 4. Enable read-only mode for non-critical operations
echo "Enabling emergency mode..."
kubectl create configmap emergency-mode -n production --from-literal=enabled=true

# 5. Disable non-essential features
kubectl set env deployment/musicgen-api -n production FEATURE_ANALYTICS=false
kubectl set env deployment/musicgen-api -n production FEATURE_RECOMMENDATIONS=false

# 6. Increase rate limits temporarily
redis-cli -h redis.musicgen.ai SET emergency_rate_limit_multiplier 3 EX 3600

# 7. Alert team
curl -X POST $SLACK_WEBHOOK_URL \
    -H 'Content-Type: application/json' \
    -d '{
        "text": "🚨 EMERGENCY SCALING ACTIVATED",
        "attachments": [{
            "color": "danger",
            "fields": [{
                "title": "System Status",
                "value": "Emergency scaling in progress",
                "short": true
            }]
        }]
    }'

echo "✅ Emergency scaling completed"
echo "Monitor system performance and scale down when load decreases"
```

---

**Remember**: Always monitor the impact of scaling operations and have rollback procedures ready. Scale gradually when possible, and use aggressive scaling only during emergencies.