# Music Gen AI - Comprehensive Deployment Guide

**Version**: 1.0.0  
**Last Updated**: January 2024  
**Classification**: Production Critical

## Table of Contents

1. [Infrastructure Requirements](#infrastructure-requirements)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Procedures](#deployment-procedures)
4. [Post-Deployment Verification](#post-deployment-verification)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Scaling Procedures](#scaling-procedures)
7. [Disaster Recovery Plan](#disaster-recovery-plan)
8. [Security Procedures](#security-procedures)
9. [Monitoring & Alerting](#monitoring--alerting)
10. [Rollback Procedures](#rollback-procedures)

---

## Infrastructure Requirements

### 1. Hardware Requirements

#### Minimum Production Configuration
```yaml
# API Servers (3x for HA)
CPU: 8 cores (Intel Xeon or AMD EPYC)
RAM: 32GB DDR4 ECC
Storage: 500GB NVMe SSD
Network: 10Gbps

# GPU Servers (2x minimum)
CPU: 16 cores
RAM: 128GB DDR4 ECC
GPU: NVIDIA A100 80GB or 4x RTX 4090
Storage: 2TB NVMe SSD
Network: 25Gbps

# Database Servers (3x for replication)
CPU: 16 cores
RAM: 64GB DDR4 ECC
Storage: 2TB NVMe SSD (RAID 10)
Network: 10Gbps
```

#### Recommended Production Configuration
```yaml
# API Servers (5x with auto-scaling)
CPU: 16 cores
RAM: 64GB DDR4 ECC
Storage: 1TB NVMe SSD
Network: 25Gbps

# GPU Cluster (4x nodes)
CPU: 32 cores
RAM: 256GB DDR4 ECC
GPU: 2x NVIDIA A100 80GB per node
Storage: 4TB NVMe SSD
Network: 100Gbps InfiniBand

# Database Cluster (5x with sharding)
CPU: 32 cores
RAM: 128GB DDR4 ECC
Storage: 4TB NVMe SSD (RAID 10)
Network: 25Gbps
```

### 2. Software Requirements

#### Operating System
```bash
# Recommended OS
Ubuntu 22.04 LTS Server (production)
Rocky Linux 9 (alternative)

# Kernel requirements
Kernel: 5.15+ with NVIDIA drivers
CUDA: 12.0+
cuDNN: 8.9+
```

#### Core Dependencies
```yaml
Python: 3.10+
Node.js: 20.x LTS
PostgreSQL: 15+
Redis: 7.0+
Nginx: 1.24+
Docker: 24.0+
Kubernetes: 1.28+
```

#### Python Environment
```bash
# Production Python packages
torch==2.1.0+cu121
transformers==4.36.0
fastapi==0.108.0
uvicorn[standard]==0.25.0
celery==5.3.4
redis==5.0.1
psycopg2-binary==2.9.9
prometheus-client==0.19.0
structlog==24.1.0
```

### 3. Network Requirements

#### Bandwidth Requirements
```yaml
Inbound:
  API Traffic: 100 Mbps sustained, 1 Gbps burst
  Audio Upload: 500 Mbps sustained
  
Outbound:
  Audio Delivery: 1 Gbps sustained, 10 Gbps burst
  CDN Sync: 500 Mbps sustained

Internal:
  GPU Cluster: 100 Gbps InfiniBand
  Database Replication: 10 Gbps dedicated
  Cache Layer: 25 Gbps
```

#### Network Security
```yaml
Firewall Rules:
  - Allow 443/tcp from 0.0.0.0/0 (HTTPS)
  - Allow 80/tcp from 0.0.0.0/0 (HTTP redirect)
  - Allow 22/tcp from bastion only (SSH)
  - Allow 5432/tcp within VPC (PostgreSQL)
  - Allow 6379/tcp within VPC (Redis)
  - Deny all other inbound

Load Balancer:
  Type: Application Load Balancer (ALB)
  SSL: TLS 1.3 only
  WAF: Enabled with OWASP rules
```

### 4. Cloud Infrastructure (AWS Example)

```yaml
# Production VPC Configuration
VPC:
  CIDR: 10.0.0.0/16
  Availability Zones: 3
  
Subnets:
  Public: 10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24
  Private: 10.0.11.0/24, 10.0.12.0/24, 10.0.13.0/24
  Database: 10.0.21.0/24, 10.0.22.0/24, 10.0.23.0/24

EC2 Instances:
  API: c6i.2xlarge (auto-scaling 3-10)
  GPU: p4d.24xlarge (2 minimum)
  Database: r6i.4xlarge (3x Multi-AZ)
  
Storage:
  EBS: gp3 with 16,000 IOPS
  S3: Standard for audio files
  EFS: For shared model storage
  
Additional Services:
  CloudFront: Global CDN
  Route53: DNS management
  ACM: SSL certificates
  Secrets Manager: Credentials
  CloudWatch: Monitoring
  SNS: Alerting
```

---

## Pre-Deployment Checklist

### 1. Infrastructure Verification

```bash
#!/bin/bash
# Infrastructure verification script

echo "=== Infrastructure Verification Checklist ==="

# Check CPU cores
check_cpu() {
    cores=$(nproc)
    if [ $cores -lt 8 ]; then
        echo "❌ CPU: $cores cores (minimum 8 required)"
        return 1
    else
        echo "✅ CPU: $cores cores"
        return 0
    fi
}

# Check RAM
check_ram() {
    ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ $ram_gb -lt 32 ]; then
        echo "❌ RAM: ${ram_gb}GB (minimum 32GB required)"
        return 1
    else
        echo "✅ RAM: ${ram_gb}GB"
        return 0
    fi
}

# Check GPU
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ GPU: NVIDIA driver not installed"
        return 1
    fi
    
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ $gpu_count -eq 0 ]; then
        echo "❌ GPU: No NVIDIA GPUs detected"
        return 1
    else
        echo "✅ GPU: $gpu_count NVIDIA GPUs detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    fi
}

# Check disk space
check_disk() {
    disk_gb=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $disk_gb -lt 100 ]; then
        echo "❌ Disk: ${disk_gb}GB free (minimum 100GB required)"
        return 1
    else
        echo "✅ Disk: ${disk_gb}GB free"
        return 0
    fi
}

# Check network
check_network() {
    if ping -c 1 8.8.8.8 &> /dev/null; then
        echo "✅ Network: Internet connectivity verified"
        return 0
    else
        echo "❌ Network: No internet connectivity"
        return 1
    fi
}

# Run all checks
check_cpu
check_ram
check_gpu
check_disk
check_network
```

### 2. Security Checklist

- [ ] **SSL Certificates**
  ```bash
  # Verify SSL certificates
  openssl x509 -in /etc/ssl/certs/musicgen.crt -text -noout
  # Check expiration
  openssl x509 -enddate -noout -in /etc/ssl/certs/musicgen.crt
  ```

- [ ] **Firewall Configuration**
  ```bash
  # Verify firewall rules
  sudo ufw status verbose
  # Test from external host
  nmap -Pn -p 22,80,443,5432,6379 production.musicgen.ai
  ```

- [ ] **Secrets Management**
  ```bash
  # Verify all secrets are in environment variables
  ./scripts/check_secrets.sh
  # Ensure no hardcoded credentials
  grep -r "password\|secret\|key" --include="*.py" --include="*.js" .
  ```

- [ ] **Security Headers**
  ```bash
  # Test security headers
  curl -I https://api.musicgen.ai | grep -E "Strict-Transport-Security|X-Frame-Options|X-Content-Type-Options|Content-Security-Policy"
  ```

### 3. Database Preparation

- [ ] **Database Migrations**
  ```bash
  # Backup existing database
  pg_dump -h localhost -U musicgen -d musicgen_prod > backup_$(date +%Y%m%d_%H%M%S).sql
  
  # Run migrations
  alembic upgrade head
  
  # Verify schema
  psql -U musicgen -d musicgen_prod -c "\dt"
  ```

- [ ] **Database Optimization**
  ```sql
  -- Analyze tables
  ANALYZE;
  
  -- Check for missing indexes
  SELECT schemaname, tablename, indexname, idx_scan
  FROM pg_stat_user_indexes
  WHERE idx_scan = 0
  ORDER BY schemaname, tablename;
  
  -- Vacuum full (maintenance window required)
  VACUUM FULL ANALYZE;
  ```

### 4. Application Readiness

- [ ] **Code Quality**
  ```bash
  # Run all tests
  pytest tests/ -v --cov=music_gen --cov-report=html
  
  # Check code quality
  black --check music_gen/
  isort --check-only music_gen/
  flake8 music_gen/
  mypy music_gen/ --strict
  
  # Security scan
  bandit -r music_gen/
  safety check
  ```

- [ ] **Model Verification**
  ```python
  # Verify all models are accessible
  python scripts/verify_models.py
  
  # Test model loading and inference
  python scripts/test_model_inference.py
  ```

- [ ] **Configuration Validation**
  ```bash
  # Validate all configuration files
  python scripts/validate_configs.py
  
  # Check environment variables
  python scripts/check_env_vars.py
  ```

---

## Deployment Procedures

### 1. Blue-Green Deployment

```bash
#!/bin/bash
# Blue-Green Deployment Script

set -euo pipefail

# Configuration
BLUE_ENV="blue"
GREEN_ENV="green"
HEALTH_CHECK_URL="https://api.musicgen.ai/health"
LOAD_BALANCER="musicgen-alb"

# Function to check environment health
check_health() {
    local env=$1
    local url="${HEALTH_CHECK_URL}?env=${env}"
    local max_attempts=30
    local attempt=1
    
    echo "Checking health of ${env} environment..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "${url}" > /dev/null; then
            echo "✅ ${env} environment is healthy"
            return 0
        fi
        
        echo "Attempt ${attempt}/${max_attempts} failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    echo "❌ ${env} environment health check failed"
    return 1
}

# Function to switch traffic
switch_traffic() {
    local from_env=$1
    local to_env=$2
    
    echo "Switching traffic from ${from_env} to ${to_env}..."
    
    # Update load balancer target group
    aws elbv2 modify-listener \
        --listener-arn $(aws elbv2 describe-listeners --load-balancer-arn ${LOAD_BALANCER} --query 'Listeners[0].ListenerArn' --output text) \
        --default-actions Type=forward,TargetGroupArn=$(aws elbv2 describe-target-groups --names "${to_env}-tg" --query 'TargetGroups[0].TargetGroupArn' --output text)
    
    echo "✅ Traffic switched to ${to_env}"
}

# Main deployment flow
main() {
    # Determine current active environment
    CURRENT_ENV=$(aws ssm get-parameter --name "/musicgen/active-env" --query 'Parameter.Value' --output text)
    
    if [ "$CURRENT_ENV" == "$BLUE_ENV" ]; then
        NEW_ENV=$GREEN_ENV
    else
        NEW_ENV=$BLUE_ENV
    fi
    
    echo "Current environment: ${CURRENT_ENV}"
    echo "Deploying to: ${NEW_ENV}"
    
    # Deploy to inactive environment
    echo "Deploying application to ${NEW_ENV}..."
    ansible-playbook -i inventory/${NEW_ENV} deploy.yml
    
    # Health check new environment
    if ! check_health ${NEW_ENV}; then
        echo "Deployment failed health check"
        exit 1
    fi
    
    # Switch traffic
    switch_traffic ${CURRENT_ENV} ${NEW_ENV}
    
    # Update active environment parameter
    aws ssm put-parameter --name "/musicgen/active-env" --value "${NEW_ENV}" --overwrite
    
    echo "✅ Deployment completed successfully"
}

main
```

### 2. Kubernetes Deployment

```yaml
# musicgen-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: musicgen-api
  namespace: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: musicgen-api
  template:
    metadata:
      labels:
        app: musicgen-api
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - musicgen-api
            topologyKey: kubernetes.io/hostname
      containers:
      - name: api
        image: musicgen/api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: musicgen-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: musicgen-secrets
              key: redis-url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: musicgen-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: musicgen-api
  namespace: production
spec:
  selector:
    app: musicgen-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
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
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. Docker Compose Deployment (Development/Staging)

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  nginx:
    image: nginx:1.24-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - static_volume:/app/static:ro
    depends_on:
      - api
    networks:
      - musicgen-network
    restart: unless-stopped

  api:
    image: musicgen/api:${VERSION:-latest}
    environment:
      - DATABASE_URL=postgresql://musicgen:${DB_PASSWORD}@postgres:5432/musicgen
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - MODEL_CACHE_DIR=/models
      - LOG_LEVEL=INFO
    volumes:
      - model_cache:/models:ro
      - static_volume:/app/static
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - musicgen-network
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  worker:
    image: musicgen/worker:${VERSION:-latest}
    environment:
      - DATABASE_URL=postgresql://musicgen:${DB_PASSWORD}@postgres:5432/musicgen
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - MODEL_CACHE_DIR=/models
    volumes:
      - model_cache:/models:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - musicgen-network
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=musicgen
      - POSTGRES_USER=musicgen
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_INITDB_ARGS=--encoding=UTF8 --locale=C
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    networks:
      - musicgen-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U musicgen"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - musicgen-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - musicgen-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    networks:
      - musicgen-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  model_cache:
  static_volume:
  prometheus_data:
  grafana_data:

networks:
  musicgen-network:
    driver: bridge
```

---

## Post-Deployment Verification

### 1. Health Check Script

```python
#!/usr/bin/env python3
"""
Post-deployment health verification script
"""
import asyncio
import aiohttp
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

class HealthChecker:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.results = []
        
    async def check_endpoint(self, session: aiohttp.ClientSession, 
                           method: str, path: str, 
                           expected_status: int = 200,
                           data: Dict = None) -> Tuple[bool, str]:
        """Check a single endpoint"""
        url = f"{self.base_url}{path}"
        headers = {'X-API-Key': self.api_key}
        
        if data:
            headers['Content-Type'] = 'application/json'
            
        try:
            async with session.request(method, url, headers=headers, 
                                     json=data, timeout=30) as response:
                if response.status == expected_status:
                    return True, f"✅ {method} {path} - Status: {response.status}"
                else:
                    text = await response.text()
                    return False, f"❌ {method} {path} - Status: {response.status}, Body: {text[:200]}"
        except Exception as e:
            return False, f"❌ {method} {path} - Error: {str(e)}"
    
    async def run_checks(self):
        """Run all health checks"""
        async with aiohttp.ClientSession() as session:
            # Basic health endpoints
            checks = [
                ('GET', '/health', 200, None),
                ('GET', '/metrics', 200, None),
                ('GET', '/v1/models', 200, None),
                ('GET', '/v1/user/profile', 200, None),
            ]
            
            # Test generation endpoint
            generation_data = {
                "prompt": "Health check test",
                "duration": 5,
                "model": "musicgen-small"
            }
            checks.append(('POST', '/v1/generate', 202, generation_data))
            
            # Run all checks
            for method, path, expected_status, data in checks:
                success, message = await self.check_endpoint(
                    session, method, path, expected_status, data
                )
                self.results.append((success, message))
                print(message)
                
                # Small delay between checks
                await asyncio.sleep(0.5)
    
    async def check_database(self):
        """Check database connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/internal/database/status"
                headers = {'X-API-Key': self.api_key}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Database - Status: Connected, Version: {data.get('version')}")
                        return True
                    else:
                        print(f"❌ Database - Status: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Database - Error: {str(e)}")
            return False
    
    async def check_redis(self):
        """Check Redis connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/internal/cache/status"
                headers = {'X-API-Key': self.api_key}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Redis - Status: Connected, Memory: {data.get('memory_used')}")
                        return True
                    else:
                        print(f"❌ Redis - Status: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Redis - Error: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate health check report"""
        total_checks = len(self.results)
        passed_checks = sum(1 for success, _ in self.results if success)
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "success_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            "details": [message for _, message in self.results]
        }
        
        # Save report
        with open('health_check_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*50)
        print(f"Health Check Summary:")
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print("="*50)
        
        return report['success_rate'] == 100

async def main():
    # Configuration
    BASE_URL = "https://api.musicgen.ai"
    API_KEY = "sk_live_health_check_key"
    
    print(f"Running post-deployment health checks for {BASE_URL}")
    print("="*50)
    
    checker = HealthChecker(BASE_URL, API_KEY)
    
    # Run checks
    await checker.run_checks()
    await checker.check_database()
    await checker.check_redis()
    
    # Generate report
    all_passed = checker.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Performance Verification

```python
#!/usr/bin/env python3
"""
Performance verification script
"""
import time
import requests
import concurrent.futures
from statistics import mean, stdev

def test_endpoint_performance(url: str, headers: dict, num_requests: int = 100):
    """Test endpoint performance"""
    response_times = []
    errors = 0
    
    for _ in range(num_requests):
        start_time = time.time()
        try:
            response = requests.get(url, headers=headers)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                response_times.append(response_time)
            else:
                errors += 1
        except Exception:
            errors += 1
    
    if response_times:
        return {
            "endpoint": url,
            "requests": num_requests,
            "successful": len(response_times),
            "errors": errors,
            "avg_response_time": mean(response_times),
            "std_dev": stdev(response_times) if len(response_times) > 1 else 0,
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p95": sorted(response_times)[int(len(response_times) * 0.95)]
        }
    else:
        return {"endpoint": url, "error": "All requests failed"}

def load_test(base_url: str, api_key: str, concurrent_users: int = 10):
    """Run load test"""
    headers = {'X-API-Key': api_key}
    endpoints = [
        f"{base_url}/health",
        f"{base_url}/v1/models",
        f"{base_url}/v1/user/profile"
    ]
    
    print(f"Running load test with {concurrent_users} concurrent users...")
    print("="*60)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = []
        for endpoint in endpoints:
            for _ in range(concurrent_users):
                future = executor.submit(test_endpoint_performance, endpoint, headers, 10)
                futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Aggregate results by endpoint
    endpoint_stats = {}
    for result in results:
        endpoint = result.get('endpoint', 'unknown')
        if endpoint not in endpoint_stats:
            endpoint_stats[endpoint] = []
        endpoint_stats[endpoint].append(result)
    
    # Print results
    for endpoint, stats in endpoint_stats.items():
        successful_runs = [s for s in stats if 'avg_response_time' in s]
        if successful_runs:
            avg_times = [s['avg_response_time'] for s in successful_runs]
            print(f"\nEndpoint: {endpoint}")
            print(f"  Average Response Time: {mean(avg_times):.2f} ms")
            print(f"  Min Response Time: {min(s['min_response_time'] for s in successful_runs):.2f} ms")
            print(f"  Max Response Time: {max(s['max_response_time'] for s in successful_runs):.2f} ms")
            print(f"  Success Rate: {sum(s['successful'] for s in stats) / sum(s['requests'] for s in stats) * 100:.1f}%")

if __name__ == "__main__":
    BASE_URL = "https://api.musicgen.ai"
    API_KEY = "sk_live_performance_test_key"
    
    load_test(BASE_URL, API_KEY)
```

---

## Troubleshooting Guide

### 1. Common Issues and Solutions

#### API Server Issues

**Problem: API server not starting**
```bash
# Check logs
journalctl -u musicgen-api -n 100 -f

# Common causes and solutions:
# 1. Port already in use
sudo lsof -i :8000
sudo kill -9 <PID>

# 2. Missing environment variables
export DATABASE_URL="postgresql://user:pass@localhost/db"
export REDIS_URL="redis://localhost:6379/0"

# 3. Permission issues
sudo chown -R musicgen:musicgen /app
sudo chmod -R 755 /app
```

**Problem: High memory usage**
```python
# Debug memory usage
import tracemalloc
import psutil

# Start tracing
tracemalloc.start()

# Check current memory
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Get top memory allocations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

#### Database Issues

**Problem: Database connection refused**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U musicgen -d musicgen_prod -c "SELECT 1"

# Common fixes:
# 1. Update pg_hba.conf
sudo vim /etc/postgresql/15/main/pg_hba.conf
# Add: host all all 10.0.0.0/16 md5

# 2. Update postgresql.conf
sudo vim /etc/postgresql/15/main/postgresql.conf
# Set: listen_addresses = '*'

# 3. Restart PostgreSQL
sudo systemctl restart postgresql
```

**Problem: Slow queries**
```sql
-- Find slow queries
SELECT 
    query,
    mean_exec_time,
    calls,
    total_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_generations_user_id ON generations(user_id);
CREATE INDEX CONCURRENTLY idx_generations_created_at ON generations(created_at);

-- Analyze tables
VACUUM ANALYZE generations;
```

#### GPU/Model Issues

**Problem: CUDA out of memory**
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce batch size dynamically
try:
    output = model.generate(inputs, max_length=512)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Retry with smaller batch
    output = model.generate(inputs[:len(inputs)//2], max_length=512)
```

**Problem: Model not loading**
```bash
# Check model files
ls -la /models/musicgen/

# Download missing models
python scripts/download_models.py --model musicgen-medium

# Verify model integrity
python scripts/verify_model_checksum.py --model musicgen-medium

# Test model loading
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('/models/musicgen/medium')"
```

#### Network Issues

**Problem: SSL certificate errors**
```bash
# Check certificate validity
openssl s_client -connect api.musicgen.ai:443 -servername api.musicgen.ai

# Update certificates
sudo apt-get update
sudo apt-get install ca-certificates

# For self-signed certificates in development
export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.crt
export SSL_CERT_FILE=/path/to/ca-bundle.crt
```

**Problem: Load balancer health checks failing**
```bash
# Test health endpoint directly
curl -i http://localhost:8000/health

# Check nginx logs
tail -f /var/log/nginx/error.log

# Verify upstream servers
nginx -t
systemctl reload nginx

# Debug with tcpdump
sudo tcpdump -i any -n port 8000
```

### 2. Emergency Procedures

#### System Overload
```bash
#!/bin/bash
# Emergency response script for system overload

# 1. Enable maintenance mode
echo "true" > /app/maintenance_mode

# 2. Scale down non-critical services
kubectl scale deployment musicgen-worker --replicas=1

# 3. Clear cache
redis-cli FLUSHDB

# 4. Increase rate limits
export RATE_LIMIT_MULTIPLIER=0.5

# 5. Alert team
./scripts/send_alert.sh "System overload - emergency measures activated"
```

#### Security Incident
```bash
#!/bin/bash
# Security incident response

# 1. Block suspicious IPs
iptables -A INPUT -s $SUSPICIOUS_IP -j DROP

# 2. Rotate all secrets
./scripts/rotate_secrets.sh --all

# 3. Enable enhanced logging
export LOG_LEVEL=DEBUG
export AUDIT_MODE=true

# 4. Backup current state
pg_dump musicgen_prod > emergency_backup_$(date +%Y%m%d_%H%M%S).sql

# 5. Notify security team
./scripts/security_alert.sh "Potential security incident detected"
```

---

## Scaling Procedures

### 1. Horizontal Scaling

#### Auto-scaling Configuration
```yaml
# AWS Auto Scaling Group
AutoScalingGroup:
  Type: AWS::AutoScaling::AutoScalingGroup
  Properties:
    MinSize: 3
    MaxSize: 20
    DesiredCapacity: 5
    TargetGroupARNs:
      - !Ref ALBTargetGroup
    HealthCheckType: ELB
    HealthCheckGracePeriod: 300
    LaunchTemplate:
      LaunchTemplateId: !Ref APILaunchTemplate
      Version: !GetAtt APILaunchTemplate.LatestVersionNumber
    MetricsCollection:
      - Granularity: "1Minute"
        Metrics:
          - GroupMinSize
          - GroupMaxSize
          - GroupDesiredCapacity
          - GroupInServiceInstances

# Scaling Policies
ScaleUpPolicy:
  Type: AWS::AutoScaling::ScalingPolicy
  Properties:
    AutoScalingGroupName: !Ref AutoScalingGroup
    PolicyType: TargetTrackingScaling
    TargetTrackingConfiguration:
      PredefinedMetricSpecification:
        PredefinedMetricType: ASGAverageCPUUtilization
      TargetValue: 70.0

CustomMetricScaling:
  Type: AWS::AutoScaling::ScalingPolicy
  Properties:
    AutoScalingGroupName: !Ref AutoScalingGroup
    PolicyType: TargetTrackingScaling
    TargetTrackingConfiguration:
      CustomizedMetricSpecification:
        MetricName: RequestCountPerTarget
        Namespace: AWS/ApplicationELB
        Statistic: Average
      TargetValue: 1000.0
```

#### Kubernetes HPA
```yaml
# Horizontal Pod Autoscaler with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: musicgen-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: musicgen-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  - type: External
    external:
      metric:
        name: queue_length
        selector:
          matchLabels:
            queue: generation_tasks
      target:
        type: Value
        value: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### 2. Vertical Scaling

#### Resource Optimization Script
```python
#!/usr/bin/env python3
"""
Vertical scaling optimization script
"""
import psutil
import statistics
from datetime import datetime, timedelta

class ResourceOptimizer:
    def __init__(self):
        self.metrics_history = []
        
    def collect_metrics(self):
        """Collect current system metrics"""
        return {
            'timestamp': datetime.utcnow(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters(),
            'network_io': psutil.net_io_counters()
        }
    
    def analyze_scaling_needs(self, days=7):
        """Analyze if vertical scaling is needed"""
        # Get metrics for the past N days
        recent_metrics = [m for m in self.metrics_history 
                         if m['timestamp'] > datetime.utcnow() - timedelta(days=days)]
        
        if not recent_metrics:
            return "Insufficient data for analysis"
        
        # Calculate statistics
        cpu_values = [m['cpu_percent'] for m in recent_metrics]
        memory_values = [m['memory_percent'] for m in recent_metrics]
        
        recommendations = []
        
        # CPU analysis
        cpu_p95 = statistics.quantiles(cpu_values, n=20)[18]  # 95th percentile
        if cpu_p95 > 80:
            recommendations.append({
                'resource': 'CPU',
                'current_p95': cpu_p95,
                'recommendation': 'Increase CPU cores by 50%',
                'urgency': 'high' if cpu_p95 > 90 else 'medium'
            })
        
        # Memory analysis
        memory_p95 = statistics.quantiles(memory_values, n=20)[18]
        if memory_p95 > 85:
            recommendations.append({
                'resource': 'Memory',
                'current_p95': memory_p95,
                'recommendation': 'Increase RAM by 100%',
                'urgency': 'high' if memory_p95 > 95 else 'medium'
            })
        
        return recommendations
    
    def generate_scaling_plan(self, current_instance_type='c6i.2xlarge'):
        """Generate vertical scaling plan"""
        instance_upgrades = {
            'c6i.2xlarge': 'c6i.4xlarge',
            'c6i.4xlarge': 'c6i.8xlarge',
            'c6i.8xlarge': 'c6i.12xlarge',
            'm6i.2xlarge': 'm6i.4xlarge',
            'm6i.4xlarge': 'm6i.8xlarge'
        }
        
        next_instance = instance_upgrades.get(current_instance_type, 'c6i.4xlarge')
        
        return {
            'current_instance': current_instance_type,
            'recommended_instance': next_instance,
            'migration_steps': [
                '1. Create AMI snapshot of current instance',
                '2. Launch new instance with recommended type',
                '3. Restore from AMI snapshot',
                '4. Update DNS/load balancer',
                '5. Monitor for 24 hours',
                '6. Terminate old instance'
            ]
        }
```

### 3. Database Scaling

#### Read Replica Setup
```sql
-- On primary database
CREATE ROLE replica_user WITH REPLICATION LOGIN PASSWORD 'secure_password';

-- Configure primary for replication
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET max_wal_senders = 10;
ALTER SYSTEM SET wal_keep_size = '1GB';
ALTER SYSTEM SET hot_standby = on;

-- Restart PostgreSQL
-- sudo systemctl restart postgresql

-- Create replication slot
SELECT pg_create_physical_replication_slot('replica_1');
```

```bash
# On replica server
# Stop PostgreSQL
sudo systemctl stop postgresql

# Clear data directory
sudo rm -rf /var/lib/postgresql/15/main/*

# Create base backup
sudo -u postgres pg_basebackup \
  -h primary.musicgen.ai \
  -D /var/lib/postgresql/15/main \
  -U replica_user \
  -v -P -W \
  -R -S replica_1

# Configure recovery
echo "primary_conninfo = 'host=primary.musicgen.ai port=5432 user=replica_user'" \
  >> /var/lib/postgresql/15/main/postgresql.auto.conf

# Start replica
sudo systemctl start postgresql
```

#### Sharding Strategy
```python
# Sharding configuration
SHARD_CONFIG = {
    'shards': [
        {'id': 0, 'host': 'shard0.musicgen.ai', 'range': (0, 1000000)},
        {'id': 1, 'host': 'shard1.musicgen.ai', 'range': (1000001, 2000000)},
        {'id': 2, 'host': 'shard2.musicgen.ai', 'range': (2000001, 3000000)},
        {'id': 3, 'host': 'shard3.musicgen.ai', 'range': (3000001, 4000000)},
    ],
    'sharding_key': 'user_id'
}

def get_shard(user_id: int):
    """Determine which shard to use for a user"""
    for shard in SHARD_CONFIG['shards']:
        if shard['range'][0] <= user_id <= shard['range'][1]:
            return shard
    # Default to first shard for out-of-range IDs
    return SHARD_CONFIG['shards'][0]

# Usage
user_id = 1500000
shard = get_shard(user_id)
db_url = f"postgresql://user:pass@{shard['host']}/musicgen"
```

---

## Disaster Recovery Plan

### 1. Backup Strategy

#### Automated Backup Script
```bash
#!/bin/bash
# Comprehensive backup script

set -euo pipefail

# Configuration
BACKUP_DIR="/backups/musicgen"
S3_BUCKET="musicgen-backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

echo "Starting backup at $(date)"

# 1. Database backup
echo "Backing up database..."
pg_dump -h localhost -U musicgen -d musicgen_prod \
  --format=custom \
  --verbose \
  --file="${BACKUP_DIR}/${TIMESTAMP}/database.dump"

# 2. Redis backup
echo "Backing up Redis..."
redis-cli --rdb "${BACKUP_DIR}/${TIMESTAMP}/redis.rdb"

# 3. Configuration files
echo "Backing up configurations..."
tar -czf "${BACKUP_DIR}/${TIMESTAMP}/configs.tar.gz" \
  /app/configs \
  /etc/nginx \
  /etc/systemd/system/musicgen* \
  /etc/postgresql/15/main/*.conf

# 4. Model files (incremental)
echo "Backing up models..."
rsync -av --delete \
  /models/ \
  "${BACKUP_DIR}/${TIMESTAMP}/models/"

# 5. Application logs
echo "Backing up logs..."
tar -czf "${BACKUP_DIR}/${TIMESTAMP}/logs.tar.gz" \
  /var/log/musicgen \
  /var/log/nginx \
  /var/log/postgresql

# 6. Create backup manifest
cat > "${BACKUP_DIR}/${TIMESTAMP}/manifest.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "type": "full",
  "components": {
    "database": {
      "file": "database.dump",
      "size": $(stat -c%s "${BACKUP_DIR}/${TIMESTAMP}/database.dump"),
      "checksum": "$(sha256sum "${BACKUP_DIR}/${TIMESTAMP}/database.dump" | cut -d' ' -f1)"
    },
    "redis": {
      "file": "redis.rdb",
      "size": $(stat -c%s "${BACKUP_DIR}/${TIMESTAMP}/redis.rdb"),
      "checksum": "$(sha256sum "${BACKUP_DIR}/${TIMESTAMP}/redis.rdb" | cut -d' ' -f1)"
    },
    "configs": {
      "file": "configs.tar.gz",
      "size": $(stat -c%s "${BACKUP_DIR}/${TIMESTAMP}/configs.tar.gz"),
      "checksum": "$(sha256sum "${BACKUP_DIR}/${TIMESTAMP}/configs.tar.gz" | cut -d' ' -f1)"
    }
  }
}
EOF

# 7. Upload to S3
echo "Uploading to S3..."
aws s3 sync "${BACKUP_DIR}/${TIMESTAMP}" "s3://${S3_BUCKET}/${TIMESTAMP}/" \
  --storage-class STANDARD_IA

# 8. Cleanup old backups
echo "Cleaning up old backups..."
find "${BACKUP_DIR}" -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} +

# 9. Verify backup
echo "Verifying backup..."
aws s3 ls "s3://${S3_BUCKET}/${TIMESTAMP}/" --recursive

echo "Backup completed successfully at $(date)"

# Send notification
./scripts/send_notification.sh "Backup completed: ${TIMESTAMP}"
```

### 2. Recovery Procedures

#### Full System Recovery
```bash
#!/bin/bash
# Disaster recovery script

set -euo pipefail

# Configuration
BACKUP_TIMESTAMP=$1
S3_BUCKET="musicgen-backups"
RESTORE_DIR="/tmp/restore"

if [ -z "$BACKUP_TIMESTAMP" ]; then
    echo "Usage: $0 <backup_timestamp>"
    echo "Available backups:"
    aws s3 ls "s3://${S3_BUCKET}/" | grep PRE | awk '{print $2}' | sed 's/\///'
    exit 1
fi

echo "Starting disaster recovery for backup: ${BACKUP_TIMESTAMP}"

# 1. Create restore directory
mkdir -p "${RESTORE_DIR}"

# 2. Download backup from S3
echo "Downloading backup from S3..."
aws s3 sync "s3://${S3_BUCKET}/${BACKUP_TIMESTAMP}/" "${RESTORE_DIR}/" 

# 3. Verify backup integrity
echo "Verifying backup integrity..."
if [ -f "${RESTORE_DIR}/manifest.json" ]; then
    # Verify checksums
    while IFS= read -r line; do
        if [[ $line =~ \"checksum\":\ \"([^\"]+)\" ]]; then
            checksum="${BASH_REMATCH[1]}"
            file=$(echo "$line" | grep -oP '(?<="file": ")[^"]+')
            if [ -f "${RESTORE_DIR}/${file}" ]; then
                actual_checksum=$(sha256sum "${RESTORE_DIR}/${file}" | cut -d' ' -f1)
                if [ "$checksum" != "$actual_checksum" ]; then
                    echo "❌ Checksum mismatch for ${file}"
                    exit 1
                fi
            fi
        fi
    done < "${RESTORE_DIR}/manifest.json"
    echo "✅ Backup integrity verified"
else
    echo "⚠️  No manifest found, skipping integrity check"
fi

# 4. Stop services
echo "Stopping services..."
sudo systemctl stop musicgen-api musicgen-worker nginx

# 5. Restore database
echo "Restoring database..."
sudo -u postgres dropdb musicgen_prod || true
sudo -u postgres createdb musicgen_prod
sudo -u postgres pg_restore \
  --dbname=musicgen_prod \
  --verbose \
  --no-owner \
  --no-acl \
  "${RESTORE_DIR}/database.dump"

# 6. Restore Redis
echo "Restoring Redis..."
sudo systemctl stop redis
sudo cp "${RESTORE_DIR}/redis.rdb" /var/lib/redis/dump.rdb
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis

# 7. Restore configurations
echo "Restoring configurations..."
sudo tar -xzf "${RESTORE_DIR}/configs.tar.gz" -C /

# 8. Restore models
echo "Restoring models..."
sudo rsync -av --delete "${RESTORE_DIR}/models/" /models/

# 9. Run database migrations
echo "Running database migrations..."
cd /app && alembic upgrade head

# 10. Start services
echo "Starting services..."
sudo systemctl start musicgen-api musicgen-worker nginx

# 11. Verify services
echo "Verifying services..."
sleep 10
for service in musicgen-api musicgen-worker nginx redis postgresql; do
    if systemctl is-active --quiet $service; then
        echo "✅ $service is running"
    else
        echo "❌ $service is not running"
        sudo journalctl -u $service -n 50
    fi
done

# 12. Run health checks
echo "Running health checks..."
curl -f http://localhost/health || echo "❌ Health check failed"

echo "Disaster recovery completed at $(date)"
```

### 3. RTO/RPO Targets

| Component | RPO (Recovery Point Objective) | RTO (Recovery Time Objective) |
|-----------|-------------------------------|------------------------------|
| Database | 1 hour | 2 hours |
| User Data | 1 hour | 2 hours |
| Generated Audio | 24 hours | 4 hours |
| Configuration | 24 hours | 1 hour |
| Models | 7 days | 4 hours |
| Logs | 24 hours | 8 hours |

### 4. Failover Procedures

#### Database Failover
```python
#!/usr/bin/env python3
"""
Database failover orchestration
"""
import psycopg2
import time
import subprocess
from typing import Dict, List

class DatabaseFailover:
    def __init__(self, config: Dict):
        self.primary = config['primary']
        self.replicas = config['replicas']
        self.vip = config['vip']
        
    def check_primary_health(self) -> bool:
        """Check if primary is healthy"""
        try:
            conn = psycopg2.connect(
                host=self.primary['host'],
                port=self.primary['port'],
                dbname='postgres',
                user='health_check',
                password='password',
                connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Primary health check failed: {e}")
            return False
    
    def promote_replica(self, replica: Dict) -> bool:
        """Promote replica to primary"""
        try:
            # Promote replica
            cmd = f"ssh {replica['host']} 'sudo -u postgres pg_ctl promote -D /var/lib/postgresql/data'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to promote replica: {result.stderr}")
                return False
            
            # Wait for promotion to complete
            time.sleep(5)
            
            # Update VIP
            self.update_vip(replica['host'])
            
            return True
        except Exception as e:
            print(f"Error promoting replica: {e}")
            return False
    
    def update_vip(self, new_primary: str):
        """Update virtual IP to point to new primary"""
        cmd = f"sudo ip addr del {self.vip}/32 dev eth0; ssh {new_primary} 'sudo ip addr add {self.vip}/32 dev eth0'"
        subprocess.run(cmd, shell=True)
    
    def execute_failover(self):
        """Execute failover process"""
        print("Starting failover process...")
        
        # 1. Verify primary is really down
        retry_count = 3
        for i in range(retry_count):
            if self.check_primary_health():
                print(f"Primary came back online (attempt {i+1}/{retry_count})")
                return False
            time.sleep(2)
        
        # 2. Select best replica
        best_replica = self.select_best_replica()
        if not best_replica:
            print("No suitable replica found for failover")
            return False
        
        print(f"Selected replica {best_replica['host']} for promotion")
        
        # 3. Promote replica
        if self.promote_replica(best_replica):
            print(f"Successfully promoted {best_replica['host']} to primary")
            
            # 4. Update application configuration
            self.update_app_config(best_replica)
            
            # 5. Notify operations team
            self.send_notification(f"Database failover completed. New primary: {best_replica['host']}")
            
            return True
        else:
            print("Failover failed")
            return False
    
    def select_best_replica(self) -> Dict:
        """Select the best replica for promotion based on lag and health"""
        candidates = []
        
        for replica in self.replicas:
            try:
                conn = psycopg2.connect(
                    host=replica['host'],
                    port=replica['port'],
                    dbname='postgres',
                    user='health_check',
                    password='password'
                )
                cur = conn.cursor()
                
                # Check replication lag
                cur.execute("""
                    SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int as lag
                """)
                lag = cur.fetchone()[0]
                
                if lag is not None and lag < 10:  # Less than 10 seconds lag
                    candidates.append({
                        **replica,
                        'lag': lag
                    })
                
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Error checking replica {replica['host']}: {e}")
        
        # Sort by lag and return best candidate
        if candidates:
            return sorted(candidates, key=lambda x: x['lag'])[0]
        return None
    
    def update_app_config(self, new_primary: Dict):
        """Update application configuration with new primary"""
        # Update Kubernetes ConfigMap
        cmd = f"""
        kubectl patch configmap musicgen-config -p '{{"data":{{"DATABASE_HOST":"{new_primary['host']}"}}}}'
        kubectl rollout restart deployment musicgen-api
        """
        subprocess.run(cmd, shell=True)
    
    def send_notification(self, message: str):
        """Send notification to operations team"""
        # Implementation depends on notification system
        print(f"ALERT: {message}")

# Usage
if __name__ == "__main__":
    config = {
        'primary': {'host': 'db-primary.musicgen.ai', 'port': 5432},
        'replicas': [
            {'host': 'db-replica-1.musicgen.ai', 'port': 5432},
            {'host': 'db-replica-2.musicgen.ai', 'port': 5432},
        ],
        'vip': '10.0.1.100'
    }
    
    failover = DatabaseFailover(config)
    failover.execute_failover()
```

---

## Security Procedures

### 1. Security Hardening Checklist

#### Operating System Hardening
```bash
#!/bin/bash
# OS Security hardening script

# 1. Update system
apt-get update && apt-get upgrade -y

# 2. Configure firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp  # SSH
ufw allow 443/tcp # HTTPS
ufw --force enable

# 3. Disable unnecessary services
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon

# 4. Configure SSH hardening
cat >> /etc/ssh/sshd_config << EOF
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
AllowUsers musicgen deploy
Protocol 2
EOF

# 5. Install and configure fail2ban
apt-get install -y fail2ban
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = 22
logpath = /var/log/auth.log
EOF

# 6. Configure auditd
apt-get install -y auditd
auditctl -w /etc/passwd -p wa -k passwd_changes
auditctl -w /etc/group -p wa -k group_changes
auditctl -w /etc/sudoers -p wa -k sudoers_changes

# 7. Set up automatic security updates
apt-get install -y unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades

# 8. Configure sysctl for security
cat >> /etc/sysctl.conf << EOF
# IP Spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable IPv6
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
EOF

sysctl -p

# 9. Set file permissions
chmod 644 /etc/passwd
chmod 600 /etc/shadow
chmod 644 /etc/group
chmod 600 /etc/gshadow

# 10. Remove unnecessary packages
apt-get autoremove -y
apt-get autoclean
```

#### Application Security
```python
#!/usr/bin/env python3
"""
Application security configuration
"""
import os
import secrets
from cryptography.fernet import Fernet

class SecurityConfig:
    def __init__(self):
        self.config = {}
        
    def generate_secrets(self):
        """Generate secure secrets"""
        return {
            'SECRET_KEY': secrets.token_urlsafe(64),
            'JWT_SECRET': secrets.token_urlsafe(64),
            'DATABASE_ENCRYPTION_KEY': Fernet.generate_key().decode(),
            'API_KEY_SALT': secrets.token_hex(32),
            'CSRF_TOKEN_SECRET': secrets.token_urlsafe(32)
        }
    
    def configure_cors(self):
        """Configure CORS settings"""
        return {
            'allowed_origins': [
                'https://musicgen.ai',
                'https://app.musicgen.ai'
            ],
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
            'allowed_headers': ['Content-Type', 'Authorization', 'X-API-Key'],
            'expose_headers': ['X-RateLimit-Limit', 'X-RateLimit-Remaining'],
            'allow_credentials': True,
            'max_age': 3600
        }
    
    def configure_headers(self):
        """Configure security headers"""
        return {
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https://api.musicgen.ai wss://api.musicgen.ai; frame-ancestors 'none';",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    def configure_rate_limiting(self):
        """Configure rate limiting"""
        return {
            'default': {
                'requests': 100,
                'window': 3600,  # 1 hour
                'key_func': 'get_remote_address'
            },
            'api_endpoints': {
                '/api/v1/generate': {
                    'requests': 10,
                    'window': 3600
                },
                '/api/v1/auth/login': {
                    'requests': 5,
                    'window': 900  # 15 minutes
                }
            },
            'ip_whitelist': [
                '10.0.0.0/8',
                '172.16.0.0/12',
                '192.168.0.0/16'
            ]
        }
    
    def save_config(self, filename='security_config.json'):
        """Save security configuration"""
        import json
        
        config = {
            'secrets': self.generate_secrets(),
            'cors': self.configure_cors(),
            'headers': self.configure_headers(),
            'rate_limiting': self.configure_rate_limiting()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set appropriate permissions
        os.chmod(filename, 0o600)
        
        print(f"Security configuration saved to {filename}")
        return config

# Generate security configuration
if __name__ == "__main__":
    security = SecurityConfig()
    security.save_config()
```

### 2. Incident Response Plan

#### Incident Response Runbook
```markdown
# Security Incident Response Runbook

## 1. Initial Response (0-15 minutes)

### Identify and Classify
- [ ] Determine incident type:
  - [ ] Data breach
  - [ ] Service compromise
  - [ ] DDoS attack
  - [ ] Insider threat
  - [ ] Malware infection

### Immediate Actions
```bash
# 1. Isolate affected systems
iptables -I INPUT -s <suspicious_ip> -j DROP
iptables -I OUTPUT -d <suspicious_ip> -j DROP

# 2. Preserve evidence
tar -czf /backup/incident_$(date +%Y%m%d_%H%M%S).tar.gz \
  /var/log \
  /tmp \
  /home

# 3. Enable enhanced logging
echo "*.* @@remote-syslog.musicgen.ai:514" >> /etc/rsyslog.conf
systemctl restart rsyslog
```

## 2. Containment (15-60 minutes)

### Network Isolation
```bash
# Create isolated VLAN
vconfig add eth0 999
ifconfig eth0.999 10.99.99.1 netmask 255.255.255.0 up

# Move affected systems to isolated network
# Update firewall rules to block external access
```

### Service Degradation
```python
# Enable read-only mode
os.environ['READ_ONLY_MODE'] = 'true'

# Disable high-risk features
DISABLED_FEATURES = [
    'user_registration',
    'api_key_generation',
    'model_updates'
]
```

## 3. Investigation (1-4 hours)

### Log Analysis
```bash
# Search for indicators of compromise
grep -E "failed|error|denied|unauthorized" /var/log/auth.log | tail -100

# Check for suspicious processes
ps aux | grep -v grep | grep -E "nc|ncat|perl|python|bash|sh" | grep -v "^root"

# Network connections
netstat -tulpn | grep ESTABLISHED
lsof -i -P -n | grep LISTEN
```

### Forensic Analysis
```bash
# File integrity check
debsums -c
rpm -Va  # For RedHat-based systems

# Check for rootkits
chkrootkit
rkhunter --check

# Memory dump for analysis
sudo dd if=/dev/mem of=/backup/memory_dump.img bs=1M
```

## 4. Eradication (2-8 hours)

### Remove Threats
```bash
# Kill suspicious processes
kill -9 <pid>

# Remove malicious files
rm -rf /path/to/malicious/files

# Reset compromised accounts
passwd <username>
chage -d 0 <username>  # Force password change
```

### Patch Vulnerabilities
```bash
# Apply security updates
apt-get update && apt-get upgrade -y
# or
yum update -y

# Rebuild affected components
docker build --no-cache -t musicgen/api:emergency .
```

## 5. Recovery (4-24 hours)

### Service Restoration
```bash
# Restore from clean backup
./scripts/restore_from_backup.sh <backup_timestamp>

# Rebuild affected services
docker-compose down
docker-compose up -d --force-recreate

# Verify integrity
./scripts/verify_system_integrity.sh
```

### Monitoring Enhancement
```python
# Add additional monitoring
ENHANCED_MONITORING = {
    'failed_login_threshold': 3,
    'api_anomaly_detection': True,
    'file_integrity_monitoring': True,
    'network_traffic_analysis': True
}
```

## 6. Post-Incident (24-72 hours)

### Documentation
- [ ] Create incident report
- [ ] Document timeline
- [ ] List affected systems
- [ ] Detail remediation steps
- [ ] Calculate impact metrics

### Lessons Learned
- [ ] Conduct post-mortem meeting
- [ ] Update security procedures
- [ ] Implement additional controls
- [ ] Update incident response plan
```

### 3. Security Monitoring

#### Security Event Monitoring
```python
#!/usr/bin/env python3
"""
Real-time security event monitoring
"""
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

class SecurityMonitor:
    def __init__(self):
        self.alerts = []
        self.suspicious_ips = defaultdict(int)
        self.failed_logins = defaultdict(list)
        
    def analyze_auth_log(self, line: str):
        """Analyze authentication log entries"""
        # Failed SSH login
        if "Failed password" in line:
            match = re.search(r'Failed password for .* from ([\d.]+)', line)
            if match:
                ip = match.group(1)
                self.failed_logins[ip].append(datetime.now())
                
                # Check for brute force
                recent_failures = [t for t in self.failed_logins[ip] 
                                 if t > datetime.now() - timedelta(minutes=10)]
                if len(recent_failures) > 5:
                    self.create_alert(
                        'BRUTE_FORCE',
                        f'Possible brute force attack from {ip}',
                        {'ip': ip, 'attempts': len(recent_failures)}
                    )
        
        # Successful login after failures
        elif "Accepted password" in line or "Accepted publickey" in line:
            match = re.search(r'from ([\d.]+)', line)
            if match:
                ip = match.group(1)
                if ip in self.failed_logins and len(self.failed_logins[ip]) > 3:
                    self.create_alert(
                        'SUSPICIOUS_LOGIN',
                        f'Successful login from {ip} after multiple failures',
                        {'ip': ip, 'previous_failures': len(self.failed_logins[ip])}
                    )
    
    def analyze_application_log(self, line: str):
        """Analyze application log entries"""
        # SQL injection attempts
        sql_injection_patterns = [
            r"union.*select",
            r"';.*--",
            r'";.*--',
            r"' or '1'='1",
            r'" or "1"="1',
            r"admin'--",
            r"') or ('1'='1"
        ]
        
        for pattern in sql_injection_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                self.create_alert(
                    'SQL_INJECTION',
                    'Possible SQL injection attempt detected',
                    {'pattern': pattern, 'log_line': line[:200]}
                )
                break
        
        # Path traversal attempts
        if "../" in line or "..%2F" in line or "..%5C" in line:
            self.create_alert(
                'PATH_TRAVERSAL',
                'Possible path traversal attempt detected',
                {'log_line': line[:200]}
            )
        
        # Unauthorized access attempts
        if "403 Forbidden" in line or "401 Unauthorized" in line:
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
            if match:
                ip = match.group(1)
                self.suspicious_ips[ip] += 1
                
                if self.suspicious_ips[ip] > 50:
                    self.create_alert(
                        'EXCESSIVE_403',
                        f'Excessive forbidden requests from {ip}',
                        {'ip': ip, 'count': self.suspicious_ips[ip]}
                    )
    
    def create_alert(self, alert_type: str, message: str, details: dict):
        """Create security alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'details': details,
            'severity': self.get_severity(alert_type)
        }
        
        self.alerts.append(alert)
        
        # Send immediate notification for high severity
        if alert['severity'] in ['HIGH', 'CRITICAL']:
            self.send_notification(alert)
        
        # Log alert
        print(f"[SECURITY ALERT] {alert['severity']} - {message}")
    
    def get_severity(self, alert_type: str) -> str:
        """Determine alert severity"""
        severity_map = {
            'BRUTE_FORCE': 'HIGH',
            'SQL_INJECTION': 'CRITICAL',
            'PATH_TRAVERSAL': 'HIGH',
            'SUSPICIOUS_LOGIN': 'MEDIUM',
            'EXCESSIVE_403': 'MEDIUM'
        }
        return severity_map.get(alert_type, 'LOW')
    
    def send_notification(self, alert: dict):
        """Send email notification for critical alerts"""
        try:
            msg = MIMEText(f"""
Security Alert: {alert['message']}

Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

Details:
{json.dumps(alert['details'], indent=2)}

Please investigate immediately.
            """)
            
            msg['Subject'] = f"[{alert['severity']}] Security Alert - {alert['type']}"
            msg['From'] = 'security@musicgen.ai'
            msg['To'] = 'security-team@musicgen.ai'
            
            # Send email (configure SMTP server)
            # smtp = smtplib.SMTP('localhost')
            # smtp.send_message(msg)
            # smtp.quit()
            
        except Exception as e:
            print(f"Failed to send notification: {e}")

# Usage
if __name__ == "__main__":
    monitor = SecurityMonitor()
    
    # Example: Monitor logs in real-time
    import subprocess
    
    # Monitor auth log
    auth_log = subprocess.Popen(
        ['tail', '-f', '/var/log/auth.log'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    for line in auth_log.stdout:
        monitor.analyze_auth_log(line.decode('utf-8'))
```

---

## Monitoring & Alerting

### 1. Comprehensive Monitoring Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    region: 'us-east-1'

# Alerting
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Rule files
rule_files:
  - "alerts/*.yml"
  - "recording/*.yml"

# Scrape configs
scrape_configs:
  # API metrics
  - job_name: 'musicgen-api'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - production
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

  # Node exporter
  - job_name: 'node-exporter'
    static_configs:
      - targets:
          - 'node1.musicgen.ai:9100'
          - 'node2.musicgen.ai:9100'
          - 'node3.musicgen.ai:9100'

  # PostgreSQL exporter
  - job_name: 'postgres'
    static_configs:
      - targets:
          - 'postgres-primary.musicgen.ai:9187'
          - 'postgres-replica-1.musicgen.ai:9187'

  # Redis exporter
  - job_name: 'redis'
    static_configs:
      - targets:
          - 'redis.musicgen.ai:9121'

  # GPU metrics
  - job_name: 'gpu-metrics'
    static_configs:
      - targets:
          - 'gpu1.musicgen.ai:9400'
          - 'gpu2.musicgen.ai:9400'
```

#### Critical Alerts
```yaml
# alerts/critical.yml
groups:
  - name: critical_alerts
    interval: 30s
    rules:
      # API availability
      - alert: APIDown
        expr: up{job="musicgen-api"} == 0
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "API instance {{ $labels.instance }} is down"
          description: "API instance {{ $labels.instance }} has been down for more than 2 minutes."

      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m]))
            /
            sum(rate(http_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      # Database connection issues
      - alert: DatabaseConnectionFailure
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
          team: database
        annotations:
          summary: "PostgreSQL instance {{ $labels.instance }} is down"
          description: "Unable to connect to PostgreSQL instance {{ $labels.instance }}"

      # Disk space critical
      - alert: DiskSpaceCritical
        expr: |
          (
            node_filesystem_avail_bytes{mountpoint="/"}
            /
            node_filesystem_size_bytes{mountpoint="/"}
          ) < 0.05
        for: 5m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "Critical disk space on {{ $labels.instance }}"
          description: "Disk space is below 5% on {{ $labels.instance }}"

      # Memory pressure
      - alert: HighMemoryPressure
        expr: |
          (
            1 - (
              node_memory_MemAvailable_bytes
              /
              node_memory_MemTotal_bytes
            )
          ) > 0.95
        for: 5m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "High memory pressure on {{ $labels.instance }}"
          description: "Memory usage is above 95% on {{ $labels.instance }}"

      # GPU errors
      - alert: GPUError
        expr: nvidia_gpu_power_state > 0
        for: 1m
        labels:
          severity: critical
          team: ml
        annotations:
          summary: "GPU error on {{ $labels.instance }}"
          description: "GPU {{ $labels.gpu }} is in error state on {{ $labels.instance }}"
```

### 2. Alerting Configuration

#### AlertManager Setup
```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  smtp_from: 'alerts@musicgen.ai'
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_auth_username: 'alerts@musicgen.ai'
  smtp_auth_password: 'secure_password'

templates:
  - '/etc/alertmanager/templates/*.tmpl'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  
  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: pagerduty
      continue: true
      
    # High severity to Slack
    - match:
        severity: high
      receiver: slack
      continue: true
      
    # Database alerts to DBA team
    - match:
        team: database
      receiver: dba-team
      
    # ML alerts to ML team
    - match:
        team: ml
      receiver: ml-team

receivers:
  - name: 'default'
    email_configs:
      - to: 'platform-team@musicgen.ai'
        headers:
          Subject: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'your-pagerduty-service-key'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'dba-team'
    email_configs:
      - to: 'dba-team@musicgen.ai'
        headers:
          Subject: '[DB Alert] {{ .GroupLabels.alertname }}'

  - name: 'ml-team'
    email_configs:
      - to: 'ml-team@musicgen.ai'
        headers:
          Subject: '[ML Alert] {{ .GroupLabels.alertname }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

### 3. Custom Dashboards

#### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "id": null,
    "uid": "musicgen-prod",
    "title": "MusicGen Production Dashboard",
    "tags": ["production", "musicgen"],
    "timezone": "browser",
    "schemaVersion": 30,
    "version": 0,
    "refresh": "10s",
    "panels": [
      {
        "id": 1,
        "type": "graph",
        "title": "API Request Rate",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (method, status)",
            "legendFormat": "{{method}} - {{status}}"
          }
        ]
      },
      {
        "id": 2,
        "type": "stat",
        "title": "API Uptime",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "avg(up{job=\"musicgen-api\"}) * 100",
            "format": "percent"
          }
        ]
      },
      {
        "id": 3,
        "type": "gauge",
        "title": "Current GPU Usage",
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
        "targets": [
          {
            "expr": "avg(nvidia_gpu_usage)",
            "format": "percent"
          }
        ]
      },
      {
        "id": 4,
        "type": "graph",
        "title": "Generation Latency",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(generation_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(generation_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "id": 5,
        "type": "table",
        "title": "Active Generations",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "generation_active_tasks",
            "format": "table",
            "instant": true
          }
        ]
      }
    ]
  }
}
```

---

## Rollback Procedures

### 1. Application Rollback

#### Automated Rollback Script
```bash
#!/bin/bash
# Automated rollback script with health checks

set -euo pipefail

# Configuration
ROLLBACK_VERSION=${1:-"previous"}
HEALTH_CHECK_URL="https://api.musicgen.ai/health"
SMOKE_TEST_URL="https://api.musicgen.ai/v1/models"
MAX_ROLLBACK_TIME=600  # 10 minutes

echo "=== Starting Rollback to Version: ${ROLLBACK_VERSION} ==="

# Function to check service health
check_health() {
    local url=$1
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "${url}" > /dev/null; then
            return 0
        fi
        echo "Health check attempt ${attempt}/${max_attempts} failed..."
        sleep 10
        ((attempt++))
    done
    
    return 1
}

# 1. Create rollback checkpoint
echo "Creating rollback checkpoint..."
kubectl create configmap rollback-checkpoint \
  --from-literal=timestamp=$(date -u +%Y%m%dT%H%M%SZ) \
  --from-literal=current_version=$(kubectl get deployment musicgen-api -o jsonpath='{.spec.template.spec.containers[0].image}')

# 2. Scale down to minimal capacity
echo "Scaling down to minimal capacity..."
kubectl scale deployment musicgen-api --replicas=1

# 3. Perform rollback
if [ "$ROLLBACK_VERSION" == "previous" ]; then
    echo "Rolling back to previous version..."
    kubectl rollout undo deployment musicgen-api
else
    echo "Rolling back to specific version: ${ROLLBACK_VERSION}..."
    kubectl set image deployment/musicgen-api api=musicgen/api:${ROLLBACK_VERSION}
fi

# 4. Wait for rollout to complete
echo "Waiting for rollback to complete..."
if ! kubectl rollout status deployment musicgen-api --timeout=${MAX_ROLLBACK_TIME}s; then
    echo "❌ Rollback failed to complete within ${MAX_ROLLBACK_TIME} seconds"
    exit 1
fi

# 5. Verify health
echo "Verifying service health..."
if ! check_health "${HEALTH_CHECK_URL}"; then
    echo "❌ Health check failed after rollback"
    # Attempt to restore original version
    ORIGINAL_VERSION=$(kubectl get configmap rollback-checkpoint -o jsonpath='{.data.current_version}')
    kubectl set image deployment/musicgen-api api=${ORIGINAL_VERSION}
    exit 1
fi

# 6. Run smoke tests
echo "Running smoke tests..."
if ! curl -sf "${SMOKE_TEST_URL}" > /dev/null; then
    echo "⚠️  Smoke test failed but health check passed"
fi

# 7. Scale back up
echo "Scaling back to normal capacity..."
kubectl scale deployment musicgen-api --replicas=3

# 8. Clean up checkpoint
kubectl delete configmap rollback-checkpoint --ignore-not-found=true

echo "✅ Rollback completed successfully"

# 9. Send notification
./scripts/send_notification.sh "Rollback completed to version ${ROLLBACK_VERSION}"
```

### 2. Database Rollback

#### Database Version Control
```sql
-- Create version tracking table
CREATE TABLE IF NOT EXISTS schema_versions (
    version_id SERIAL PRIMARY KEY,
    version_number VARCHAR(20) NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(100),
    migration_script TEXT,
    rollback_script TEXT,
    checksum VARCHAR(64)
);

-- Example migration with rollback
INSERT INTO schema_versions (version_number, applied_by, migration_script, rollback_script) VALUES (
    '2024.01.15.001',
    'deploy_user',
    '-- Migration
    CREATE TABLE user_preferences (
        user_id UUID REFERENCES users(id),
        preference_key VARCHAR(100),
        preference_value JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, preference_key)
    );
    CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);',
    '-- Rollback
    DROP TABLE IF EXISTS user_preferences CASCADE;'
);
```

#### Database Rollback Script
```python
#!/usr/bin/env python3
"""
Database rollback utility
"""
import psycopg2
import hashlib
from datetime import datetime

class DatabaseRollback:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        self.cursor = self.conn.cursor()
        
    def get_current_version(self):
        """Get current database version"""
        self.cursor.execute("""
            SELECT version_number, applied_at 
            FROM schema_versions 
            ORDER BY version_id DESC 
            LIMIT 1
        """)
        return self.cursor.fetchone()
    
    def rollback_to_version(self, target_version):
        """Rollback to specific version"""
        # Get all versions after target
        self.cursor.execute("""
            SELECT version_id, version_number, rollback_script
            FROM schema_versions
            WHERE version_id > (
                SELECT version_id FROM schema_versions 
                WHERE version_number = %s
            )
            ORDER BY version_id DESC
        """, (target_version,))
        
        rollback_versions = self.cursor.fetchall()
        
        if not rollback_versions:
            print(f"Already at version {target_version} or version not found")
            return False
        
        print(f"Will rollback {len(rollback_versions)} versions")
        
        try:
            # Begin transaction
            self.conn.autocommit = False
            
            # Execute rollbacks in reverse order
            for version_id, version_number, rollback_script in rollback_versions:
                print(f"Rolling back version {version_number}...")
                
                if rollback_script:
                    self.cursor.execute(rollback_script)
                    
                # Mark as rolled back
                self.cursor.execute("""
                    UPDATE schema_versions 
                    SET rolled_back_at = %s 
                    WHERE version_id = %s
                """, (datetime.utcnow(), version_id))
            
            # Commit transaction
            self.conn.commit()
            print(f"Successfully rolled back to version {target_version}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"Rollback failed: {e}")
            return False
        finally:
            self.conn.autocommit = True
    
    def verify_database_integrity(self):
        """Verify database integrity after rollback"""
        checks = [
            ("Check foreign keys", """
                SELECT conname, conrelid::regclass 
                FROM pg_constraint 
                WHERE NOT convalidated
            """),
            ("Check indexes", """
                SELECT indexrelid::regclass 
                FROM pg_index 
                WHERE NOT indisvalid
            """),
            ("Check tables", """
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename NOT IN (
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                )
            """)
        ]
        
        all_passed = True
        for check_name, query in checks:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            if results:
                print(f"❌ {check_name}: Found issues")
                for result in results:
                    print(f"   - {result}")
                all_passed = False
            else:
                print(f"✅ {check_name}: Passed")
        
        return all_passed

# Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: rollback_database.py <target_version>")
        sys.exit(1)
    
    target_version = sys.argv[1]
    db = DatabaseRollback("postgresql://user:pass@localhost/musicgen")
    
    current = db.get_current_version()
    print(f"Current version: {current[0]} (applied at {current[1]})")
    
    if db.rollback_to_version(target_version):
        if db.verify_database_integrity():
            print("Database integrity verified")
        else:
            print("Database integrity check failed - manual intervention required")
```

---

## Testing Documentation with New Team Member

### New Team Member Onboarding Checklist

#### Day 1: Environment Setup
- [ ] **Access Verification**
  ```bash
  # Verify SSH access
  ssh deploy@bastion.musicgen.ai
  
  # Test VPN connection
  ping internal-api.musicgen.ai
  
  # Verify AWS CLI access
  aws sts get-caller-identity
  ```

- [ ] **Documentation Review**
  - Read deployment guide thoroughly
  - Note any unclear sections
  - List questions for team lead

- [ ] **Development Environment**
  ```bash
  # Clone repositories
  git clone git@github.com:musicgen/api.git
  git clone git@github.com:musicgen/infrastructure.git
  
  # Install dependencies
  pip install -r requirements-dev.txt
  npm install
  
  # Run local tests
  pytest tests/
  ```

#### Day 2: Staging Deployment
- [ ] **Deploy to Staging**
  ```bash
  # Follow deployment checklist
  ./scripts/deploy_staging.sh
  
  # Verify deployment
  ./scripts/verify_staging.sh
  ```

- [ ] **Practice Rollback**
  ```bash
  # Simulate failure and rollback
  ./scripts/simulate_failure.sh
  ./scripts/rollback_staging.sh
  ```

#### Day 3: Monitoring and Troubleshooting
- [ ] **Access Monitoring Tools**
  - Grafana: https://grafana.musicgen.ai
  - Prometheus: https://prometheus.musicgen.ai
  - Logs: https://logs.musicgen.ai

- [ ] **Troubleshooting Exercise**
  ```bash
  # Investigate simulated issue
  ./scripts/create_test_issue.sh
  
  # Use troubleshooting guide to resolve
  # Document steps taken
  ```

#### Day 4: Security and DR
- [ ] **Security Procedures**
  - Review security checklist
  - Run security scan
  - Practice incident response

- [ ] **Disaster Recovery**
  ```bash
  # Test backup restoration
  ./scripts/test_restore.sh test_backup_20240115
  
  # Verify recovery procedures
  ```

#### Day 5: Production Shadowing
- [ ] **Shadow Production Deployment**
  - Observe senior engineer deployment
  - Ask questions during process
  - Document observations

- [ ] **Documentation Feedback**
  - List unclear sections
  - Suggest improvements
  - Create documentation PRs

### Feedback Collection Template

```markdown
# Deployment Documentation Feedback

**Team Member**: [Name]
**Date**: [Date]
**Experience Level**: [Junior/Mid/Senior]

## Clarity Ratings (1-5)
- Infrastructure Requirements: [ ]
- Deployment Procedures: [ ]
- Troubleshooting Guide: [ ]
- Security Procedures: [ ]
- Disaster Recovery: [ ]

## Sections That Were Unclear
1. 
2. 
3. 

## Missing Information
1. 
2. 
3. 

## Suggested Improvements
1. 
2. 
3. 

## Commands That Failed
```bash
# List any commands that didn't work as expected
```

## Time Estimates
- Reading documentation: [ ] hours
- Setting up environment: [ ] hours
- First deployment: [ ] hours
- Understanding monitoring: [ ] hours

## Overall Assessment
[ ] Documentation is production-ready
[ ] Documentation needs minor updates
[ ] Documentation needs major revisions

## Additional Comments
```

---

## Continuous Improvement

### Documentation Maintenance

1. **Monthly Review**
   - Check for outdated versions
   - Update deprecated commands
   - Add new troubleshooting scenarios

2. **Quarterly Updates**
   - Review and update infrastructure requirements
   - Update security procedures
   - Refresh disaster recovery plans

3. **Annual Overhaul**
   - Complete documentation audit
   - Incorporate team feedback
   - Align with industry best practices

### Version Control

```bash
# Documentation versioning
git tag -a deployment-guide-v1.0.0 -m "Initial deployment guide"
git push origin deployment-guide-v1.0.0

# Track changes
git log --oneline docs/deployment/
```

---

**End of Deployment Guide**

*This document is maintained by the Platform Team. For questions or updates, please contact: platform-team@musicgen.ai*