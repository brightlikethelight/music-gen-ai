# Kubernetes Deployment for MusicGen AI

This directory contains production-grade Kubernetes manifests for deploying MusicGen AI.

## Prerequisites

- Kubernetes cluster (1.21+)
- kubectl configured
- NVIDIA GPU operator (for GPU nodes)
- Ingress controller (nginx recommended)
- cert-manager (for SSL)

## Quick Start

```bash
# Create namespace
kubectl create namespace musicgen

# Apply all manifests
kubectl apply -f .

# Check deployment status
kubectl get pods -n musicgen
kubectl get svc -n musicgen
```

## Files Overview

### Core Application
- `deployment.yaml` - Main MusicGen API deployment with GPU support
- `service.yaml` - Load balancing and service discovery
- `pvc.yaml` - Persistent storage for models and outputs

### Infrastructure
- `ingress.yaml` - SSL termination and routing
- `autoscaling.yaml` - Horizontal Pod Autoscaler (HPA)
- `monitoring.yaml` - Prometheus and Grafana configuration
- `security.yaml` - RBAC, NetworkPolicies, and secrets

## Configuration

### GPU Support
The deployment is configured to use NVIDIA GPUs:
```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

### Auto-scaling
HPA scales based on:
- CPU utilization (70%)
- Memory utilization (80%)
- Active generation requests (5 avg)

### Monitoring
Access monitoring dashboards:
- Prometheus: `https://monitoring.musicgen.example.com/prometheus`
- Grafana: `https://monitoring.musicgen.example.com/grafana`

### Storage
Persistent volumes are used for:
- Model storage: 50Gi
- Generated outputs: 100Gi
- Prometheus data: 20Gi

## Security

- NetworkPolicies restrict pod-to-pod communication
- RBAC limits service account permissions
- Secrets store API keys and credentials
- Basic auth protects monitoring endpoints

## Production Checklist

- [ ] Update domain names in `ingress.yaml`
- [ ] Configure real SSL certificates
- [ ] Set resource limits based on load testing
- [ ] Update storage class names
- [ ] Configure backup strategy
- [ ] Set up log aggregation
- [ ] Configure alerting rules

## Troubleshooting

```bash
# Check pod logs
kubectl logs -f deployment/musicgen-api -n musicgen

# Describe pod for events
kubectl describe pod musicgen-api-xxx -n musicgen

# Check GPU availability
kubectl get nodes -o json | jq '.items[].status.allocatable'

# Test service connectivity
kubectl port-forward svc/musicgen-api-service 8000:8000 -n musicgen
```