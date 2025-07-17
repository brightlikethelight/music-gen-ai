# 🚀 Enterprise Resource Management System

**Music Gen AI Resource Management & Monitoring Guide**

---

## 📊 Overview

The Music Gen AI Resource Management System provides comprehensive GPU/memory monitoring and management to ensure stable production operation without resource failures. This system prevents OOM (Out of Memory) errors, tracks resource usage, and provides optimization recommendations.

---

## 🏗️ Architecture

### Core Components

1. **ResourceManager**: Central orchestrator for resource allocation and tracking
2. **ResourceMonitor**: Real-time monitoring of CPU/GPU resources
3. **ResourceOptimizer**: Provides optimization recommendations
4. **Integration**: Seamless integration with model loading and generation

### Key Features

- ✅ **Proactive Resource Validation** - Validates resources before model loading
- ✅ **Real-time Monitoring** - Continuous tracking of CPU/GPU usage
- ✅ **Automatic Cleanup** - Memory pressure detection and cleanup
- ✅ **Resource Alerts** - Configurable thresholds and notifications
- ✅ **API Monitoring** - REST endpoints for resource status
- ✅ **Performance Optimization** - Batch size and configuration recommendations

---

## 🎯 Resource Requirements by Model

### Predefined Model Requirements

| Model | CPU Memory | GPU Memory | Min Compute | Batch Size |
|-------|------------|------------|-------------|------------|
| `facebook/musicgen-small` | 3.0 GB | 6.0 GB | 6.0 | 2 |
| `facebook/musicgen-medium` | 6.0 GB | 12.0 GB | 7.0 | 1 |
| `facebook/musicgen-large` | 12.0 GB | 24.0 GB | 8.0 | 1 |
| Generic `small` | 2.0 GB | 4.0 GB | 6.0 | 4 |
| Generic `medium` | 4.0 GB | 8.0 GB | 7.0 | 2 |
| Generic `large` | 8.0 GB | 16.0 GB | 7.5 | 1 |

### Resource Validation

```python
# Automatic validation before model loading
resource_manager = ResourceManager(config)

# This will raise InsufficientResourcesError if resources are inadequate
validation_result = resource_manager.validate_system_resources("facebook/musicgen-medium")

# Results include warnings and specific requirements
if validation_result.get("warnings"):
    for warning in validation_result["warnings"]:
        logger.warning(f"Resource warning: {warning}")
```

---

## 🔍 Monitoring System

### Real-time Resource Tracking

The ResourceMonitor continuously tracks:

- **CPU Usage**: Percentage, memory usage, available memory
- **GPU Usage**: Memory allocation, utilization, temperature
- **Process Metrics**: Application-specific resource consumption
- **Historical Data**: 5-minute rolling history with statistics

### Alert System

Automatic alerts are triggered when:

- **CPU Memory > 80%**: Warning alert
- **CPU Memory > 90%**: Critical alert
- **GPU Memory > 90%**: Critical alert with cleanup recommendations
- **GPU Temperature > 85°C**: Temperature warning

### Example Monitoring Usage

```python
from music_gen.core.resource_manager import ResourceManager

# Initialize resource manager
resource_manager = ResourceManager(config)

# Get current snapshot
snapshot = resource_manager.monitor.get_current_snapshot()
print(f"CPU Memory: {snapshot.cpu_memory_percent:.1f}%")
print(f"GPU Memory: {snapshot.gpu_memory_percent:.1f}%")

# Get comprehensive report
report = resource_manager.get_resource_report()
print(f"System Health: {report['health_status']}")
```

---

## 📡 API Endpoints

### Resource Status

```bash
GET /api/v1/monitoring/status
```

**Response:**
```json
{
  "timestamp": "2024-12-15T10:30:00Z",
  "cpu": {
    "usage_percent": 45.2,
    "memory_used_gb": 8.4,
    "memory_available_gb": 23.6,
    "memory_percent": 26.2
  },
  "gpu": {
    "available": true,
    "memory_used_gb": 4.2,
    "memory_total_gb": 24.0,
    "memory_percent": 17.5,
    "utilization": 23.4,
    "temperature": 67.0
  },
  "process": {
    "memory_gb": 3.2,
    "gpu_memory_gb": 2.1
  }
}
```

### Comprehensive Report

```bash
GET /api/v1/monitoring/report
```

**Response:**
```json
{
  "timestamp": "2024-12-15T10:30:00Z",
  "current": { /* current status */ },
  "average_5min": {
    "cpu_percent": 42.1,
    "cpu_memory_percent": 25.8,
    "gpu_memory_percent": 18.2,
    "gpu_utilization": 21.7
  },
  "allocated_resources": {
    "model_load_facebook_musicgen_small_1734260000": {
      "cpu_memory_gb": 3.0,
      "gpu_memory_gb": 6.0,
      "timestamp": "2024-12-15T10:25:00Z"
    }
  },
  "cached_models": {
    "facebook/musicgen-small": [3.2, "2024-12-15T10:25:00Z"]
  },
  "alerts": {
    "total": 2,
    "critical": 0,
    "warnings": 2,
    "recent": [
      {
        "timestamp": "2024-12-15T10:28:00Z",
        "severity": "warning",
        "type": "cpu_memory",
        "message": "High CPU memory usage",
        "value": 82.1,
        "threshold": 80
      }
    ]
  },
  "health_status": "moderate"
}
```

### Resource Alerts

```bash
GET /api/v1/monitoring/alerts?severity=critical&limit=10
```

### Optimization Suggestions

```bash
GET /api/v1/monitoring/optimization-suggestions
```

**Response:**
```json
{
  "suggestions": [
    "High CPU memory usage detected:",
    "- Consider reducing model cache size",
    "- Enable model offloading to disk",
    "- Use smaller model variants"
  ],
  "optimal_batch_sizes": {
    "small": 4,
    "medium": 2,
    "large": 1
  },
  "current_health": "moderate"
}
```

### Manual Cleanup

```bash
POST /api/v1/monitoring/cleanup?force=false
```

**Response:**
```json
{
  "success": true,
  "cpu_memory_freed_gb": 1.2,
  "gpu_memory_freed_gb": 0.8,
  "message": "Cleanup completed. CPU: 1.2GB freed, GPU: 0.8GB freed"
}
```

### Resource History

```bash
GET /api/v1/monitoring/history?window_seconds=300
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Resource thresholds
MUSICGEN_MEMORY_PRESSURE_THRESHOLD=0.85  # 85% triggers cleanup
MUSICGEN_MONITORING_INTERVAL=1.0          # 1 second sampling
MUSICGEN_HISTORY_SIZE=300                 # 5 minutes of history

# Model cache settings
MUSICGEN_MODEL_CACHE_SIZE=5               # Max cached models
MUSICGEN_MODEL_CACHE_DIR=/tmp/models      # Cache directory
```

### Custom Resource Requirements

```python
from music_gen.core.resource_manager import ResourceRequirements

# Define custom model requirements
custom_requirements = {
    "my-custom-model": ResourceRequirements(
        cpu_memory_gb=6.0,
        gpu_memory_gb=12.0,
        min_gpu_compute=7.0,
        recommended_batch_size=1,
        notes="High-quality custom model"
    )
}

# Add to resource manager
resource_manager.MODEL_REQUIREMENTS.update(custom_requirements)
```

---

## 🛠️ Integration Examples

### Model Service Integration

```python
from music_gen.core.resource_manager import resource_monitored

class ModelService:
    @resource_monitored
    async def load_model(self, model_id: str):
        # Automatic resource validation
        validation_result = self.resource_manager.validate_system_resources(model_id)
        
        # Load model with resource tracking
        model = await self._load_model_impl(model_id)
        
        # Track in cache
        model_size = self._estimate_model_size(model)
        self.resource_manager.track_model_cache(model_id, model_size)
        
        return model
```

### Generation Service Integration

```python
class GenerationService:
    @resource_monitored
    async def generate(self, request: GenerationRequest):
        # Pre-validate resources
        try:
            self.resource_manager.validate_system_resources(request.model_id)
        except InsufficientResourcesError as e:
            raise GenerationError(f"Cannot start generation: {e}")
        
        # Proceed with generation
        return await self._generate_impl(request)
```

---

## 🎛️ Advanced Features

### Automatic Resource Cleanup

The system automatically triggers cleanup when:

1. **Memory Pressure**: CPU/GPU memory exceeds 85% threshold
2. **Model Cache Overflow**: Too many models in cache
3. **Alert Conditions**: Critical resource alerts

```python
# Emergency cleanup methods
resource_manager._emergency_cleanup()      # CPU memory cleanup
resource_manager._emergency_gpu_cleanup()  # GPU memory cleanup
```

### Resource Allocation Tracking

```python
# Allocate resources for an operation
allocation_id = "my_operation_123"
resource_manager.allocate_resources(
    allocation_id,
    cpu_memory_gb=2.0,
    gpu_memory_gb=4.0
)

# Release when done
resource_manager.release_resources(allocation_id)
```

### Performance Optimization

```python
from music_gen.core.resource_manager import ResourceOptimizer

# Get optimal batch size
optimal_batch = ResourceOptimizer.get_optimal_batch_size(
    model_size="medium",
    available_gpu_memory_gb=16.0
)

# Get optimization suggestions
suggestions = ResourceOptimizer.get_optimization_suggestions(resource_report)
```

---

## 🚨 Error Handling

### Common Exceptions

```python
from music_gen.core.exceptions import InsufficientResourcesError, ResourceExhaustionError

try:
    model = await model_service.load_model("facebook/musicgen-large")
except InsufficientResourcesError as e:
    # Handle insufficient resources
    logger.error(f"Cannot load model: {e}")
    # Suggest smaller model or cleanup
    
except ResourceExhaustionError as e:
    # Handle resource exhaustion during operation
    logger.error(f"Resources exhausted: {e}")
    # Trigger cleanup and retry
```

### Error Recovery

```python
# Automatic retry with cleanup
async def load_model_with_retry(model_id: str, max_retries: int = 2):
    for attempt in range(max_retries):
        try:
            return await model_service.load_model(model_id)
        except InsufficientResourcesError:
            if attempt < max_retries - 1:
                # Trigger cleanup and retry
                resource_manager._emergency_cleanup()
                await asyncio.sleep(1)
            else:
                raise
```

---

## 📈 Performance Monitoring

### Key Metrics to Track

1. **Resource Utilization**
   - Average CPU/GPU usage over time
   - Peak memory consumption
   - Resource efficiency ratios

2. **Model Performance**
   - Model loading times
   - Memory footprint per model
   - Cache hit rates

3. **Generation Performance**
   - Generation latency vs resource usage
   - Throughput optimization
   - Batch size efficiency

### Dashboard Integration

The monitoring system provides structured data perfect for dashboard integration:

```javascript
// Example dashboard data fetching
async function updateResourceDashboard() {
    const status = await fetch('/api/v1/monitoring/status').then(r => r.json());
    const report = await fetch('/api/v1/monitoring/report').then(r => r.json());
    
    // Update CPU gauge
    updateGauge('cpu-usage', status.cpu.usage_percent);
    
    // Update GPU gauge  
    updateGauge('gpu-memory', status.gpu.memory_percent);
    
    // Update health indicator
    updateHealthStatus(report.health_status);
    
    // Update alerts
    updateAlerts(report.alerts.recent);
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **High Memory Usage**
```bash
# Check current status
curl http://localhost:8000/api/v1/monitoring/status

# Get optimization suggestions
curl http://localhost:8000/api/v1/monitoring/optimization-suggestions

# Trigger cleanup
curl -X POST http://localhost:8000/api/v1/monitoring/cleanup
```

#### 2. **Model Loading Failures**
```bash
# Check resource requirements for model
python -c "
from music_gen.core.resource_manager import ResourceManager
rm = ResourceManager(config)
reqs = rm.get_model_requirements('facebook/musicgen-large')
print(f'Required: CPU {reqs.cpu_memory_gb}GB, GPU {reqs.gpu_memory_gb}GB')
"
```

#### 3. **Performance Issues**
```bash
# Get resource history
curl "http://localhost:8000/api/v1/monitoring/history?window_seconds=600"

# Check for resource alerts
curl "http://localhost:8000/api/v1/monitoring/alerts?severity=warning"
```

### Debug Commands

```python
# Enable debug logging
import logging
logging.getLogger('music_gen.core.resource_manager').setLevel(logging.DEBUG)

# Get detailed resource snapshot
snapshot = resource_manager.monitor.get_current_snapshot()
print(f"Detailed snapshot: {snapshot}")

# Check allocated resources
print(f"Allocated: {resource_manager._allocated_resources}")

# Check model cache tracking
print(f"Cached models: {resource_manager._model_cache_tracker}")
```

---

## ✅ Production Deployment Checklist

### Pre-deployment

- [ ] **Resource Requirements Documented**: All model requirements defined
- [ ] **Monitoring Configured**: Alerts and thresholds set appropriately
- [ ] **Dashboard Setup**: Resource monitoring dashboard deployed
- [ ] **Cleanup Procedures**: Automatic cleanup policies configured

### Deployment

- [ ] **Resource Validation**: System passes resource validation checks
- [ ] **Monitoring Active**: Resource monitor running with alerts
- [ ] **API Endpoints**: All monitoring endpoints responding correctly
- [ ] **Cleanup Testing**: Manual and automatic cleanup tested

### Post-deployment

- [ ] **Health Monitoring**: System health status monitored continuously
- [ ] **Performance Tracking**: Resource usage patterns analyzed
- [ ] **Alert Response**: Alert handling procedures in place
- [ ] **Optimization**: Performance optimization recommendations implemented

---

## 📚 Additional Resources

- **API Documentation**: `/docs` endpoint for complete API reference
- **Performance Tuning**: See `docs/performance_optimization.md`
- **Monitoring Integration**: See `docs/monitoring_integration.md`
- **Troubleshooting**: See `docs/troubleshooting.md`

---

**Status: ✅ Production Ready**

The Enterprise Resource Management System provides comprehensive resource monitoring and management suitable for commercial deployment. All core functionality is implemented with proper error handling, monitoring, and optimization features.