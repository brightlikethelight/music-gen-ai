# Structured Logging Implementation Guide

## Overview

This document describes the comprehensive structured logging system implemented for Music Gen AI, following 2024 best practices for production observability, distributed tracing, and log aggregation.

## Features

- **JSON Structured Logging**: All logs are formatted as JSON for easy parsing and analysis
- **Correlation IDs**: Distributed tracing support with automatic correlation ID generation and propagation
- **Performance Logging**: Detailed request performance metrics with automatic categorization
- **Audit Logging**: Security-sensitive operations tracking with compliance features
- **Log Aggregation**: Support for ELK Stack, Fluentd, and OpenTelemetry Collector
- **Automatic Rotation**: Configurable log rotation and retention policies
- **Health Monitoring**: Real-time log system health monitoring and alerting
- **Environment-Aware**: Different configurations for development, staging, and production

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │    │   Middleware     │    │   Log Files     │
│                 │    │                  │    │                 │
│ • FastAPI       │────┤ • Correlation ID │────┤ • app.log       │
│ • Business      │    │ • Performance    │    │ • audit.log     │
│   Logic         │    │ • Audit          │    │ • performance.  │
│ • Error         │    │ • Security       │    │ • error.log     │
│   Handling      │    │   Headers        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                       ▼                                 ▼                                 ▼
              ┌─────────────────┐              ┌─────────────────┐               ┌─────────────────┐
              │   ELK Stack     │              │    Fluentd      │               │ OpenTelemetry   │
              │                 │              │                 │               │   Collector     │
              │ • Elasticsearch │              │ • Log           │               │                 │
              │ • Logstash      │              │   Processing    │               │ • Unified       │
              │ • Kibana        │              │ • Aggregation   │               │   Observability │
              │ • Filebeat      │              │ • Forwarding    │               │ • Trace         │
              └─────────────────┘              └─────────────────┘               │   Correlation   │
                                                                                 └─────────────────┘
```

## Components

### 1. Core Logging Configuration (`music_gen/core/logging_config.py`)

**Purpose**: Central logging configuration with JSON formatting, correlation IDs, and structured processors.

**Key Features**:
- Environment-specific log levels
- JSON formatting with custom fields
- Correlation ID injection
- Sensitive data filtering
- Multiple log handlers (file, console, external)

**Configuration**:
```python
# Environment Variables
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR=/var/log/musicgen        # Log directory path
ENVIRONMENT=production            # development, staging, production
SERVICE_NAME=musicgen-api         # Service identifier
SERVICE_VERSION=1.0.0            # Version for tracking
```

### 2. Correlation ID Middleware (`music_gen/api/middleware/correlation_id.py`)

**Purpose**: Ensures every request has a unique correlation ID for distributed tracing.

**Features**:
- Automatic correlation ID generation (UUID4)
- Header propagation (`x-correlation-id`)
- Request state injection
- Thread-local storage support

**Usage**:
```python
# Automatic generation
GET /api/health
Response Headers: x-correlation-id: 550e8400-e29b-41d4-a716-446655440000

# Client-provided correlation ID
GET /api/health
Request Headers: x-correlation-id: custom-trace-123
Response Headers: x-correlation-id: custom-trace-123
```

### 3. Performance Logging Middleware (`music_gen/api/middleware/performance_logging.py`)

**Purpose**: Captures detailed performance metrics for all HTTP requests.

**Metrics Captured**:
- Request duration (milliseconds)
- Memory usage (MB)
- Database query count and duration
- Response size
- HTTP method, path, status code
- User context (if available)

**Performance Tiers**:
- **Fast**: < 100ms
- **Normal**: 100ms - 1s
- **Slow**: 1s - 5s
- **Very Slow**: > 5s (triggers alerts)

**Configuration**:
```bash
LOG_SLOW_REQUESTS_ONLY=false         # Only log slow requests
SLOW_REQUEST_THRESHOLD_MS=1000       # Threshold for slow requests
INCLUDE_MEMORY_METRICS=true          # Include memory usage
INCLUDE_DATABASE_METRICS=true        # Include DB metrics
```

### 4. Audit Logging Middleware (`music_gen/api/middleware/audit_logging.py`)

**Purpose**: Tracks security-sensitive operations for compliance and security monitoring.

**Events Tracked**:
- Authentication attempts (login, logout, registration)
- Authorization failures
- Data access (create, read, update, delete)
- Administrative actions
- API key usage
- Failed security validations

**Sensitive Data Filtering**:
- Passwords: `[REDACTED]`
- API Keys: `sk-1234...cdef` (partial masking)
- Credit Cards: `[REDACTED]`
- JWT Tokens: `eyJ0...xyz` (partial masking)

**Geographic Enrichment**:
- GeoIP lookup for external IP addresses
- Country, region, city information
- Security risk scoring

### 5. Log Management API (`music_gen/api/routes/logging.py`)

**Purpose**: RESTful API for log monitoring, statistics, and management.

**Endpoints**:
- `GET /api/v1/logs/health` - Log system health status
- `GET /api/v1/logs/stats` - Log statistics and metrics
- `GET /api/v1/logs/recent` - Recent log entries with filtering
- `GET /api/v1/logs/performance` - Performance metrics analysis
- `GET /api/v1/logs/audit` - Audit log entries for security monitoring
- `POST /api/v1/logs/test` - Test logging functionality
- `GET /api/v1/logs/config` - Current logging configuration

## Log Aggregation

### ELK Stack Integration

**Components**:
- **Elasticsearch**: Log storage and indexing
- **Logstash**: Log processing and enrichment
- **Kibana**: Visualization and dashboards
- **Filebeat**: Log shipping

**Configuration Files**:
- `configs/logging/elasticsearch.yml` - Elasticsearch settings
- `configs/logging/logstash.conf` - Log processing pipeline
- `configs/logging/elasticsearch-template.json` - Index mapping

**Index Strategy**:
```
musicgen-application-YYYY.MM.DD
musicgen-audit-YYYY.MM.DD
musicgen-performance-YYYY.MM.DD
musicgen-error-YYYY.MM.DD
musicgen-security-YYYY.MM.DD
```

**Sample Queries**:
```json
# Find all errors in the last hour
{
  "query": {
    "bool": {
      "must": [
        {"term": {"log.level": "ERROR"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  }
}

# Trace a request by correlation ID
{
  "query": {
    "term": {"trace.correlation_id": "550e8400-e29b-41d4-a716-446655440000"}
  }
}
```

### Fluentd Integration

**Features**:
- High-throughput log processing
- Real-time log aggregation
- Multiple output destinations
- Buffer management for reliability

**Configuration**: `configs/logging/fluentd.conf`

**Key Processors**:
- JSON parsing and validation
- Field normalization and enrichment
- Performance classification
- Security event detection
- OpenTelemetry correlation

### OpenTelemetry Collector

**Purpose**: Modern unified observability with traces, metrics, and logs.

**Configuration**: `configs/logging/otel-collector.yaml`

**Features**:
- Automatic trace correlation
- Multi-format log parsing
- Resource attribute injection
- Batch processing for performance
- Multiple exporter support

**Exporters Supported**:
- Elasticsearch
- Jaeger (traces)
- Prometheus (metrics)
- Datadog
- Custom webhooks

## Log Rotation and Retention

### Logrotate Configuration

**File**: `configs/logging/logrotate.conf`

**Policies**:
- **Application Logs**: Daily rotation, 30 days retention, 100MB max size
- **Audit Logs**: Daily rotation, 365 days retention (compliance), 50MB max size
- **Performance Logs**: Daily rotation, 30 days retention, 200MB max size
- **Error Logs**: Daily rotation, 90 days retention, 50MB max size

**Compliance Features**:
- Audit logs kept for 7 years (2555 days)
- Automatic archival to long-term storage
- Integrity verification (checksums)
- Compliance reporting webhooks

### Automated Log Management

**Script**: `scripts/musicgen-log-manager.sh`

**Features**:
- Disk usage monitoring (85% threshold)
- Old log cleanup
- Archive management
- Health monitoring
- Activity detection
- Statistics generation

**Systemd Integration**: `configs/logging/systemd-log-management.service`
- Hourly execution via timer
- Resource limits and security hardening
- Automatic restart on failure

## Environment Configuration

### Development Environment

```bash
# Basic logging for development
LOG_LEVEL=DEBUG
LOG_DIR=./logs
ENVIRONMENT=development
AUDIT_LOG_ALL_REQUESTS=false
LOG_SLOW_REQUESTS_ONLY=false
INCLUDE_MEMORY_METRICS=true
```

### Staging Environment

```bash
# Enhanced logging for testing
LOG_LEVEL=INFO
LOG_DIR=/var/log/musicgen
ENVIRONMENT=staging
AUDIT_LOG_ALL_REQUESTS=true
LOG_SLOW_REQUESTS_ONLY=false
INCLUDE_MEMORY_METRICS=true
INCLUDE_DATABASE_METRICS=true
ELK_HOST=staging-elk.internal
```

### Production Environment

```bash
# Production-optimized logging
LOG_LEVEL=INFO
LOG_DIR=/var/log/musicgen
ENVIRONMENT=production
AUDIT_LOG_ALL_REQUESTS=false
LOG_SLOW_REQUESTS_ONLY=true
SLOW_REQUEST_THRESHOLD_MS=1000
INCLUDE_MEMORY_METRICS=false
INCLUDE_DATABASE_METRICS=true

# Aggregation endpoints
ELASTICSEARCH_HOST=prod-elasticsearch.internal
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USER=musicgen_logger
ELASTICSEARCH_PASSWORD=secure_password

# Monitoring webhooks
MONITORING_WEBHOOK=https://monitoring.internal/webhooks/logs
ALERT_WEBHOOK=https://alerts.internal/webhooks/critical
ERROR_WEBHOOK=https://ops.internal/webhooks/errors

# Archive configuration
ARCHIVE_DIR=/mnt/log-archive/musicgen
MAX_LOG_AGE_DAYS=90
MAX_ARCHIVE_AGE_DAYS=2555
DISK_USAGE_THRESHOLD=85
```

## Security Considerations

### Data Protection

1. **Sensitive Data Filtering**:
   - Automatic PII detection and redaction
   - Configurable sensitive field patterns
   - Partial masking for identifiers

2. **Access Control**:
   - Log file permissions (644)
   - Service user isolation
   - API endpoint authentication

3. **Encryption**:
   - TLS for log aggregation endpoints
   - At-rest encryption for archived logs
   - Secure webhook communications

### Compliance Features

1. **Audit Trails**:
   - Immutable audit logs
   - Chronological ordering
   - Digital signatures (configurable)

2. **Retention Policies**:
   - Legal hold capabilities
   - Automated deletion schedules
   - Export for legal discovery

3. **Privacy Controls**:
   - GDPR compliance features
   - User data anonymization
   - Consent tracking

## Monitoring and Alerting

### Health Metrics

1. **System Health**:
   - Log file accessibility
   - Disk usage monitoring
   - Write performance tracking

2. **Application Health**:
   - Error rate monitoring
   - Performance degradation detection
   - Security event tracking

3. **Infrastructure Health**:
   - Aggregation pipeline status
   - Storage system monitoring
   - Network connectivity checks

### Alert Conditions

1. **Critical Alerts**:
   - Disk usage > 85%
   - Multiple authentication failures
   - System errors > 10/minute
   - Log aggregation failures

2. **Warning Alerts**:
   - Slow request rate increase
   - Disk usage > 75%
   - Log file growth anomalies
   - Archive failures

3. **Info Alerts**:
   - Log rotation events
   - Archive completion
   - Statistics updates

## Performance Optimization

### High-Throughput Configuration

1. **Buffering**:
   - Memory buffers for high-frequency logs
   - Async log writing
   - Batch processing

2. **Compression**:
   - Gzip compression for archives
   - Real-time compression for aggregation
   - Efficient JSON formatting

3. **Indexing**:
   - Optimized Elasticsearch mappings
   - Proper field types and analyzers
   - Index lifecycle management

### Resource Management

1. **Memory Usage**:
   - Bounded log buffers
   - Memory-mapped file access
   - Garbage collection optimization

2. **CPU Usage**:
   - Efficient JSON serialization
   - Minimal processing overhead
   - Async I/O operations

3. **Storage Usage**:
   - Intelligent log rotation
   - Compression ratios optimization
   - Predictive capacity planning

## Troubleshooting

### Common Issues

1. **Log Files Not Created**:
   ```bash
   # Check permissions
   ls -la /var/log/musicgen/
   
   # Check service status
   systemctl status musicgen-api
   
   # Check log configuration
   curl http://localhost:8000/api/v1/logs/config
   ```

2. **Correlation IDs Missing**:
   ```bash
   # Check middleware configuration
   curl -H "x-correlation-id: test-123" http://localhost:8000/health
   
   # Verify response headers
   curl -I http://localhost:8000/health
   ```

3. **High Disk Usage**:
   ```bash
   # Check disk usage
   df -h /var/log/musicgen
   
   # Run manual cleanup
   /usr/local/bin/musicgen-log-manager
   
   # Check rotation configuration
   logrotate -d /etc/logrotate.d/musicgen
   ```

4. **Aggregation Failures**:
   ```bash
   # Check Elasticsearch connectivity
   curl http://elasticsearch:9200/_cluster/health
   
   # Check Logstash pipeline
   systemctl status logstash
   tail -f /var/log/logstash/logstash.log
   
   # Check Fluentd status
   systemctl status fluentd
   tail -f /var/log/fluentd/fluentd.log
   ```

### Debugging Commands

```bash
# Test logging functionality
curl -X POST http://localhost:8000/api/v1/logs/test

# Get recent logs
curl "http://localhost:8000/api/v1/logs/recent?limit=10&level=error"

# Check log statistics
curl http://localhost:8000/api/v1/logs/stats

# Monitor log health
curl http://localhost:8000/api/v1/logs/health

# Check performance metrics
curl http://localhost:8000/api/v1/logs/performance?hours=1
```

## Testing

### Integration Tests

Run the comprehensive test suite:
```bash
pytest tests/integration/test_structured_logging.py -v
```

**Test Coverage**:
- Logging initialization
- Correlation ID generation and propagation
- Performance logging middleware
- Audit logging for security events
- Sensitive data filtering
- JSON log format validation
- Log level filtering
- Concurrent logging performance
- Log management API endpoints

### Performance Tests

```bash
# Test logging performance under load
python -m pytest tests/integration/test_structured_logging.py::TestStructuredLogging::test_async_logging_performance -v

# Test concurrent logging
python -m pytest tests/integration/test_structured_logging.py::TestStructuredLogging::test_concurrent_logging -v
```

### Manual Testing

```bash
# Generate test logs
curl -X POST http://localhost:8000/api/v1/logs/test

# Verify log format
tail -f /var/log/musicgen/app.log | jq .

# Test correlation ID propagation
CORRELATION_ID=$(uuidgen)
curl -H "x-correlation-id: $CORRELATION_ID" http://localhost:8000/health
grep "$CORRELATION_ID" /var/log/musicgen/app.log
```

## Migration Guide

### From Basic Logging

1. **Install Dependencies**:
   ```bash
   pip install structlog uvloop
   ```

2. **Update Configuration**:
   ```python
   # Replace basic logging configuration
   from music_gen.core.logging_config import setup_logging
   setup_logging()
   ```

3. **Update Log Calls**:
   ```python
   # Old
   import logging
   logger = logging.getLogger(__name__)
   logger.info("User action", extra={"user_id": user_id})
   
   # New
   from music_gen.core.logging_config import get_logger
   logger = get_logger(__name__)
   logger.info("User action", user_id=user_id, correlation_id=correlation_id)
   ```

### From JSON Logging

1. **Add Correlation IDs**:
   ```python
   # Update middleware stack
   app.add_middleware(CorrelationIdMiddleware)
   ```

2. **Add Performance Tracking**:
   ```python
   # Add performance middleware
   app.add_middleware(PerformanceLoggingMiddleware)
   ```

3. **Configure Aggregation**:
   - Set up ELK Stack or Fluentd
   - Configure index templates
   - Set up dashboards

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**:
   - Anomaly detection in log patterns
   - Predictive error analysis
   - Intelligent alerting

2. **Advanced Correlation**:
   - Cross-service tracing
   - Business transaction tracking
   - User journey analysis

3. **Enhanced Security**:
   - Threat detection patterns
   - Behavioral analysis
   - Automated response triggers

4. **Performance Optimization**:
   - Edge caching for logs
   - Distributed log processing
   - Real-time analytics

### Research Areas

1. **Log Data Science**:
   - Pattern recognition algorithms
   - Performance prediction models
   - User behavior analytics

2. **Distributed Systems**:
   - Multi-region log aggregation
   - Consensus-based log ordering
   - Fault-tolerant log pipelines

3. **Privacy Engineering**:
   - Differential privacy techniques
   - Homomorphic encryption for logs
   - Zero-knowledge audit proofs

## Support and Resources

### Documentation
- [FastAPI Middleware Documentation](https://fastapi.tiangolo.com/tutorial/middleware/)
- [Structlog Documentation](https://www.structlog.org/)
- [OpenTelemetry Logging](https://opentelemetry.io/docs/specs/otel/logs/)
- [ELK Stack Documentation](https://www.elastic.co/guide/)

### Community
- [Music Gen AI Slack Channel](https://musicgen.slack.com/channels/logging)
- [GitHub Issues](https://github.com/musicgen-ai/musicgen/issues)
- [Discussion Forum](https://discuss.musicgen.ai/c/logging)

### Contact
- Technical Lead: engineering-leads@musicgen.ai
- DevOps Team: devops@musicgen.ai
- Security Team: security@musicgen.ai

---

*Last Updated: $(date -Iseconds)*
*Version: 1.0.0*
*Environment: Production*