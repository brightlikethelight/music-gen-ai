# Deployment Execution Checklist

**Date**: ________________  
**Deployment Version**: ________________  
**Deployment Start Time**: ________________  
**Primary Deployer**: ________________  
**Secondary Deployer**: ________________

## Phase 1: Pre-Deployment (T-30 minutes)

### Final Preparations
- [ ] Pre-deployment checklist completed and approved
- [ ] All team members in position
  - [ ] Primary deployer ready
  - [ ] Secondary deployer ready
  - [ ] On-call engineer available
  - [ ] Database admin on standby

- [ ] Communication channels open
  ```bash
  # Join deployment channel
  slack-cli join #deployment-live
  ```

- [ ] Monitoring dashboards open
  - [ ] Grafana: https://grafana.musicgen.ai/d/deployment
  - [ ] Prometheus: https://prometheus.musicgen.ai
  - [ ] Application logs: https://logs.musicgen.ai
  - [ ] Error tracking: https://sentry.musicgen.ai

## Phase 2: Deployment Initiation (T-0)

### 1. Enable Deployment Mode
- [ ] Enable maintenance mode (if required)
  ```bash
  kubectl create configmap maintenance-mode --from-literal=enabled=true
  ```

- [ ] Start deployment logging
  ```bash
  # Start deployment recorder
  script -f deployment_$(date +%Y%m%d_%H%M%S).log
  ```

- [ ] Record deployment start
  ```bash
  # Log deployment start
  echo "DEPLOYMENT START: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a deployment.log
  ./scripts/notify_deployment_start.sh v1.0.0
  ```

### 2. Database Updates
- [ ] Create database backup point
  ```bash
  # Create named backup
  pg_dump -h postgres.musicgen.ai -U musicgen -d musicgen_prod \
    --format=custom --file=pre_deployment_$(date +%Y%m%d_%H%M%S).dump
  ```

- [ ] Run database migrations
  ```bash
  # Apply migrations
  cd /app && alembic upgrade head
  
  # Verify migrations
  alembic current
  ```

- [ ] Verify database integrity
  ```sql
  -- Check constraints
  SELECT conname FROM pg_constraint WHERE NOT convalidated;
  
  -- Check indexes
  SELECT indexrelid::regclass FROM pg_index WHERE NOT indisvalid;
  ```

## Phase 3: Application Deployment

### Blue-Green Deployment (Option A)
- [ ] Deploy to inactive environment
  ```bash
  # Identify inactive environment
  CURRENT_ENV=$(aws ssm get-parameter --name /musicgen/active-env --query 'Parameter.Value' --output text)
  NEW_ENV=$([[ "$CURRENT_ENV" == "blue" ]] && echo "green" || echo "blue")
  
  echo "Deploying to ${NEW_ENV} environment"
  ```

- [ ] Deploy new version
  ```bash
  # Deploy application
  kubectl set image deployment/musicgen-api-${NEW_ENV} \
    api=musicgen/api:v1.0.0 \
    --namespace=${NEW_ENV}
  
  # Wait for rollout
  kubectl rollout status deployment/musicgen-api-${NEW_ENV} \
    --namespace=${NEW_ENV} --timeout=600s
  ```

- [ ] Health check new environment
  ```bash
  # Check health endpoint
  curl -f https://${NEW_ENV}.musicgen.ai/health || exit 1
  ```

- [ ] Run smoke tests
  ```bash
  # Run smoke test suite
  pytest tests/smoke/ --env=${NEW_ENV} -v
  ```

- [ ] Switch traffic (if tests pass)
  ```bash
  # Update load balancer
  ./scripts/switch_traffic.sh ${CURRENT_ENV} ${NEW_ENV}
  
  # Update active environment
  aws ssm put-parameter --name /musicgen/active-env --value ${NEW_ENV} --overwrite
  ```

### Rolling Deployment (Option B)
- [ ] Update first instance
  ```bash
  # Scale down to trigger update
  kubectl scale deployment musicgen-api --replicas=2
  
  # Update image
  kubectl set image deployment/musicgen-api api=musicgen/api:v1.0.0
  
  # Watch rollout
  kubectl rollout status deployment/musicgen-api --watch
  ```

- [ ] Monitor metrics during rollout
  ```bash
  # Watch error rates
  watch -n 5 'curl -s http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])'
  ```

- [ ] Pause if issues detected
  ```bash
  # Pause rollout if needed
  kubectl rollout pause deployment/musicgen-api
  
  # Resume when ready
  kubectl rollout resume deployment/musicgen-api
  ```

## Phase 4: Service Updates

### 1. Worker Services
- [ ] Update Celery workers
  ```bash
  # Update worker deployment
  kubectl set image deployment/musicgen-worker worker=musicgen/worker:v1.0.0
  
  # Verify workers healthy
  celery -A music_gen inspect active
  ```

- [ ] Clear stale tasks
  ```bash
  # Purge old tasks
  celery -A music_gen purge -f
  ```

### 2. Cache Management
- [ ] Clear application caches
  ```bash
  # Clear Redis cache
  redis-cli -h redis.musicgen.ai FLUSHDB
  
  # Warm up critical caches
  ./scripts/warm_cache.sh
  ```

- [ ] Update CDN
  ```bash
  # Invalidate CDN cache
  aws cloudfront create-invalidation \
    --distribution-id E1234567890ABC \
    --paths "/*"
  ```

### 3. Model Updates
- [ ] Deploy new models (if applicable)
  ```bash
  # Sync models to production
  aws s3 sync s3://musicgen-models/v1.0.0/ /models/ --delete
  
  # Verify model loading
  python scripts/verify_models.py --production
  ```

## Phase 5: Verification

### 1. Functional Verification
- [ ] API endpoints responding
  ```bash
  # Test critical endpoints
  ./scripts/test_endpoints.sh production
  ```

- [ ] Authentication working
  ```bash
  # Test auth flow
  ./scripts/test_auth.sh production
  ```

- [ ] Generation pipeline functional
  ```bash
  # Test generation
  curl -X POST https://api.musicgen.ai/v1/generate \
    -H "X-API-Key: $TEST_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Deployment test", "duration": 5}'
  ```

### 2. Performance Verification
- [ ] Response times normal
  ```bash
  # Check p95 latency
  curl -s http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))
  ```

- [ ] No memory leaks
  ```bash
  # Check memory usage
  kubectl top pods -l app=musicgen-api
  ```

- [ ] CPU usage stable
  ```bash
  # Monitor CPU
  watch -n 10 'kubectl top nodes'
  ```

### 3. Integration Verification
- [ ] Database connections stable
  ```sql
  -- Check connection count
  SELECT count(*) FROM pg_stat_activity WHERE datname = 'musicgen_prod';
  ```

- [ ] External services connected
  ```bash
  # Check service health
  ./scripts/check_external_services.sh
  ```

- [ ] Monitoring/alerting working
  ```bash
  # Send test metric
  echo "deployment_test_metric 1" | nc -w 1 -u prometheus-pushgateway 9091
  ```

## Phase 6: Post-Deployment

### 1. Cleanup
- [ ] Remove old deployments
  ```bash
  # Clean up old versions
  kubectl delete deployment musicgen-api-${CURRENT_ENV} --namespace=${CURRENT_ENV}
  
  # Prune old images
  docker image prune -a --filter "until=24h"
  ```

- [ ] Archive deployment logs
  ```bash
  # Save deployment logs
  kubectl logs -l app=musicgen-api --since=1h > deployment_logs_$(date +%Y%m%d_%H%M%S).txt
  
  # Upload to S3
  aws s3 cp deployment_logs_*.txt s3://musicgen-deployments/logs/
  ```

### 2. Monitoring Period
- [ ] Monitor for 30 minutes
  - [ ] Error rates normal
  - [ ] No new alerts
  - [ ] Performance stable
  - [ ] No customer complaints

- [ ] Document any issues
  ```bash
  # Create issue report
  cat > deployment_issues.md << EOF
  # Deployment Issues - $(date)
  
  ## Issues Encountered
  
  ## Resolution Steps
  
  ## Follow-up Actions
  EOF
  ```

### 3. Communication
- [ ] Send deployment complete notification
  ```bash
  ./scripts/notify_deployment_complete.sh v1.0.0 SUCCESS
  ```

- [ ] Update status page
  ```bash
  # Update status
  curl -X POST https://status.musicgen.ai/api/v1/incidents \
    -H "Authorization: Bearer $STATUS_PAGE_TOKEN" \
    -d '{"status": "resolved", "message": "Deployment completed successfully"}'
  ```

- [ ] Document deployment
  ```bash
  # Create deployment record
  cat > deployments/$(date +%Y%m%d)_v1.0.0.md << EOF
  # Deployment Record
  
  **Version**: v1.0.0
  **Date**: $(date)
  **Duration**: ${DEPLOYMENT_DURATION}
  **Deployer**: ${DEPLOYER}
  **Status**: SUCCESS
  
  ## Changes
  - Feature: New audio generation model
  - Fix: Memory leak in worker processes
  - Improvement: API response time optimization
  
  ## Metrics
  - Downtime: 0 seconds
  - Rollout duration: X minutes
  - Error spike: None
  
  ## Notes
  [Any special notes or observations]
  EOF
  ```

## Rollback Decision Points

### Automatic Rollback Triggers
- [ ] Error rate >5% for 5 minutes
- [ ] Response time >2s p95 for 5 minutes  
- [ ] Health check failures >50%
- [ ] Database connection errors
- [ ] Out of memory errors

### Manual Rollback Decision
- [ ] Evaluate severity of issues
- [ ] Consider partial rollback options
- [ ] Get stakeholder input if needed
- [ ] Execute rollback if necessary
  ```bash
  # Initiate rollback
  ./scripts/emergency_rollback.sh
  ```

## Deployment Completion

**Deployment End Time**: ________________  
**Total Duration**: ________________  

### Final Status
- [ ] ✅ **SUCCESS** - Deployment completed successfully
- [ ] ⚠️ **SUCCESS WITH ISSUES** - Deployed but monitoring closely
- [ ] ❌ **ROLLED BACK** - Deployment failed and rolled back

### Metrics Summary
- **API Availability**: _____%
- **Peak Error Rate**: _____%
- **Average Response Time**: ____ms
- **Deployment Downtime**: ____seconds

### Team Sign-offs
- [ ] Primary Deployer
  - Name: ________________
  - Time: ________________

- [ ] Secondary Deployer
  - Name: ________________
  - Time: ________________

- [ ] Operations Lead
  - Name: ________________
  - Time: ________________

---

**Post-Deployment Actions Required**:
```
[List any follow-up actions needed]
```

**Lessons Learned**:
```
[Document any insights for future deployments]
```