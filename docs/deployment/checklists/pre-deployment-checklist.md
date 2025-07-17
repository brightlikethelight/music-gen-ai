# Pre-Deployment Checklist

**Date**: ________________  
**Deployment Version**: ________________  
**Deployer**: ________________  
**Reviewer**: ________________

## 1. Code Readiness

### Code Quality
- [ ] All tests passing in CI/CD pipeline
  ```bash
  # Verify CI status
  gh run list --branch main --limit 5
  ```

- [ ] Code coverage meets threshold (>80%)
  ```bash
  # Check coverage report
  pytest --cov=music_gen --cov-report=term-missing
  ```

- [ ] No critical security vulnerabilities
  ```bash
  # Run security scan
  bandit -r music_gen/
  safety check
  ```

- [ ] Code reviewed and approved by 2+ team members
  ```bash
  # Check PR approvals
  gh pr view --json reviews
  ```

### Version Control
- [ ] All changes committed and pushed
  ```bash
  git status
  git log --oneline -10
  ```

- [ ] Release branch created and protected
  ```bash
  git checkout -b release/v1.0.0
  git push -u origin release/v1.0.0
  ```

- [ ] Version tags created
  ```bash
  git tag -a v1.0.0 -m "Release version 1.0.0"
  git push origin v1.0.0
  ```

## 2. Infrastructure Verification

### Resource Availability
- [ ] CPU capacity verified (>30% available)
  ```bash
  # Check cluster resources
  kubectl top nodes
  ```

- [ ] Memory capacity verified (>4GB per node available)
  ```bash
  # Check memory usage
  free -h
  ```

- [ ] GPU availability confirmed
  ```bash
  # Check GPU status
  nvidia-smi
  ```

- [ ] Disk space verified (>100GB available)
  ```bash
  # Check disk usage
  df -h
  ```

### Network Configuration
- [ ] Load balancer health checks passing
  ```bash
  # Test load balancer
  curl -I https://api.musicgen.ai/health
  ```

- [ ] SSL certificates valid (>30 days)
  ```bash
  # Check certificate expiration
  echo | openssl s_client -servername api.musicgen.ai -connect api.musicgen.ai:443 2>/dev/null | openssl x509 -noout -dates
  ```

- [ ] DNS resolution working
  ```bash
  # Test DNS
  nslookup api.musicgen.ai
  dig api.musicgen.ai
  ```

## 3. Database Preparation

### Backup Verification
- [ ] Recent backup exists (<24 hours old)
  ```bash
  # List recent backups
  aws s3 ls s3://musicgen-backups/ --recursive | grep $(date +%Y%m%d)
  ```

- [ ] Backup restoration tested
  ```bash
  # Test restore to staging
  ./scripts/test_backup_restore.sh
  ```

### Migration Readiness
- [ ] Database migrations reviewed
  ```bash
  # Check pending migrations
  alembic history
  alembic current
  ```

- [ ] Migration rollback scripts prepared
  ```sql
  -- Verify rollback scripts exist
  SELECT version_number, rollback_script IS NOT NULL as has_rollback 
  FROM schema_versions 
  ORDER BY version_id DESC 
  LIMIT 5;
  ```

- [ ] Database performance baseline recorded
  ```sql
  -- Capture current performance metrics
  SELECT 
    query,
    calls,
    mean_exec_time,
    total_exec_time
  FROM pg_stat_statements
  ORDER BY mean_exec_time DESC
  LIMIT 20;
  ```

## 4. Configuration Management

### Environment Variables
- [ ] All required environment variables set
  ```bash
  # Verify environment
  ./scripts/check_env_vars.sh production
  ```

- [ ] Secrets rotated if needed (>90 days old)
  ```bash
  # Check secret age
  aws secretsmanager describe-secret --secret-id musicgen/prod | jq '.LastRotatedDate'
  ```

- [ ] Configuration files validated
  ```bash
  # Validate YAML configs
  yamllint configs/
  python scripts/validate_configs.py
  ```

### Feature Flags
- [ ] Feature flags configured correctly
  ```python
  # Check feature flags
  python scripts/check_feature_flags.py --env production
  ```

- [ ] A/B test configurations verified
  ```bash
  # Verify A/B tests
  curl https://api.musicgen.ai/internal/experiments/active
  ```

## 5. Dependencies

### External Services
- [ ] Redis cluster healthy
  ```bash
  redis-cli -h redis.musicgen.ai ping
  redis-cli -h redis.musicgen.ai info replication
  ```

- [ ] S3 bucket accessible
  ```bash
  aws s3 ls s3://musicgen-audio/
  ```

- [ ] CDN configuration up-to-date
  ```bash
  # Check CDN status
  aws cloudfront get-distribution --id E1234567890ABC
  ```

### Third-party APIs
- [ ] Payment provider API accessible
- [ ] Email service API accessible  
- [ ] Analytics service API accessible
- [ ] Model hosting service healthy

## 6. Security Verification

### Access Control
- [ ] Deployment credentials valid
  ```bash
  # Test deployment access
  ssh deploy@production.musicgen.ai "echo 'Access verified'"
  ```

- [ ] API keys rotated if needed
  ```bash
  # Check API key age
  python scripts/audit_api_keys.py
  ```

- [ ] Firewall rules reviewed
  ```bash
  # List current rules
  sudo iptables -L -n -v
  ```

### Security Scanning
- [ ] Vulnerability scan completed
  ```bash
  # Run security scan
  ./scripts/security_scan.sh
  ```

- [ ] Dependency audit passed
  ```bash
  # Check dependencies
  pip-audit
  npm audit
  ```

## 7. Monitoring & Alerting

### Monitoring Setup
- [ ] Prometheus targets configured
  ```bash
  # Verify Prometheus config
  curl http://prometheus.musicgen.ai:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health}'
  ```

- [ ] Grafana dashboards updated
  ```bash
  # Check dashboard versions
  curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
    http://grafana.musicgen.ai/api/dashboards/uid/musicgen-prod | jq '.dashboard.version'
  ```

- [ ] Log aggregation working
  ```bash
  # Test log shipping
  logger -t deployment-test "Pre-deployment test message"
  # Verify in log aggregator
  ```

### Alert Configuration
- [ ] Critical alerts configured
  ```yaml
  # Verify alert rules
  promtool check rules alerts/*.yml
  ```

- [ ] Alert routing tested
  ```bash
  # Send test alert
  ./scripts/send_test_alert.sh
  ```

- [ ] On-call schedule verified
  ```bash
  # Check PagerDuty schedule
  pd-cli schedule show
  ```

## 8. Communication

### Team Notifications
- [ ] Deployment window communicated
  - [ ] Email sent to stakeholders
  - [ ] Slack announcement posted
  - [ ] Status page updated

- [ ] Rollback plan shared
  - [ ] Rollback procedures documented
  - [ ] Team members aware of plan
  - [ ] Emergency contacts updated

### Customer Communication
- [ ] Maintenance window announced (if applicable)
- [ ] API deprecation notices sent (if applicable)
- [ ] Documentation updates prepared

## 9. Rollback Preparation

### Rollback Resources
- [ ] Previous version containers available
  ```bash
  # Verify images exist
  docker images | grep musicgen/api
  ```

- [ ] Rollback scripts tested
  ```bash
  # Test rollback in staging
  ./scripts/test_rollback.sh staging
  ```

- [ ] Database rollback scripts ready
  ```sql
  -- Verify rollback capability
  SELECT COUNT(*) FROM schema_versions WHERE rollback_script IS NOT NULL;
  ```

### Recovery Time Objectives
- [ ] RTO documented and achievable
- [ ] RPO requirements met
- [ ] Disaster recovery plan reviewed

## 10. Final Verification

### Staging Environment
- [ ] Staging deployment successful
  ```bash
  # Deploy to staging
  ./scripts/deploy_staging.sh
  ```

- [ ] Staging tests passed
  ```bash
  # Run staging tests
  pytest tests/integration/ -m staging
  ```

- [ ] Performance benchmarks met
  ```bash
  # Run performance tests
  locust -f tests/load/locustfile.py --host=https://staging.musicgen.ai
  ```

### Sign-offs
- [ ] Engineering lead approval
  - Name: ________________
  - Date: ________________

- [ ] Operations lead approval
  - Name: ________________
  - Date: ________________

- [ ] Security lead approval
  - Name: ________________
  - Date: ________________

- [ ] Product owner approval (if major changes)
  - Name: ________________
  - Date: ________________

## Pre-Deployment Status

**Overall Status**: 
- [ ] ✅ **GO** - All checks passed, proceed with deployment
- [ ] ⚠️ **CONDITIONAL GO** - Minor issues noted, proceed with caution
- [ ] ❌ **NO GO** - Critical issues found, deployment blocked

**Notes/Issues**:
```
[Document any issues, exceptions, or special considerations here]
```

**Risk Assessment**:
- **Low Risk**: All checks green, standard deployment
- **Medium Risk**: Some warnings, increased monitoring needed
- **High Risk**: Multiple issues, consider postponing

---

**Checklist Completed By**: ________________  
**Date/Time**: ________________  
**Reviewed By**: ________________  
**Review Date/Time**: ________________