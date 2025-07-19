# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :x:                |

## Reporting a Vulnerability

We take the security of MusicGen AI seriously. If you have discovered a security vulnerability in this project, please report it to us as described below.

### Reporting Process

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email your findings to `security@musicgen-ai.example.com` (replace with your actual security email)
3. Include the following information:
   - Type of vulnerability (e.g., XSS, SQL Injection, Authentication Bypass)
   - Full paths of source file(s) related to the vulnerability
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if possible)
   - Impact assessment

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Initial Assessment**: Within 7 days, we will send an initial assessment and expected timeline
- **Updates**: We will keep you informed about the progress at least every 14 days
- **Disclosure**: We follow responsible disclosure and will coordinate with you on public disclosure

### Security Best Practices for Users

1. **API Keys**: Never commit API keys or secrets to the repository
2. **Docker Images**: Always use official images or verify third-party images
3. **Network Security**: Run the API behind a reverse proxy with proper SSL/TLS
4. **Access Control**: Implement proper authentication for production deployments
5. **Updates**: Keep all dependencies up to date

### Known Security Considerations

1. **ML Model Security**: Models can potentially be exploited to generate inappropriate content
   - Implement content filtering for production use
   - Monitor generated content for abuse

2. **Resource Limits**: Music generation is resource-intensive
   - Implement rate limiting
   - Set generation duration limits
   - Monitor for DoS attempts

3. **File System Access**: The API writes generated audio files
   - Ensure proper file permissions
   - Implement disk quota limits
   - Regular cleanup of old files

### Security Features

This project includes several security features:

- CORS configuration for API endpoints
- Input validation and sanitization
- Secure file handling with path validation
- Rate limiting capabilities
- Prometheus metrics for security monitoring
- Kubernetes security policies (when deployed)

### Dependencies

We regularly update dependencies to patch known vulnerabilities. Run security scans with:

```bash
# Using safety
safety check

# Using bandit
bandit -r src/

# Using pip-audit
pip-audit
```

## Contact

For any security concerns, please contact:
- Security Team: `security@musicgen-ai.example.com`
- Project Maintainer: `brightliu@college.harvard.edu`

Thank you for helping keep MusicGen AI secure!