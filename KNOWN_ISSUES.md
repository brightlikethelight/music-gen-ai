# Known Issues and Technical Debt

> **⚠️ ACADEMIC PROJECT**: This document provides transparency about current technical issues in this Harvard CS 109B academic project.

## 🚨 Critical Issues

### CI/CD Pipeline Failures
**Status**: ❌ BROKEN (100% failure rate)  
**Last Working**: Never fully functional  
**Impact**: Cannot validate code changes automatically

#### Root Causes:
1. **Black Formatting Violations** - Fixed in latest commit
2. **Test Authentication Issues** - Tests expect 422 but get 401 errors
3. **Missing System Dependencies** - ffmpeg not installed in CI environment
4. **Test Timeouts** - ML model downloads during test execution
5. **Dependency Conflicts** - Multiple requirements files with conflicts

#### Failed Workflows:
- `CI Pipeline` - Code quality and unit tests
- `Comprehensive Test Suite` - Full test battery

### Test Infrastructure
**Unit Tests**: 65 skipped, 1-2 failing  
**Integration Tests**: 20/20 passing ✅  
**Coverage**: ~13% (target: 60%+)

#### Issues:
- Most unit tests skip due to missing auth module imports
- Test expectations don't match actual API behavior
- Mocking infrastructure incomplete
- Heavy ML dependencies cause timeouts

### Code Quality
**Linting Violations**: 225+ issues
- Line length (E501): 203 instances
- Unused imports (F401): 17 instances
- Type annotation issues: 40+ MyPy errors

### Documentation vs Reality Gaps
| Feature | Documentation Claims | Actual State |
|---------|---------------------|--------------|
| Test Coverage | "6.2%" → Actually ~13% | Misleading |
| Failing Tests | "50+ failing" → Actually 65 skipped | Inaccurate |
| Docker Image | Uses wrong project image | Broken |
| PyPI Package | Shows badge but doesn't exist | False |

## ⚠️ Technical Debt

### Architecture Issues
- **Authentication**: Middleware expects tokens but tests don't provide them
- **Configuration**: Multiple overlapping config systems
- **Dependencies**: requirements.txt vs pyproject.toml conflicts
- **Imports**: Circular dependencies in some modules

### Deployment Concerns
- Docker references incorrect base image (`ashleykza/tts-webui`)
- Kubernetes configs are untested examples only
- Environment variables not properly documented
- SSL/TLS configuration missing

### Security Considerations
- Hardcoded JWT secrets in test files
- No rate limiting on authentication endpoints
- CORS configuration may be too permissive
- No input validation on some endpoints

## 🔧 Recommended Fixes

### Immediate (Critical)
1. Fix test authentication mocking
2. Mock ML model loading in tests
3. Consolidate requirements files
4. Add ffmpeg to CI environment

### Short-term (Important)
1. Increase test coverage to 60%+
2. Fix all linting violations
3. Update Docker base image
4. Document environment variables

### Long-term (Nice to Have)
1. Implement proper monitoring
2. Add comprehensive security audit
3. Create integration test suite
4. Implement proper CI/CD pipeline

## 📝 Notes for Contributors

This is an academic project created for learning purposes. The issues documented here are part of the learning process and demonstrate real-world software engineering challenges. Contributors should:

1. Focus on educational value over perfection
2. Document any new issues discovered
3. Prioritize learning objectives
4. Be transparent about limitations

---

*Last Updated: January 2025*  
*Course: Harvard CS 109B - Advanced Data Science*  
*Instructor Notification: This project intentionally includes technical debt for educational discussion*