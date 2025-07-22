# Contributing to MusicGen Unified - Academic Project Guidelines

Thank you for your interest in contributing to this educational project! This document outlines how to contribute effectively to an academic research project with specific limitations and learning objectives.

## 🎓 Project Context

This is an **academic research project** developed for Harvard CS 109B: Advanced Data Science. It is:
- **NOT a production system** - No production deployment is planned
- **Educational in nature** - Designed for learning ML engineering concepts
- **Experimentally implemented** - Many features are incomplete or broken
- **Limited maintenance** - Active development tied to academic timeline

## 🚨 Before You Contribute

### Current Project Status
Please read these documents first to understand the project's current state:
- **[LIMITATIONS.md](LIMITATIONS.md)** - Honest assessment of what's broken
- **[ACADEMIC_DISCLAIMER.md](ACADEMIC_DISCLAIMER.md)** - Legal and academic disclaimers
- **README.md** - Updated project overview with realistic expectations

### Key Issues (As of last update)
- **Test Coverage**: 6.2% (needs significant improvement)
- **Failing Tests**: 50+ unit tests fail
- **Code Quality**: 225+ linting violations
- **CI/CD**: Pipeline is completely broken
- **Documentation**: Some features documented but not implemented

## 🎯 Types of Contributions We Welcome

### 1. Educational Improvements ✅
- **Learning examples**: Better code examples or tutorials
- **Documentation fixes**: Correcting inaccurate claims
- **Test improvements**: Adding tests to increase coverage
- **Code quality**: Fixing linting issues and type annotations
- **Academic context**: Enhancing educational value

### 2. Bug Fixes and Basic Functionality ✅
- **Fixing failing tests**: Help make unit tests pass
- **Import errors**: Resolving missing module issues
- **Configuration fixes**: Making setup more reliable
- **CLI improvements**: Making command-line tools work better

### 3. What We Don't Need ❌
- **Production features**: This isn't a production system
- **Performance optimization**: Focus on learning value instead
- **Enterprise features**: Monitoring, scaling, etc. are placeholder examples
- **Complex architecture**: Keep it simple for educational purposes

## 🛠️ Development Setup

### Prerequisites
- **Python**: 3.10 or 3.11 (NOT 3.12 - it's incompatible)
- **Git**: For version control
- **GPU**: Optional but recommended for testing music generation

### Initial Setup
```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/music-gen-ai.git
cd music-gen-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# 4. Run tests to see current status
python -m pytest tests/ -v
# Expect many failures - that's the current state!
```

### Understanding the Codebase
```bash
# Check test coverage (expect low coverage)
python -m pytest --cov=src/musicgen --cov-report=html

# Check code quality issues
flake8 src/ --count --statistics

# See what actually works
python -m pytest tests/test_complete_system.py -v  # These should pass
```

## 📝 Making Changes

### Before Starting
1. **Check existing issues** - Look for similar problems
2. **Start small** - Focus on one specific improvement
3. **Understand limitations** - Review LIMITATIONS.md
4. **Ask questions** - Email brightliu@college.harvard.edu for clarification

### Development Workflow
```bash
# 1. Create a feature branch
git checkout -b fix/specific-issue-description

# 2. Make your changes
# Focus on educational value and basic functionality

# 3. Test your changes
python -m pytest tests/ -v  # Try not to break more tests

# 4. Check code quality
black src/ tests/
isort src/ tests/
flake8 src/ --count

# 5. Commit with clear message
git commit -m "fix: specific description of what you fixed"

# 6. Push and create pull request
git push origin fix/specific-issue-description
```

### Code Style
- **Follow existing patterns**: Don't introduce new complexity
- **Add type annotations**: Help with code quality
- **Write docstrings**: Explain educational concepts
- **Keep it simple**: This is for learning, not production

## 🧪 Testing Guidelines

### Current Test Status
- **Integration tests**: ✅ 20/20 passing (these work!)
- **Unit tests**: ❌ 50+ failing (need major fixes)
- **Coverage**: ❌ 6.2% (needs significant improvement)

### Testing Priorities
1. **Fix existing failing tests** before writing new ones
2. **Add tests for core functionality** (music generation, API basics)
3. **Don't worry about edge cases** - focus on main functionality
4. **Add integration tests** for new features

### Example Test Contribution
```python
# Good: Simple, educational test
def test_music_generator_basic_functionality():
    """Test basic music generation works (educational example)."""
    generator = MusicGenerator()
    result = generator.generate("happy music", duration=5.0)
    assert result is not None
    assert len(result) > 0

# Avoid: Complex production-style tests
def test_enterprise_scaling_with_kubernetes_deployment():
    # Too complex for educational project
    pass
```

## 📚 Documentation Contributions

### What Helps
- **Fixing inaccurate claims**: Remove "production-ready" language
- **Adding realistic examples**: Show what actually works
- **Explaining limitations**: Help users understand constraints
- **Educational context**: Connect to learning objectives

### Documentation Style
- **Be honest**: Don't oversell capabilities
- **Explain concepts**: Help others learn
- **Provide context**: Why decisions were made
- **Link to resources**: Reference Meta's AudioCraft, etc.

## 🔍 Code Review Process

### What We Look For
- **Educational value**: Does this help people learn?
- **Honesty**: Are claims accurate?
- **Simplicity**: Is it easy to understand?
- **Functionality**: Does it work as expected?

### What We Don't Require
- **Production standards**: This isn't production code
- **Perfect optimization**: Learning value over performance
- **Comprehensive testing**: Basic coverage is fine
- **Enterprise features**: Keep it academic

## 🤝 Community Guidelines

### Academic Collaboration
- **Be respectful**: This is a learning environment
- **Ask questions**: Educational discussions are welcome
- **Share knowledge**: Help others understand concepts
- **Give context**: Explain your educational background

### Communication
- **Issues**: Use GitHub issues for bugs and improvements
- **Email**: brightliu@college.harvard.edu for academic questions
- **Pull Requests**: Clear descriptions of educational value

## 🎯 Project Goals

### What Success Looks Like
- **Higher test coverage**: Get above 60%
- **Passing CI/CD**: Green build pipeline
- **Educational value**: Good learning resource
- **Honest documentation**: Accurate representation of capabilities
- **Basic functionality**: Core features work reliably

### What Success Doesn't Require
- **Production deployment**: Not the goal
- **Enterprise features**: Keep it simple
- **Perfect performance**: Focus on education
- **Commercial viability**: Academic project only

## 📞 Getting Help

### For Technical Issues
1. Check [LIMITATIONS.md](LIMITATIONS.md) first
2. Review failing tests to understand current status
3. Create GitHub issue with clear description
4. Include relevant error messages and context

### For Academic Questions
- **Email**: brightliu@college.harvard.edu
- **Subject**: Include "CS 109B MusicGen"
- **Context**: Mention your educational background
- **Scope**: Focus on learning objectives

## 📜 License and Attribution

By contributing, you agree that:
- Your contributions will be licensed under MIT License
- You understand this is an academic project
- Your work may be used for educational purposes
- You won't hold the project liable for any issues

## 🙏 Acknowledgments

Thanks for helping improve this educational project! Your contributions help:
- **Students**: Learn ML engineering concepts
- **Instructors**: Teach real-world challenges
- **Researchers**: Understand prototype limitations
- **Community**: Build better academic resources

---

**Remember**: This is a learning project, not a production system. Focus on educational value, honesty, and helping others understand ML engineering concepts. Perfect is the enemy of educational!

*Last updated: [Date of last revision]*