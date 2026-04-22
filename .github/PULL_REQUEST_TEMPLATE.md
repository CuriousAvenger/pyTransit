---
name: Pull Request
about: Describe the changes and motivation for this PR
---

## Summary

<!-- One or two sentences describing what this PR does. -->

## Motivation

<!-- Why is this change needed? Link related issues with "Closes #NNN". -->

## Changes

- [ ] New feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactoring
- [ ] Test coverage

## Testing

<!-- How was this tested? All existing tests must still pass. -->

```bash
pytest tests/ -v
```

## Checklist

- [ ] Tests pass locally (`pytest tests/`)
- [ ] Code is formatted (`black pyTransitPhotometry/ tests/`)
- [ ] Linting passes (`flake8 pyTransitPhotometry/ tests/ --max-line-length=100`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Docstrings added/updated for new public functions
