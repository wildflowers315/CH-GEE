# CHMapper2 Refactoring Plan

## Current State Analysis

The current codebase is organized as a collection of Python scripts with the following main components:

1. **Core Functionality**
   - `chm_main.py`: Main processing pipeline
   - `train_predict_map.py`: Training and prediction logic
   - `dl_models.py`: Deep learning model implementations

2. **Data Sources**
   - `sentinel1_source.py`
   - `sentinel2_source.py`
   - `alos2_source.py`
   - `l2a_gedi_source.py`
   - `canopyht_source.py`

3. **Utilities**
   - `raster_utils.py`
   - `evaluation_utils.py`
   - `utils.py`

4. **Evaluation and Output**
   - `evaluate_predictions.py`
   - `save_evaluation_pdf.py`
   - `combine_heights.py`

## Refactoring Goals

1. **Package Structure**
   - Convert to a proper Python package
   - Implement modular architecture
   - Improve code reusability
   - Enable both standalone and library usage

2. **Code Organization**
   - Separate core logic from utilities
   - Implement proper dependency injection
   - Reduce code duplication
   - Improve error handling

3. **Testing and Documentation**
   - Expand test coverage
   - Add comprehensive documentation
   - Implement CI/CD pipeline

## Proposed New Structure

```
CHMapper2/
├── src/
│   └── chmapper/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   ├── training.py
│       │   └── prediction.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── dl_models.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── sources/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── sentinel.py
│       │   │   ├── alos.py
│       │   │   └── gedi.py
│       │   └── processing.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── raster.py
│       │   ├── evaluation.py
│       │   └── common.py
│       └── visualization/
│           ├── __init__.py
│           └── plotting.py
├── tests/
│   ├── __init__.py
│   ├── test_core/
│   ├── test_models/
│   ├── test_data/
│   └── test_utils/
├── examples/
│   ├── basic_usage.py
│   ├── basic_usage.ipynb
│   ├── advanced_usage.py
│   └── advanced_usage.ipynb
├── docs/
│   ├── api/
│   ├── guides/
│   └── examples/
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Implementation Steps

### Phase 1: Package Structure Setup
1. Create new directory structure
2. Set up package configuration files
3. Move existing code to new structure
4. Update imports and dependencies

### Phase 2: Core Refactoring
1. Implement base classes for data sources
2. Refactor pipeline logic into modular components
3. Implement proper configuration management
4. Add logging and error handling

### Phase 3: Testing and Documentation
1. Write unit tests for core functionality
2. Add integration tests
3. Generate API documentation
4. Create usage examples

### Phase 4: CI/CD and Distribution
1. Set up GitHub Actions for CI/CD
2. Configure package distribution
3. Add version management
4. Create release workflow

## Migration Strategy

1. **Incremental Migration**
   - Keep existing functionality working during refactoring
   - Move one component at a time
   - Maintain backward compatibility where possible

2. **Testing Strategy**
   - Write tests before refactoring
   - Ensure test coverage for new code
   - Implement regression testing

3. **Documentation**
   - Document API changes
   - Update usage examples
   - Maintain changelog

## Dependencies and Requirements

Current dependencies will be organized into:
- Core requirements
- Development requirements
- Optional dependencies

## Timeline and Milestones

1. **Week 1-2**: Package Structure Setup
2. **Week 3-4**: Core Refactoring
3. **Week 5-6**: Testing and Documentation
4. **Week 7-8**: CI/CD and Distribution

## Success Criteria

1. All tests passing
2. Documentation complete
3. Example notebooks working
4. CI/CD pipeline operational
5. Package installable via pip

## Notes

- Maintain backward compatibility where possible
- Focus on code quality and maintainability
- Consider performance implications
- Plan for future extensibility 