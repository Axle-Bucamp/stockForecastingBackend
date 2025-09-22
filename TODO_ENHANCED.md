# 🚀 Hygdra Forecasting - Enhanced TODO System

## 📋 **Project Overview**
**Hygdra Forecasting** is a fast stock trend forecasting algorithm using deep learning models for financial market predictions. The project consists of:
- **Core Package**: `hygdra_forecasting` - ML models and utilities
- **API Service**: `app/api` - FastAPI backend
- **Frontend**: `app/frontend` - Dash dashboard
- **Scheduler**: `app/sheduler` - Training and inference automation

## 🎯 **Critical Issues Identified**

### 🔴 **HIGH PRIORITY - Model Loading Issue**
**Problem**: Model weights produce different loss on same data between training and inference

**Root Causes Found**:
1. **Sequence Creation Mismatch**:
   - Training: Uses `TIME_DELTA_LABEL = 12` offset in `create_sequences()`
   - Inference: Uses `create_sequences_inference()` with NO offset
   - **Fix**: Align sequence creation logic

2. **Normalization Bug** in `app/sheduler/inference.py:50`:
   ```python
   # WRONG:
   df[ticker]["close"] = df[ticker]["close"] * dict_unorm[ticker]["close"]["std"] + dict_unorm[ticker]["close"]["std"]
   # CORRECT:
   df[ticker]["close"] = df[ticker]["close"] * dict_unorm[ticker]["close"]["std"] + dict_unorm[ticker]["close"]["mean"]
   ```

3. **Model Mode Differences**:
   - Training: `.train()` mode with dropout active
   - Inference: `.eval()` mode with dropout disabled
   - **Impact**: Different forward pass behavior

4. **Data Shuffling**:
   - Training: `shuffle=True`
   - Inference: Sequential processing

## 📝 **TODO Tasks**

### 🏗️ **Infrastructure & Setup**
- [x] **Project Analysis Complete** - Analyzed project structure and identified issues
- [ ] **Add UV to app/api** - Convert to uv dependency management
- [ ] **Add UV to app/frontend** - Convert to uv dependency management  
- [ ] **Add UV to app/sheduler** - Convert to uv dependency management
- [ ] **Run MyPy Type Checking** - Check type annotations on hygdra_forecasting

### 🐛 **Bug Fixes**
- [ ] **Fix Normalization Bug** - Correct denormalization in inference.py:50
- [ ] **Align Sequence Creation** - Make training and inference use same sequence logic
- [ ] **Standardize Model Modes** - Ensure consistent model state between training/inference
- [ ] **Fix Data Shuffling** - Make inference deterministic for reproducibility

### 🔧 **Code Quality**
- [ ] **Add Type Annotations** - Improve type hints across codebase
- [ ] **Standardize Error Handling** - Consistent exception handling
- [ ] **Add Logging** - Structured logging for debugging
- [ ] **Documentation** - Update README and docstrings

### 🚀 **Performance & Features**
- [ ] **Optimize Data Loading** - Improve DataLoader performance
- [ ] **GPU Memory Management** - Better CUDA memory handling
- [ ] **Model Validation** - Add comprehensive model validation
- [ ] **Testing Framework** - Unit tests for critical components

### 🏗️ **Architecture Improvements**
- [ ] **Unified Data Pipeline** - Single data preprocessing pipeline
- [ ] **Configuration Management** - Centralized config system
- [ ] **Model Registry** - Version and track model weights
- [ ] **Monitoring** - Add performance monitoring

## 🔍 **Technical Debt**
- **Multiple DataLoaders**: Both `dataloader.py` and `dataloader_kraken.py` exist
- **Inconsistent Naming**: `sheduler` vs `scheduler` spelling
- **Hardcoded Values**: Magic numbers throughout codebase
- **Mixed Dependencies**: Some files use requirements.txt, others use pyproject.toml

## 📊 **Model Architecture Notes**
- **ConvCausalLTSM**: Main model with Conv1D + 4 LSTM layers
- **LtsmAttentionforecastPred**: Alternative with attention mechanism
- **Input Shape**: (36, 7) - 36 timesteps, 7 features per ticker
- **Features**: close, width, rsi, roc, volume, diff, percent_change_close

## 🎯 **Next Steps**
1. Fix the critical normalization bug in inference.py
2. Align sequence creation between training and inference
3. Add uv dependency management to all app subfolders
4. Run comprehensive testing to verify fixes

---
**Last Updated**: $(date)
**Status**: 🔴 Critical issues identified, ready for fixes
