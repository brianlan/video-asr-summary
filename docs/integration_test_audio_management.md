# Integration Test Audio File Management

## 🎯 **Problem Solved**

Fixed the critical issue where integration tests relied on a hard-coded local file path that wouldn't work on other machines or CI/CD environments.

## ❌ **Previous Issues**

```python
# BEFORE: Hard-coded, machine-specific path
self.test_audio_path = Path("/Users/rlan/Downloads/ruige-huangjin-4000/audio.wav")
```

**Problems:**
- ❌ Machine-dependent (only works on your machine)
- ❌ Not portable (breaks in CI/CD)
- ❌ Fragile (fails if file moved/deleted)
- ❌ Not self-contained (external dependency)

## ✅ **New Solution**

### **1. Environment Variable Support**
```bash
# For real audio file testing
export FUNASR_TEST_AUDIO_PATH="/path/to/your/audio.wav"
python -m pytest tests/ -m "integration"
```

### **2. Synthetic Audio Generation** 
```python
# Automatically creates synthetic test audio if no env var set
def _create_test_audio_file(self) -> Union[Path, None]:
    # Creates 3-second 440Hz sine wave in temporary file
    # Returns None if scipy/numpy not available (graceful skip)
```

### **3. Graceful Skipping**
```python
if self.test_audio_path is None:
    pytest.skip("No test audio available. Set FUNASR_TEST_AUDIO_PATH or install scipy/numpy")
```

## 🔧 **How It Works**

### **Priority Order:**
1. **Environment Variable** (`FUNASR_TEST_AUDIO_PATH`) - Use real audio file
2. **Synthetic Audio** - Generate temporary audio file with scipy/numpy
3. **Graceful Skip** - Skip integration tests if neither available

### **Test Scenarios:**

| Environment | Behavior | Result |
|-------------|----------|---------|
| **Your Machine** (with env var set) | Uses your real audio file | ✅ Full testing with expected content |
| **Developer Machine** (scipy installed) | Creates synthetic audio | ✅ Tests structure validation |
| **CI/CD Environment** (scipy installed) | Creates synthetic audio | ✅ Automated testing |
| **Minimal Environment** (no scipy) | Skips integration tests | ⚠️ Unit tests still run |

## 📦 **Dependencies**

### **For Synthetic Audio Generation:**
```bash
pip install numpy scipy
```

### **Alternative: Use Real Audio File**
```bash
export FUNASR_TEST_AUDIO_PATH="/path/to/your/test/audio.wav"
```

## 🚀 **Usage Examples**

### **Developer Workflow**
```bash
# Run all tests (unit + integration with synthetic audio)
python -m pytest tests/

# Run only unit tests (fast, no audio needed)
python -m pytest tests/ -m "not integration"

# Run with your specific audio file
export FUNASR_TEST_AUDIO_PATH="/path/to/your/audio.wav"
python -m pytest tests/ -m "integration"
```

### **CI/CD Configuration**
```yaml
# GitHub Actions example
- name: Install test dependencies
  run: pip install numpy scipy

- name: Run tests
  run: python -m pytest tests/
  # Integration tests will use synthetic audio automatically
```

## ✅ **Benefits**

1. **🌐 Portable**: Works on any machine
2. **🤖 CI/CD Ready**: No external file dependencies
3. **🔧 Flexible**: Supports both real and synthetic audio
4. **⚡ Fast**: Synthetic audio generation is quick
5. **🛡️ Robust**: Graceful degradation if dependencies missing
6. **🧹 Clean**: Automatic cleanup of temporary files

## 🎯 **Test Categories**

### **Unit Tests (16)** - Always Available
- ✅ Fast execution (~1.2s)
- ✅ No external dependencies
- ✅ 100% reliable

### **Integration Tests (3)** - Content Aware
- ✅ **With real audio**: Validates actual transcription content
- ✅ **With synthetic audio**: Validates transcription structure
- ✅ **Graceful skip**: If no audio available

## 🔍 **Validation Differences**

### **Real Audio File**
```python
# Can test actual transcription content
assert "黄金" in result.text  # Real content validation
```

### **Synthetic Audio File**  
```python
# Tests transcription pipeline structure
self._validate_transcription_result(result)  # Structure validation
# Note: Synthetic audio may not contain recognizable text
```

This approach ensures **robust, portable tests** that work everywhere while maintaining the ability to do comprehensive testing with real audio files when available! 🎉
