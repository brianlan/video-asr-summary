# LLM Model Configuration for Video Processing Pipeline

## 🎯 **New Features Added**

The video processing pipeline now supports configurable LLM models and endpoints, giving you flexibility to choose the best model for your specific needs.

## 🔧 **New Parameters**

### **Command Line Arguments**

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--llm-model` | `-m` | `gemini-2.5-pro-preview-03-25` | LLM model name for content analysis |
| `--llm-endpoint` | `-e` | `https://openai.newbotai.cn/v1` | LLM API endpoint URL |
| `--llm-timeout` | | `1200` | LLM API request timeout in seconds |

## 📚 **Usage Examples**

### **Basic Usage with Default Model**
```bash
python scripts/process_video.py video.mp4 ./output --analysis-language zh
```

### **OpenAI GPT-4**
```bash
python scripts/process_video.py video.mp4 ./output \
  --analysis-language zh \
  --llm-model "gpt-4" \
  --llm-endpoint "https://api.openai.com/v1" \
  --llm-timeout 300
```

### **Alibaba Cloud Qwen**
```bash
python scripts/process_video.py video.mp4 ./output \
  --analysis-language zh \
  --llm-model "qwen-max" \
  --llm-endpoint "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --llm-timeout 600
```

### **DeepSeek Chat**
```bash
python scripts/process_video.py video.mp4 ./output \
  --analysis-language zh \
  --llm-model "deepseek-chat" \
  --llm-endpoint "https://api.deepseek.com/v1" \
  --llm-timeout 300
```

### **Zhipu AI GLM**
```bash
python scripts/process_video.py video.mp4 ./output \
  --analysis-language zh \
  --llm-model "glm-4" \
  --llm-endpoint "https://open.bigmodel.cn/api/paas/v4" \
  --llm-timeout 300
```

## 🌐 **Popular Model Options**

### **For Chinese Content Analysis**

| Provider | Model | Endpoint | Strengths |
|----------|-------|----------|-----------|
| **Default** | `gemini-2.5-pro-preview-03-25` | `https://openai.newbotai.cn/v1` | Good Chinese understanding, reliable |
| **Alibaba** | `qwen-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Excellent Chinese, fast |
| **DeepSeek** | `deepseek-chat` | `https://api.deepseek.com/v1` | Cost-effective, good reasoning |
| **Zhipu AI** | `glm-4` | `https://open.bigmodel.cn/api/paas/v4` | Chinese-focused, affordable |

### **For English Content Analysis**

| Provider | Model | Endpoint | Strengths |
|----------|-------|----------|-----------|
| **OpenAI** | `gpt-4` | `https://api.openai.com/v1` | High quality, comprehensive |
| **OpenAI** | `gpt-4-turbo` | `https://api.openai.com/v1` | Faster, more efficient |
| **OpenAI** | `gpt-3.5-turbo` | `https://api.openai.com/v1` | Cost-effective |

## ⚙️ **Configuration Guidelines**

### **Timeout Settings**
- **Fast models** (GPT-3.5, Qwen-Plus): 300-600 seconds
- **Premium models** (GPT-4, Qwen-Max): 600-1200 seconds  
- **Complex analysis**: 1200+ seconds

### **API Key Setup**
```bash
# Set your API key for the chosen service
export OPENAI_ACCESS_TOKEN="your-api-key-here"
```

### **Model Selection Criteria**

**Choose based on:**
1. **Language**: Chinese models for Chinese content, international models for English
2. **Cost**: Balance between quality and pricing
3. **Speed**: Some models are faster but may sacrifice quality
4. **Accuracy**: Premium models generally provide better analysis

## 🔍 **Testing Different Models**

### **Quick Test**
```bash
# Test configuration without processing
python scripts/process_video.py video.mp4 ./output \
  --llm-model "your-model" \
  --llm-endpoint "your-endpoint" \
  --status
```

### **Resume with Different Model**
```bash
# If analysis failed, you can resume with a different model
python scripts/process_video.py video.mp4 ./output \
  --llm-model "backup-model" \
  --llm-endpoint "backup-endpoint" \
  --resume
```

## 🎯 **Best Practices**

1. **Start with defaults** for most use cases
2. **Use Chinese models** for Chinese content analysis
3. **Increase timeout** for long videos or complex content
4. **Test with --status** flag before full processing
5. **Keep backup model/endpoint** ready for failures

## 📋 **Configuration Display**

The pipeline now shows your LLM configuration:
```
🎬 VIDEO PROCESSING PIPELINE
========================================
📹 Input: /path/to/video.mp4
📁 Output: ./output
🌐 Analysis Language: zh
🤖 LLM Model: qwen-max
📡 LLM Endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
⏱️ LLM Timeout: 600s
📋 Content Type: Auto-detect
```

## 🚀 **Benefits**

- **Flexibility**: Choose the best model for your content and budget
- **Reliability**: Switch models if one service is down
- **Performance**: Optimize timeout for your specific use case
- **Cost Control**: Use more affordable models when appropriate
- **Language Optimization**: Use native language models for better results

This enhancement makes the video processing pipeline much more versatile and production-ready! 🎉
