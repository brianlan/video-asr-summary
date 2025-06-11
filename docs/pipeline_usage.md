# Video Processing Pipeline

This document shows how to use the complete video processing pipeline with state management and resumability.

## Quick Start

### Basic Usage

```bash
# Process a video file
python scripts/process_video.py input_video.mp4 ./output

# With specific analysis language and content type
python scripts/process_video.py interview.mp4 ./results \
    --analysis-language es --content-type political_commentary

# Resume interrupted processing  
python scripts/process_video.py video.mp4 ./output --resume

# Check pipeline status
python scripts/process_video.py video.mp4 ./output --status

# Clean up when done
python scripts/process_video.py video.mp4 ./output --cleanup
```

### Pipeline Features

1. **State Management**: Saves progress at each step
2. **Resumability**: Continue from where processing was interrupted  
3. **Intermediate Files**: Debug and inspect results at each stage
4. **Multi-language Analysis**: Get results in your preferred language
5. **Content Type Detection**: Automatic or manual content classification

### Output Structure

```
output_directory/
├── pipeline_state.json      # Processing state and metadata
├── audio.wav               # Extracted audio from video
├── audio_metadata.json     # Audio file information  
├── transcription.json      # ASR results with timestamps
├── analysis.json          # LLM content analysis
└── pipeline_result.json   # Final combined results
```

### Pipeline Steps

1. **Video Info Extraction**: Duration, resolution, file size
2. **Audio Extraction**: Extract optimized audio for ASR
3. **Transcription**: Speech-to-text with confidence scores
4. **Content Analysis**: LLM-based conclusion and argument extraction  
5. **Finalization**: Combine results and save final output

### State Management Benefits

- **Resume from failures**: Network issues, system crashes, API rate limits
- **Iterative development**: Test different analysis parameters
- **Cost efficiency**: Avoid re-running expensive LLM calls
- **Debugging**: Inspect intermediate results to troubleshoot issues
- **Batch processing**: Process large collections with checkpoints

### Error Handling

The pipeline automatically saves state when errors occur:

```bash
# If processing fails, check status
python scripts/process_video.py video.mp4 ./output --status

# Resume from the failed step
python scripts/process_video.py video.mp4 ./output --resume
```

### Advanced Usage

```bash
# Set custom LLM endpoint
export OPENAI_BASE_URL="https://your-custom-endpoint.com"
export OPENAI_ACCESS_TOKEN="your-api-key"

# Force restart (ignore existing state)
python scripts/process_video.py video.mp4 ./output --no-resume

# Verbose output for debugging
python scripts/process_video.py video.mp4 ./output --verbose
```

## Integration with Existing Code

```python
from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator

# Create orchestrator
orchestrator = PipelineOrchestrator("./output")

# Process video with full control
results = orchestrator.process_video(
    video_path="input.mp4",
    analysis_language="en", 
    content_type="technical_review",
    resume=True,
    cleanup_intermediate=False
)

# Access results
transcription = results["transcription"]
analysis = results["analysis"] 
```

This design provides a robust, user-friendly pipeline that handles real-world challenges like interruptions, debugging needs, and cost optimization.
