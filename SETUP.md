# Setup and Installation Guide

Complete guide for installing and configuring OpenJailbreak for LLM jailbreak research.

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 4GB minimum (16GB+ recommended for local models)
- **Storage**: 2GB for framework + model storage space

### Network Requirements
- Internet connection for API-based models
- Firewall access for LLM provider APIs
- Optional: GPU access for local model deployment

## Installation

### Option 1: From Source (Recommended)
```bash
# Clone the repository
git clone https://github.com/zenithfang/OpenJailbreak.git
cd OpenJailbreak

# Install the framework
pip install -e .

# Install with vLLM support (for local models)
pip install -e ".[vllm]"

# Install development dependencies
pip install -e ".[dev]"  # pytest, black, isort, ruff, pre-commit
```

## API Keys Configuration

### Environment Variables Setup

Create a `.env` file in the project root (copy from `.env.example`):

```bash
# Copy the template
cp .env.example .env
# Edit with your actual API keys
```

### Provider-Specific Configuration

#### OpenAI
```bash
export OPENAI_API_KEY="sk-your_openai_api_key_here"
```

#### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

#### Azure OpenAI
```bash
export AZURE_API_KEY="your_azure_api_key_here"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2025-01-01-preview"
```

#### AWS Bedrock
```bash
export AWS_ACCESS_KEY_ID="your_aws_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key_here"
export AWS_REGION="us-east-1"
```

#### Google Vertex AI
```bash
# Method 1: Service Account Key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/vertex_ai_service_account.json"
export GOOGLE_CLOUD_PROJECT="your_project_id"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Method 2: Application Default Credentials
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your_project_id"
```

#### Aliyun DashScope
```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key_here"
```

#### Together AI
```bash
export TOGETHER_API_KEY="your_together_api_key_here"
```

### Self-Hosted Models
For self-deployed OpenAI-compatible models (SGLang, vLLM):
```bash
export OPENAI_API_BASE="http://your-endpoint/v1"
export OPENAI_API_KEY="your_custom_key"  # or any placeholder
```

## Dependencies

### Core Dependencies
Automatically installed with `pip install -e .`:
- **torch** (≥2.0.0) - Core ML framework
- **transformers** (≥4.36.0) - Hugging Face models
- **litellm** (≥1.72.0) - Multi-provider LLM integration
- **datasets** (≥2.14.0) - Dataset loading
- **pandas** (≥2.0.0) - Data manipulation
- **click** (≥8.1.3) - CLI framework
- **pyyaml** (≥6.0) - Configuration files

### Optional Dependencies
```bash
# For local model deployment
pip install -e ".[vllm]"  # Installs vLLM

# For development
pip install -e ".[dev]"   # Installs pytest, black, ruff, etc.
```

### Manual Dependencies (if needed)
```bash
# NLP libraries
pip install nltk spacy sentence-transformers

# Advanced ML libraries
pip install nanogcg peft

# Utilities
pip install tqdm colorama matplotlib
```

## Verification

### Test Installation
```bash
# Basic functionality test
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --samples 1 \
    --verbose

# Reproduction scripts (from the writeup)
./examples/scripts/test_abj_attack.sh
./examples/scripts/test_mousetrap.sh
./examples/scripts/test_query_attack.sh
./examples/scripts/test_wiki_text_infilling.sh
```

### Test API Connections
```bash
# Test your primary provider
python examples/scripts/test_comprehensive.py --attack_name abj --samples 1

# Multi-provider test
python tests/test_model_integration.py
```

## Configuration Files

### Project Configuration
- `pyproject.toml` - Project metadata and dependencies
- `requirements.txt` - Alternative dependency specification
- `.env` - Environment variables (create from `.env.example`)

### Runtime Configuration
- Config files in `examples/config/` for advanced usage
- YAML/JSON configuration support for batch operations

## Troubleshooting

### Common Installation Issues

#### Python Version
```bash
# Check Python version
python --version  # Should be 3.8+

# If using pyenv
pyenv install 3.9.0
pyenv local 3.9.0
```

#### Permission Issues
```bash
# Use user installation if needed
pip install --user -e .

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

#### GPU Dependencies (for local models)
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Common Runtime Issues

#### API Key Problems
- Verify API keys are correctly set in environment
- Check for typos in environment variable names
- Ensure API keys have sufficient credits/permissions

#### Model Access Issues
- Verify model names match provider specifications
- Check if your API key has access to the requested model
- Some models require special access approval

#### Rate Limiting
- Use `--samples` to limit request volume
- Add delays between requests if needed
- Consider switching to less restricted providers for testing

### Getting Help

1. **Check Issues**: Search existing GitHub issues
2. **Enable Verbose**: Use `--verbose` for detailed error messages
3. **Test Components**: Use individual test scripts in `examples/scripts/`
4. **Community**: Open an issue or discussion on GitHub

## Advanced Configuration

### Performance Optimization
```bash
# Set worker limits
export MAX_WORKERS=4
export TIMEOUT_SECONDS=60

# Cache configuration
export DATASET_CACHE_DIR=".cache/datasets"
export HUGGINGFACE_CACHE_DIR=".cache/huggingface"
```

### Logging Configuration
```bash
export LOG_LEVEL=INFO
export LOG_FILE="logs/openjailbreak.log"
```

### Security Settings
```bash
export ENABLE_CONTENT_FILTER=true
export MAX_RESPONSE_LENGTH=4096
export RATE_LIMIT_PER_MINUTE=60
```

---

*For usage instructions, see the [User Guide](docs/USER_GUIDE.md). For development, see the [Developer Guide](docs/DEVELOPER_GUIDE.md).*
