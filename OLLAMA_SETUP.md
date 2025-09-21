# Ollama CORS Configuration

## Issue
When accessing Ollama API from a web browser, you may encounter a 403 error due to CORS (Cross-Origin Resource Sharing) restrictions.

## Solution
Set the `OLLAMA_ORIGINS` environment variable to allow your web application to access the Ollama API.

### Windows
```cmd
set OLLAMA_ORIGINS=*
```

Or for specific origins:
```cmd
set OLLAMA_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

### Linux/macOS
```bash
export OLLAMA_ORIGINS="*"
```

Or for specific origins:
```bash
export OLLAMA_ORIGINS="http://localhost:8000,http://127.0.0.1:8000"
```

### Permanent Setup
Add the environment variable to your system's environment variables or shell profile to make it persistent.

### Restart Ollama
After setting the environment variable, restart the Ollama service:
```cmd
ollama serve
```

## Security Note
Using `OLLAMA_ORIGINS=*` allows all origins to access your Ollama server. For production environments, specify only the domains that need access.
