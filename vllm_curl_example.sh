#!/bin/bash

# Example curl commands to call vLLM API based on trajectory_runner.py
# Server: http://112.30.139.26:50599

BASE_URL="http://112.30.139.26:50599"

echo "=== 1. Check if server is running (root endpoint) ==="
curl -X GET "${BASE_URL}/"

echo -e "\n\n=== 2. Check health endpoint (if available) ==="
curl -X GET "${BASE_URL}/health" || echo "Health endpoint not available"

echo -e "\n\n=== Option A: Direct vLLM endpoint (/v1/chat/completions) ==="
curl -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "frequency_penalty": 0.0,
    "seed": 42
  }'

echo -e "\n\n=== Option B: Wrapper service endpoint (/generate) - Simple text ==="
curl -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "parameters": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 512,
      "frequency_penalty": 0.0,
      "seed": 42
    }
  }'

echo -e "\n\n=== Option B2: Wrapper service endpoint (/generate) - With list content ==="
curl -X POST "${BASE_URL}/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hello, how are you?"
          }
        ]
      }
    ],
    "parameters": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 512,
      "frequency_penalty": 0.0,
      "seed": 42
    }
  }'

echo -e "\n\n=== Option C: Multimodal request with image (direct vLLM) ==="
curl -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            }
          }
        ]
      }
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "frequency_penalty": 0.0,
    "seed": 42
  }'

echo -e "\n\n=== Option D: Multimodal request with image (wrapper service) ==="
curl -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            }
          }
        ]
      }
    ],
    "parameters": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 512,
      "frequency_penalty": 0.0,
      "seed": 42,
      "logprobs": false,
      "return_tokens_as_token_ids": false
    }
  }'

echo -e "\n\n=== Option E: Minimal request (no parameters) ==="
curl -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'

