#!/bin/bash

# Working curl command for vLLM wrapper service at http://112.30.139.26:50599
# Based on trajectory_runner.py and model_service.py

BASE_URL="http://112.30.139.26:50599"

echo "=== Working Example: Simple text generation ==="
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
      "seed": 42,
      "logprobs": false,
      "return_tokens_as_token_ids": false
    }
  }' | jq '.'

echo -e "\n\n=== Working Example: Multimodal with image ==="
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
  }' | jq '.'

echo -e "\n\n=== Minimal request (no parameters) ==="
curl -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }' | jq '.'

echo -e "\n\n=== With task_id, trace_id, step (if save_local is enabled) ==="
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
      "max_tokens": 512,
      "task_id": "test_task_123",
      "trace_id": "test_trace_456",
      "step": 1
    }
  }' | jq '.'

