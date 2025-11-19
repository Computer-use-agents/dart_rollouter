#!/bin/bash

# Simple test for vLLM wrapper service
BASE_URL="http://112.30.139.26:50599"

echo "=== Test 1: Minimal request (no parameters) ==="
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

echo -e "\n\n=== Test 2: With parameters ==="
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
      "max_tokens": 100
    }
  }'

echo -e "\n\n=== Test 3: With list content (multimodal format) ==="
curl -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hello"
          }
        ]
      }
    ],
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 100
    }
  }'

