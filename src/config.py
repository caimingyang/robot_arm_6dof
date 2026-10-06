"""
VLM / LLM Configuration for Alibaba Cloud Bailian (Qwen)
==========================================================
Compatible with OpenAI API format.

NOTE: The available qwen3.7-* models are text-only. The VLA controller
sends structured scene-state text descriptions rather than images.
For true vision-based control, use --vlm openai with a vision-capable
model (e.g. gpt-4o) and set OPENAI_API_KEY.
"""

# Alibaba Cloud Bailian (Qwen) API Configuration
QWEN_CONFIG = {
    "api_key": "",
    "base_url": "https://ws-9fj29p5g4c84eygc-vpc-uf6mkccrxko3e999sns4z.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    "model": "qwen3.7-max",  # Options: qwen3.7-max, qwen3.7-plus, qwen3.7-flash
}
