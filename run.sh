export CUDA_VISIBLE_DEVICES=0
# 离线推理
# flashtts infer -i "这里是北京责任有限公司" -m ./models/Spark-TTS-0.5B -b sglang -o out/demo.wav

flashtts serve \
    --model_path ./models/Spark-TTS-0.5B \
    --backend sglang \
    --llm_device cuda \
    --tokenizer_device cuda \
    --detokenizer_device cuda \
    --wav2vec_attn_implementation sdpa \
    --llm_attn_implementation sdpa \
    --torch_dtype "bfloat16" \
    --max_length 32768 \
    --llm_gpu_memory_utilization 0.8 \
    --fix_voice \
    --host 0.0.0.0 \
    --port 8000