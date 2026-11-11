export JAVA_HOME="/data_b/qzluo/miniconda3/envs/verl-webshop"
export PATH="/data_b/qzluo/workspace/wlx/Repro-verl-agent/.venv/bin:$JAVA_HOME/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$JAVA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-r2-jvmcap-20261108:/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-r2-jvmcap-20261108/source:/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-r2-jvmcap-20261108/deps-direct:/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-r2-jvmcap-20261108/deps"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 RAY_ADDRESS=local
export VLLM_ATTENTION_BACKEND=FLASH_ATTN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export JAVA_TOOL_OPTIONS="-Xmx128m -Xms16m -XX:+UseSerialGC"
export VLLM_CACHE_ROOT="/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-r2-jvmcap-20261108/vllm_cache" TMPDIR=/tmp/ws-gradus-h800-r2-20261108-driver
mkdir -p "$VLLM_CACHE_ROOT" "$TMPDIR"
