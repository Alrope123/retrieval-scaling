
CLUSTER="ai2/jupiter-cirrascale-2"
PRIORITY="urgent"

export BEAKER_EXPERIMENT_NAME="Contriever-embedding"

gantry run \
    --task-name "Contriever-embedding-$1" \
    --description "Embed docs for dense retrieval $1" \
    --allow-dirty \
    --workspace ai2/OLMo-modular \
    --beaker-image 'lucas/refine1' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 1 \
    --replicas 1 \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=SEWONM_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=SEWONM_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=SEWONM_WANDB_API_KEY \
    --preemptible \
    --install "pip install faiss-cpu==1.8.0 omegaconf hydra-core tqdm transformers sentence_transformers pyserini datasketch boto3 smart_open s3fs necessary platformdirs>=4.2.0 smart-open fsspec>=2023.6.0 && pip install --force-reinstall -U --no-deps -v gritlm" \
    --shared-memory 10GiB \
    --weka oe-data-default:/weka_data \
    --weka oe-training-default:/weka_training \
    --yes \
    -- python -m ric.main_ric --config-name $1 tasks.datastore.embedding=true


 
