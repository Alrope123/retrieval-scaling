
CLUSTER="ai2/jupiter-cirrascale-2"
PRIORITY="urgent"

export BEAKER_EXPERIMENT_NAME="Contriever-add_to_index"

gantry run \
    --task-name "Contriever-add_to_index-$1" \
    --description "Adding to index for dense retrieval $1" \
    --allow-dirty \
    --workspace ai2/OLMo-modular \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 0 \
    --preemptible \
    --replicas 1 \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --memory 900GiB \
    --weka oe-data-default:/weka_data \
    --install "conda install -c pytorch -c nvidia faiss-gpu=1.8.0 && pip install omegaconf hydra-core tqdm transformers sentence_transformers pyserini datasketch boto3 smart_open s3fs" \
    --yes \
    -- python -m ric.main_ric --config-name $1 tasks.datastore.add_to_index=true

# --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
#     --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
#     --env-secret WANDB_API_KEY=WANDB_API_KEY \