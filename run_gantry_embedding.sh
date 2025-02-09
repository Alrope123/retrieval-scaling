
CLUSTER="ai2/jupiter*"
PRIORITY="high"

export BEAKER_EXPERIMENT_NAME="Contriever-embedding"

gantry run \
    --task-name "Contriever-embedding" \
    --description "Embed docs for dense retrieval" \
    --allow-dirty \
    --workspace ai2/ds-olmo \
    --beaker-image 'lucas/refine1' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 1 \
    --replicas 8 \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=SEWONM_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=SEWONM_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=SEWONM_WANDB_API_KEY \
    --install "pip install necessary platformdirs>=4.2.0 smart-open fsspec>=2023.6.0" \
    --shared-memory 10GiB \
    --weka oe-data-default:/data \
    --yes \
    -- python -m ric.main_ric --config-name lb_sf tasks.datastore.embedding=true


#    --beaker-image 'petew/olmo-torch23-gantry' \
 
