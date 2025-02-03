
CLUSTER="ai2/jupiter*"
PRIORITY="high"

export BEAKER_EXPERIMENT_NAME="Contriever-search"

gantry run \
    --task-name "Contriever-search" \
    --description "Search for dense retrieval" \
    --allow-dirty \
    --workspace ai2/oe-data \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 1 \
    --replicas 24 \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --shared-memory 10GiB \
    --weka oe-data-default:/data \
    --yes \
    -- python -m ric.main_ric --config-name example_config tasks.eval.search=true
