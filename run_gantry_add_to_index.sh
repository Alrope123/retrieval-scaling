
CLUSTER="ai2/jupiter*"
PRIORITY="high"

export BEAKER_EXPERIMENT_NAME="Contriever-add_to_index"

gantry run \
    --task-name "Contriever-add_to_index-$1" \
    --description "Adding to index for dense retrieval $1" \
    --allow-dirty \
    --workspace ai2/ds-olmo \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 0 \
    --not-preemptible \
    --memory 1900GiB \
    --replicas 1 \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --shared-memory 10GiB \
    --weka oe-data-default:/weka_data \
    --yes \
    -- python -m ric.main_ric --config-name $1 tasks.datastore.add_to_index=true
