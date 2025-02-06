
CLUSTER="ai2/jupiter*"
PRIORITY="high"

export BEAKER_EXPERIMENT_NAME="Contriever-search"

#config_name=dclm_ft7percentile_fw3_dense_retrieval 
#config_name=dclm_ft7percentile_fw3_gtr
#config_name=dclm_ft7percentile_fw3_e5
config_name=dclm_ft7percentile_fw3_sf

INPUT_DIR=/data/input/sewonm/retrieval-scaling/examples
RETRIEVED_FILE=/data/input/sewonm/dense-retrieval/dclm_ft7percentile_fw3_shard00/retrieved_results/facebook/contriever-msmarco/0_datastore-256_chunk_size/top_5/cot_queries_retrieved_results.jsonl

command="python -m ric.main_ric --config-name $config_name tasks.eval.task_name=lm-eval tasks.eval.search=true"

gantry run \
    --task-name "Contriever-search" \
    --description "Search for dense retrieval" \
    --allow-dirty \
    --workspace ai2/ds-olmo \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 1 \
    --replicas 1 \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --env-secret HF_TOKEN=SEWONM_HF_TOKEN \
    --install "pip install necessary" \
    --shared-memory 10GiB \
    --weka oe-data-default:/data \
    --yes \
    -- $command
    
