
CLUSTER="ai2/jupiter*,ai2/saturn*,ai2/neptune*"
PRIORITY="high"

export BEAKER_EXPERIMENT_NAME="Contriever-search"

config_name=dclm_ft7percentile_fw3_dense_retrieval 

INPUT_DIR=/data/input/sewonm/retrieval-scaling/examples
RETRIEVED_FILE=/data/input/sewonm/dense-retrieval/dclm_ft7percentile_fw3_shard00/retrieved_results/facebook/contriever-msmarco/0_datastore-256_chunk_size/top_3/mmlu_retrieved_results.jsonl  # where retrieved documents are saved

#pip install -e rag-evaluation-harness

#lm_eval --tasks "mmlu" --inputs_save_dir $INPUT_DIR --save_inputs_only

command='PYTHONPATH=.  python ric/main_ric.py --config-name $config_name \
	tasks.eval.task_name=lm-eval \
	tasks.eval.search=true \
	evaluation.domain=mmlu \
	evaluation.search.n_docs=3'

lm_eval --model hf \
	--model_args pretrained="meta-llama/Llama-3.1-8B" \
	--tasks mmlu \
	--batch_size auto \
	--inputs_save_dir $INPUT_DIR \
	--retrieval_file $RETRIEVED_FILE \
	--concat_k 3 \
	--num_fewshot 5 \
	--results_only_save_path out/mmlu_contriever_llama8B_top3_5shot.jsonl

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
    --shared-memory 10GiB \
    --weka oe-data-default:/data \
    --yes \
    -- $command
    
