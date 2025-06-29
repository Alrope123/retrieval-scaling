
CLUSTER="ai2/jupiter-cirrascale-2"
PRIORITY="urgent"

export BEAKER_EXPERIMENT_NAME="Contriever-search"

#config_name=dclm_ft7percentile_fw3_dense_retrieval 
#config_name=dclm_ft7percentile_fw3_gtr
#config_name=dclm_ft7percentile_fw3_e5
#config_name=dclm_ft7percentile_fw3_sf
#config_name=lb_dense_retrieval
#config_name=c4_dense_retrieval

n=4
for ((i=0; i<n; i++)); do
# for i in 0 ; do
    # /weka_data/xinxil/private-retrieval-lm/eval_datasets/mmlu:mc::retrieval_full_q.jsonl
    # /weka_data/xinxil/private-retrieval-lm/eval_datasets/retrieval_v3_q.jsonl
    command="python -m ric.main_ric --config-name $1 tasks.eval.task_name=lm-eval tasks.eval.search=true evaluation.data.eval_data=/weka_data/xinxil/private-retrieval-lm/eval_datasets/mmlu:mc::retrieval_q/mmlu:mc::retrieval_q_${i}.jsonl evaluation.search.n_docs=100"
    # evaluation.data.eval_data=/weka_data/xinxil/private-retrieval-lm/eval_datasets/retrieval_unique.jsonl
    # command="python -m ric.main_ric --config-name $1 tasks.eval.task_name=lm-eval tasks.eval.search=true evaluation.data.eval_data=/weka_data/xinxil/private-retrieval-lm/eval_datasets/minerva_math_500::retrieval_q/minerva_math_500::retrieval_q_${i}.jsonl evaluation.search.n_docs=1000"
    
    gantry run \
        --task-name "Contriever-search-$1-$i" \
        --description "Search for dense retrieval $1-$i" \
        --allow-dirty \
        --workspace ai2/OLMo-modular \
        --beaker-image 'petew/olmo-torch23-gantry' \
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
        --env-secret HF_TOKEN=SEWONM_HF_TOKEN \
        --install "pip install faiss-cpu==1.8.0 omegaconf hydra-core tqdm transformers sentence_transformers pyserini datasketch boto3 smart_open s3fs necessary platformdirs>=4.2.0 smart-open fsspec>=2023.6.0 && pip install necessary && pip install --force-reinstall -U --no-deps -v gritlm && pip install --force-reinstall -U --no-deps -v gritlm" \
        --weka oe-data-default:/weka_data \
        --memory 900GiB \
        --yes \
        --preemptible \
        -- $command
        # --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
        # --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
        # --env-secret WANDB_API_KEY=WANDB_API_KEY \
       
done
