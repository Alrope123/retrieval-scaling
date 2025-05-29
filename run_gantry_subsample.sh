CLUSTER="ai2/jupiter-cirrascale-2"
PRIORITY="normal"

export BEAKER_EXPERIMENT_NAME="Contriever-search"

command="python ric/sample_offline.py --embed_paths $1/*.pkl --output_path $1/embeddings_sampled_10000000.pkl"

gantry run \
    --task-name "Subsampling" \
    --description "Subsampling" \
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
    --replicas 1 \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --not-preemptible \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret HF_TOKEN=SEWONM_HF_TOKEN \
    --install "conda install -c pytorch -c nvidia faiss-gpu=1.8.0 && pip install omegaconf hydra-core tqdm transformers sentence_transformers pyserini datasketch boto3 smart_open s3fs necessary platformdirs>=4.2.0 smart-open fsspec>=2023.6.0 && pip install necessary && pip install --force-reinstall -U --no-deps -v gritlm && pip install --force-reinstall -U --no-deps -v gritlm" \
    --weka oe-data-default:/weka_data \
    --yes \
    -- $command
    
    # --memory 1900GiB \
    # --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    # --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
    # --env-secret WANDB_API_KEY=WANDB_API_KEY \
    # 