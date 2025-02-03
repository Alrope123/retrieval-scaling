# PYTHONPATH=. python3.11 src/embed.py --model_name_or_path contriever

# PYTHONPATH=.  python ric/main_ric.py --config-name example_config

# BEAKER_REPLICA_RANK=1 BEAKER_REPLICA_COUNT=2 python -m ric.main_ric --config-name example_config

# BEAKER_REPLICA_RANK=0 BEAKER_REPLICA_COUNT=2 python -m ric.main_ric --config-name reddit_dense_retrieval tasks.datastore.embedding=true

# BEAKER_REPLICA_RANK=1 BEAKER_REPLICA_COUNT=2 python -m ric.main_ric --config-name reddit_dense_retrieval tasks.datastore.embedding=true

# python -m ric.main_ric --config-name reddit_dense_retrieval tasks.datastore.index=true

BEAKER_REPLICA_RANK=0 BEAKER_REPLICA_COUNT=1 python -m ric.main_ric --config-name reddit_dense_retrieval tasks.eval.search=true

# export PYTHONPATH=.

# torchrun \
#     --nnodes=1 \
#     --nproc-per-node=4 \
#     -m ric.main_ric --config-name example_config 


