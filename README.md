# CompactDS
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

This repository contains the codes for building and obtaining the retrieval results from the datastore in [Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks](TODO).

Refer to [Private-Retrieval](TODO) for running evaluations using the retrieval results.

### Citation
```
TODO
```

### Announcement
**xx /xx**: We officially relase the index and the code for CompactDS.

## Installation
To create a conda environment `scaling` with Python 3.11:
```python
conda env create -f environment.yml
conda activate scaling
huggingface-cli login --token <your_hf_token> # ignore if use custom data
```

## Quick Start
- Download the pre-built vectors and raw passages for PeS2o, and search queries for MMLU.
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/pes2o datastores/pes2o # vectors and passage
aws s3 cp s3://ai2-llm/pretraining-data/sources/ds-olmo-data/queries/mmlu:mc::retrieval_q.jsonl queries # search queries
```

- To build an faiss index with pre-embeded PeS2o vectors and raw passages:
```bash
python -m src.main_ric \
    --config-name pes2o_deduped_dense_retrieval.yaml \
    tasks.datastore.index=true \
    datastore.embedding.passages_dir=datastores/pes2o/passages \
    datastore.index.passages_embeddings=datastores/pes2o/embeddings/facebook/contriever-msmarco/pes2o/*.pkl
```

- To search with the queries for MMLU with the built index:
```bash
python -m src.main_ric \
    --config-name pes2o_deduped_dense_retrieval.yaml \
    tasks.eval.task_name=lm-eval \
    tasks.eval.search=true \
    evaluation.data.eval_data=queries/mmlu:mc::retrieval_q.jsonl
```

## Step 0: Configuration and Command Format
We define all the parameters in `ric/conf/*.yaml` files. At the runtime, you will specify the name of the config file with `--config-name`. 

As an alternative to directly modify the config files, you can also specify the specific parameters in the cli command (e.g., `evaluation.data.eval_data=queries/mmlu:mc::retrieval_q.jsonl`). 
<!-- The files with pattern `ric/conf/*_deduped_dense_retrieval.yaml` are the ones we used to build indices (e.g. `pes2o_deduped_dense_retrieval.yaml`). -->

Therefore, the command for any of the following steps will be:
```bash
python -m src.main_ric \
    --config-name <config_name> \
    xxx.yyy=zzz \
    ...
```

Refer to the existing config files for the default settings for our datastores.


## Step 1: Vector Building (Optional)
To build an datastore, the raw text from data sources are required to be chuncked into passages and embeded into a vector space. **Skip this step if you would like to use our pre-built index instead.**

## To build vectors for a single data source in CompactDS
Download the raw text data of a single data sources:
```bash
python scripts/download.py --dataset_name alrope/compactds_raw_text --download_path raw_data --subfolder_path pes2o
```
To build the vectors for a single data source:
```bash
python -m src.main_ric \
    --config-name pes2o \
    tasks.datastore.embedding=true \
    datastore.raw_data_path=raw_data/pes2o \
    datastore.embedding.output_dir=datastores/pes2o
```

## To build vectors with full CompactDS
Download the raw text data of all 10 data sources:
```bash
python scripts/download.py --dataset_name alrope/compactds_raw_text --download_path raw_data
```

Build the vectors for all 10 data sources:
```bash
bash scripts/build_all_vectors.py raw_data
```

#### Important Parameters for Customization
- `model.datastore_encoder`, `model.datastore_tokenizer`, `query_encoder`, `query_tokenizer`: the models / tokenizers used for text embedding. These parameters should all be the same in most cases.
- `datastore.domain`: the customized name of the datastore.
- `datastore.raw_data_path`: the path to the directoy that contains the raw data. The path needs to contain jsonl files (can be compressed) where each data point contains a field `text`.
- `datastore.chunk_size`: the number of words to embed into a vector.
- `datastore.embedding.datastore.no_fp16`: use compactds precision if set to True. 
- `datastore.embedding.per_gpu_batch_size`: batch size for embedding.
- `datastore.embedding.output_dir`: path to the output dir.

### To use built vectors
To download the built vectors and raw passages used for CompactDS:
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/ datastore
```
File structure:
```
s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/
    high-quality_cc/
        embeddings/ # contains vector files
        passages/ # contain passage files
    pes2o/
        .../
        .../
    ....
```
Alternatively, obtain the files for a single-source datastore (e.g., PeS2o) by downloading the corresponding subdirectory only:
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/pes2o datastores/pes2o
```

## Step 2: Index Preparation
We use [Faiss](https://github.com/facebookresearch/faiss/tree/main) to build the index. To make it feasible to deploy the datastore of huge sizes with conventional RAM limit, we used IVFPQ (Inverted File Product Quantization) indices in our paper.

### To build the index for single-source datastore
To build the index for a single-source datastore (e.g. PeS2o):
```bash
python -m src.main_ric \
    --config-name pes2o \
    tasks.datastore.index=true \
    datastore.embedding.embedding_dir=datastores/pes2o \
    datastore.embedding.passages_dir=datastores/pes2o/passages
```
### To build the index for full CompactDS
The vectors and passages need to be aggregated in to the same directories. In order to do that, we create symbolic link for vectors from all data sources under `datastores` into `datastores/compactds`:
```bash
bash create_symlink_vectors.sh datastores datastores/compactds
bash create_symlink_passages.sh datastores datastores/compactds
```

Now, run:
```bash
python -m src.main_ric \
    --config-name CompactDS \
    tasks.datastore.index=true \
    datastore.embedding.embedding_dir=datastores/compactds \
    datastore.embedding.passages_dir=datastores/compactds/passages
```

#### Important Parameters for Customization
- `datastore.embedding.embedding_dir`: path to the vector files.
- `datastore.embedding.passages_dir`: path to directory that contains raw passage files.
- `datastore.index.index_type`: index type. We use `IVFPQ` for our paper. Alternatively, setting it to `Flat` will build an index for exact search without approximation.
- `datastore.index.ncentroids`: number of clusters. Theoretically it is positively correlated with build speed and negatively correlated with search speed. The recommand value is $4\sqrt{n vectors}$ to $8\sqrt{n vectors}$.
- `datastore.index.n_subquantizers`: number of quantizer.  Theoretically it is positively correlated with precision and resulting index size (linearly).
- `datastore.index.sample_train_size`: number of the sample size for training the index. The recommanded value is 1% - 10% of the total number of vectors.
- `datastore.index.n_bits`: number of bits per subquantizer for compression.
- `datastore.index.save_intermediate_index`: will save an intermediate index after adding the vectors from each domain if set to True.
- `datastore.index.deprioritized_domains`: list of domains that will be added last during index building.

### To use the pre-built index for full CompactDS
To download the index we built for CompactDS:
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/built_index compactds/embeddings
```
You still need to aggregate the raw passages under one single directory via symbolic link to build the map between index positions to passages. To aggregate the raw passages, run:
```bash
bash create_symlink_passages.sh datastores datastores/compactds
```
Then build the position to passages map:
```bash
python -m src.main_ric \
    --config-name  \
    tasks.datastore.index=true \
    datastore.embedding.embedding_dir=datastores/compactds \
    datastore.embedding.passages_dir=datastores/compactds/passages
```

## Step 3: Search on the index with queries

### To run prepared search queries
Download the search queries for the five datasets--MMLU, MMLU Pro, AGI Eval, GPQA, and Minerva Math--that we included the paper:
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/queries queries
```

Now, to run retrieval on MMLU queries:
```bash
python -m src.main_ric \
    --config-name CompactDS \
    tasks.eval.search=true \
    tasks.eval.task_name=lm-eval \ # Optional
    evaluation.data.eval_data=queries/minerva_math::retrieval_q.jsonl \ # Optional 
    evaluation.search.n_docs=1000
```

### To custom search queries
To search with custom queries, with `tasks.eval.task_name=lm-eval` the search queries are expected to be in a jsonl file with the following format:
```
{"query": xxx...}
{"query": xxx...}
{"query": xxx...}
```
where each json object should contains a field `text` whose value is a query. Any other field will be perserved in the result file. 

Alternatively, change `tasks.eval.task_name` to support different file format. See details in `load_eval_data()` in [src/data.py](src/data.py).

### Important Parameters
- `evaluation.data.eval_data`: the path to the query file.
- `tasks.eval.task_name`: used to specify the function to load the query file in `src.data.load_eval_data()`.
- `evaluation.search.n_docs`: number of relevant documents to retriever for each query. 
- `evaluation.search.probe`: number of probes. Theoretical it's positively correlated with precision and search speed.