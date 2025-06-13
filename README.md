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
### Set up CompactDS
- Download the index we built for CompactDS:
```bash
python scripts/download_index.py --output_path datastores/compactds
```
- Build the index positions to passages map for your local file system:
```bash
python -m src.main_ric \
    --config-name  \
    tasks.datastore.index=true \
    datastore.embedding.embedding_dir=datastores/compactds \
    datastore.embedding.passages_dir=datastores/compactds/passages
```

### Run Retreval
- Download the search queries we included in the paper for the five datasets: MMLU, MMLU Pro, AGI Eval, GPQA, and Minerva Math.
```bash
python scripts/download_queries.py --output_path queries
```
- Now, to obtain the top 1000 documents for each MMLU Pro query from CompactDS:
```bash
python -m src.main_ric \
    --config-name CompactDS \
    tasks.eval.search=true \
    tasks.eval.task_name=lm-eval \ # Optional
    evaluation.data.eval_data=queries/mmlu_pro.jsonl \ # Optional 
    evaluation.search.n_docs=1000
```

## Custom Index and Search 
### Step 0: Configuration and Command Format
- We define all the parameters in `ric/conf/*.yaml` files. At the runtime, you will specify the name of the config file with `--config-name`. 

- As an alternative to directly modify the config files, you can also specify the specific parameters in the cli command (e.g., `evaluation.data.eval_data=queries/mmlu:mc::retrieval_q.jsonl`). 
<!-- The files with pattern `ric/conf/*_deduped_dense_retrieval.yaml` are the ones we used to build indices (e.g. `pes2o_deduped_dense_retrieval.yaml`). -->

- Therefore, the command for any of the following steps will be:
```bash
python -m src.main_ric \
    --config-name <config_name> \
    xxx.yyy=zzz \
    ...
```

- Refer to the existing config files for the default settings for our datastores.


### Step 1: Vector Building
To build an datastore, the raw text from data sources are required to be chuncked into passages and embeded into a vector space. **Skip this step if you would like to use our pre-built index instead.**

#### To build vectors for a single data source (e.g., PeS2o)
- Download the raw text data:
```bash
python scripts/download_raw_data.py --output_path raw_data --subfolder_path pes2o
```
- To build the vectors:
```bash
python -m src.main_ric \
    --config-name pes2o \
    tasks.datastore.embedding=true \
    datastore.raw_data_path=raw_data/pes2o \
    datastore.embedding.output_dir=datastores/pes2o
```

#### To build vectors with multiple data source (e.g., full CompactDS)
- Download the raw text data of all 10 data sources:
```bash
python scripts/download_raw_data.py --output_path raw_data
```

- Build the vectors for all 10 data sources:
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


### Step 2: Build the Index
We use [Faiss](https://github.com/facebookresearch/faiss/tree/main) to build the index. To make it feasible to deploy the datastore of huge sizes with conventional RAM limit, we used IVFPQ (Inverted File Product Quantization) indices in our paper.

#### To build the index for single-source datastore (e.g., PeS2o)
- Run:
```bash
python -m src.main_ric \
    --config-name pes2o \
    tasks.datastore.index=true \
    datastore.embedding.embedding_dir=datastores/pes2o \
    datastore.embedding.passages_dir=datastores/pes2o/passages
```
#### To build the index for multiple-source datastore (e.g., full CompactDS) 
- The vectors and passages need to be aggregated in to the same directories. 
- In order to do that, we create symbolic link for vectors from all 10 data sources under `datastores` into `datastores/compactds`:
```bash
bash create_symlink_vectors.sh datastores datastores/compactds
bash create_symlink_passages.sh datastores datastores/compactds
```

- Now, run:
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


### Step 3: Search with Queries
- To search with custom queries, with `tasks.eval.task_name=lm-eval` the search queries are expected to be in a jsonl file (e.g., `your_queries.jsonl`) with the following format:
```
{"query": xxx...}
{"query": xxx...}
{"query": xxx...}
```where each json object should contains a field `text` whose value is a query. Any other field will be perserved in the result file. 

- Alternatively, change `tasks.eval.task_name` to support different file format. See details in `load_eval_data()` in [src/data.py](src/data.py).

- Now, run:
```bash
python -m src.main_ric \
    --config-name CompactDS \
    tasks.eval.search=true \
    tasks.eval.task_name=lm-eval \
    evaluation.data.eval_data=your_queries \ 
    evaluation.search.n_docs=1000
```

#### Important Parameters
- `evaluation.data.eval_data`: the path to the query file.
- `tasks.eval.task_name`: used to specify the function to load the query file in `src.data.load_eval_data()`.
- `evaluation.search.n_docs`: number of relevant documents to retriever for each query. 
- `evaluation.search.probe`: number of probes. Theoretical it's positively correlated with precision and search speed.