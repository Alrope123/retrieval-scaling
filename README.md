# Dense Retrieval Codebase


## Gantry scripts
There are three scripts for running embedding, indexing, and search via gantry. Current parallelization method assumes 1 GPU per replica, so for all scripts **keep the setting of "--gpus 1"**.

**1. Embedding**: `run_gantry_embedding.sh {config_name}`. (For instance, if your config file is `ric/conf/c4_dense_retrieval.yaml`, do `run_gantry_embedding.sh c4_dense_retrieval`)

This is the first place where we distribute across GPUs, to accelerate the embedding process. To increase parallelization, increase `--replicas`. Note that the parallelization works by partitioning input files across replicas, so this won't be effective if you have very few input files.
```
--gpus 1 \
--replicas 8 \
``` 

**2. Indexing**: `run_gantry_index.sh {config_name}`.
gpus can be 0 or 1.

**3. Search**: `run_gantry_search.sh {config_name}`.
gpus needs to be 1.

By default, it uses MMLU queries as search queries. To change the data, modify `evaluation.data.eval_data` either in the config file or as a command line. See the example MMLU queries data in s3 to see how the input to the search should look like. For multiple datasets, you can specify multiple paths separated by a comma. See `wiki_dense_retrieval.yaml` for a reference, which specify two paths for NQ and TQA.

Search results are saved under `{datastore.embedding.output_dir}/retrieved_results` (see `eval_output_dir` in the config file).` These results can directly be fed into the private-retrieval-lm codebase.



## Configs
All configs are under `ric/conf`. Use `ric/conf/c4_dense_retrieval.yaml` as an example.

**Adding a new data source**: Copy an existing yaml file and change three places: `datastore.raw_data_path`, `datastore.domain`, and `datastore.embedding.output_dir`.

**Ablating the choice of embedder**: Change four places: `model.datastore_tokenizer`, `model.query_tokenizer`, `model.query_encoder`, and `model.datastore_encoder`. Check out four config files under `ric/conf/dclm*.yaml` for the configs for Contriever (`dclm_ft7percentile_fw3_dense_retrieval.yaml`), e5, me5 and GTR. `projection_size` might have to be updated too if the dimension of the embedder is different.

**Ablating the choice of index types**: See three files under `ric/conf/wiki*dense_retrieval.yaml` for examples
- `wiki_dense_retrieval.yaml`: Flat index (exact search). Can just put `index_type` to be "Flat" and other hyperparams don't matter. Most accurate as it is exact search, and the original MassiveDS uses it. However, slow, and uses a lot of RAM. Most data sources are OK with Flat with 1TB RAM, but C4, DCLM, and other comparable sources can't use it. To calculate how much RAM would use, you can use `# of chunks * dimension (768 for Contriever) * 32 bits`.
- `wiki_ivfflat_dense_retrieval.yaml`: Faster than Flat, mostly accurate, but uses the same amount of RAM. Hyperparmas that matter are: `ncentroids` and `probe`. Difference `ncentroids` requires re-building index; `probe` is a hyperparameter that can differ at search time, without needing to re-build the index.
- `wiki_ivfpq_dense_retrieva.yaml`: IVF + compress vectors to reduce RAM use. Less accurate, but the only _implemented_ solution if we want to reduce RAM. Hyperparams that matter are those matter for IVF plus `n_subquentizers`. (`n_bits` matters too but everyone seems to use 8, so would fix it.) To calculate how much RAM it would use, you can do `# of chunks * n_quantizers * 8 bits`. Using 256 as `n_quantizers` offers about 12x compression.

----------------------------------------------------------------------------------

(From previous branch)

This repo is based on the code from paper "Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore".

[[Website](https://retrievalscaling.github.io)][[Paper](https://arxiv.org/abs/2407.12854)]

```
@article{shao2024scaling,
  title={Scaling Retrieval-Based Language Models with a Trillion-Token Datastore},
  author={Shao, Rulin and He, Jacqueline and Asai, Akari and Shi, Weijia and Dettmers, Tim and Min, Sewon and Zettlemoyer, Luke and Koh, Pang Wei},
  journal={arXiv preprint arXiv:2407.12854},
  year={2024}
}
```

# Overview
This codebase contains:

1. Embedding
2. Indexing
3. Search


# Latest Update 🔥
* [2025/01] We support a new hashing-based passage loading method that could load the passage from disk by passage ID **without loading the full passages in RAM**, which will greatly reduce the memory requirement for large datastore serving. Set `datastore.use_passage_pos_id_map=true` to turn it on!
* [2025/01] We now support semantic chunking which chunks the data based on semantic boundaries. Set `datastore.chunking_strategy="semantic"` to give it a try! It takes `datastore.chunk_size` as the maximum number of tokens it should accomodate. 
* [2024/12] We added API serving codes with support for multiple index types, i.e., Flat and IVF_Flat in the current version. Find instructions in `api/` folder.


## Installation
Clone repo and checkout `reddit-exps` branch.

Note from original repo (I didn't try this, though it may be installed on the image we're using): "to accelerate the inference, we recommend users to install flash attention based on their accelerator type following the instructions [here](https://github.com/Dao-AILab/flash-attention)."

## Gantry scripts
There are three scripts for running embedding, indexing, and search via gantry. Current parallelization method assumes 1 GPU per replica, so for all scripts **keep the setting of "--gpus 1"**.

**1. Embedding**: `run_gantry_embedding.sh`. 
This is the first place where we distribute across GPUs, to accelerate the embedding process. To increase parallelization, increase `--replicas`. Note that the parallelization works by partitioning input files across replicas, so this won't be effective if you have very few input files.
```
--gpus 1 \
--replicas 8 \
``` 

**2. Indexing**: `run_gantry_index.sh`.
Not currently set up to distribute across GPUs, so keep as below.
```
--gpus 1 \
--replicas 1 \
```

**3. Search**: `run_gantry_search.sh`.
Was not originally planning to parallelize this step, but I had a large number of files with queries (57 files corresponding to MMLU categories) so I ended up modifying the code to partition those query files across replicas to speed things up. In the current example I use 24 replicas. 
```
--gpus 1 \
--replicas 24 \
```
## Config

Example config file `example_config.yaml` can be found in `ric/conf/`. That is the config the current gantry scripts are pointing to.

Settings that need to be filled in:

- `datastore.raw_data_path`: This should be a directory containing the data to be embedded/indexed/searched. Assumes json/jsonl files with a json dict object per line, containing "text" field with the text to be embedded. Should work for directories on S3 or local, and should work for .gz files. (Not currently set up for glob inputs.)
- `datastore.embedding.output_dir`: This is the top-level location where all outputs (embeddings, passages, index, retrieval outputs) will be written. The gantry scripts mount WEKA (oe-data-default) to /data and I've been writing the outputs there. (S3 ended up causing problems as a location for the index in particular.)

Other things to note:

- `datastore.embedding.max_files_per_shard`: Max number of input files to be included in an embedding file shard. I had very large files so I set this to 1.
- `datastore.embedding.fields_to_add`: Other fields in the input jsons that you want saved with the passages (other than "text"). I used this to keep the "subreddit" field.
- `datastore.index.max_files_per_index_shard`: Max number of embedding files per index shard. 
- `evaluation.data.eval_data`: This expects a path to the query file/files, in glob format.
- `tasks.eval.task_name`: This affects how the query data is processed. For now I've sidestepped what the original code did, and just defined a "gen" setting that uses the input query text raw as the search query.






# Advanced Usage (notes from prior repo)
*I'm leaving this here for now for reference, since some of it may be informative about other settings I didn't mention above -- but please note that **some of these parameters may no longer work/exist** after the changes that have been made in the current repo.*

## Content
1. [Model Configuration](#model-configuration)
2. [Datastore Configuration](#datastore-configuration)
3. [Distributed Datastore Construction](#distributed-datastore-construction)
4. [Distributed Document Retrieval](#distributed-document-retrieval)
5. [Data Filtering](#data-filtering)
6. [Subsampling](#subsampling)
7. [Evaluation](#evaluation)

## Model Configuration
Retrieval-based LMs involve two models: the reader LM and the retriever.
In the below sections, we illustrate ways to configure the models.
### Retriever
The default configuration uses Contriever-MSMACRO as the retriever. Additionally, we support off-the-shelf retriever models implemented in [HuggingFace](https://huggingface.co/models) or [SentenceTransformers](https://sbert.net/). To change the dense retriever, we require the user to define the 4 arguments below properly, as some models may use different encoders for query and context:

```bash
model:
  datastore_tokenizer: ???
  query_tokenizer: ???
  query_encoder: ???
  datastore_encoder: ???
```

**HuggingFace Dense Retrievers**

You can use HuggingFace retrievers by passing its huggingface model name to the corresponding arguments.

For example, to use [DRAGON-RoBERTa](https://huggingface.co/facebook/dragon-roberta-query-encoder) as the retriever, set
```bash
model:
  datastore_tokenizer: facebook/dragon-roberta-query-encoder
  query_tokenizer: facebook/dragon-roberta-query-encoder
  query_encoder: facebook/dragon-roberta-query-encoder
  datastore_encoder: facebook/dragon-roberta-context-encoder
```

**SentenceTransformers Dense Retrievers**

Similarly, if you want to use a dense retriever implemented in sentence-transformers, you can pass its model name to the arguments in the same way, where the script will automatically identify which package to use.

For example, to use [GTR-T5-Base](https://huggingface.co/sentence-transformers/gtr-t5-base) as the retriever, set:

```bash
model:
  datastore_tokenizer: sentence-transformers/gtr-t5-base
  query_tokenizer: sentence-transformers/gtr-t5-base
  query_encoder: sentence-transformers/gtr-t5-base
  datastore_encoder: sentence-transformers/gtr-t5-base
```

**Sparse Retriever**

We also support BM25 as a sparse retriever. To use BM25, set
```bash
model:
  sparse_retriever: bm25
```
Note: once `model.sparse_retriever` is set to bm25, the arguments for dense retrievers will be ignored. 


### Reader LM
We support all HuggingFace decoder models as the reader LM. You can set the reader LM by passing the model name or path to `model.lm_model`:
```bash
model:
  lm_model: EleutherAI/pythia-1b
```
Note: the reader LM for RAG-Evaluation-Harnesses is passed to the `lm-eval` command separately. The LM defined here is only for perplexity evaluation.

## Datastore Configuration

**General Configuration**

There are 3 arguments for datastore general configuration:
```bash
datastore:
  domain: fineweb_edu_1m
  raw_data_path: raw_data/fineweb-edu-1m.jsonl
  chunk_size: 256
```
* `domain` is a string used for naming the paths to save the datastore; 
* `raw_data_path` provides the raw data that will be used to construct the datastore, which could be a single JSONL file or a directory of JSONL files; 
* `chunk_size` is used to split raw text into passages, where every passage will have no more than `chunk_size` natural words.

**Embedding Configuration**

We introduce several key arguments for embedding below. The others are either straightforward by its name or do not need to be changed in general usage.
```bash
datastore:
  embedding:
    shard_ids: [0]
    num_shards: 1
    keep_last_chunk: true
    passages_dir: scaling_out/passages/${datastore.domain}/${datastore.embedding.num_shards}-shards

    per_gpu_batch_size: 512

    prefix: "passages"
    embedding_dir: scaling_out/embeddings/${model.datastore_encoder}/${datastore.domain}/${datastore.embedding.num_shards}-shards
    use_saved_if_exists: true
```
* `num_shards` defines the number of shards that the raw data will be divided into for distributed datastore construction. 
* `shard_ids` speficies the IDs of shards that the current worker is building or using. For example, `[0]` means only the first shard will be embeded or used, and `[0,1,2]` means the 0-2 shards will be embeded or used by the current worker.
* `keep_last_chunk`: if set to `true`, the chunked documents that have less than `datastore.chunk_size` will be included in the datastore; these documents will be discarded if `keep_last_chunk` is set to `false`.
* `per_gpu_batch_size` is the batch size per GPU used when running embedding.
* `prefix` is a prefix naming string used to name the outputs.
* `embedding_dir` is the directory to save the embeddings.
* `use_saved_if_exists`: if set to `true`, the embedding task will be skipped if the embedding exists; it will rerun the embedding and overwrite the exisitng ones if `use_saved_if_exists` is set to `false`.

**Index Configuration**

```bash
datastore:
  index:
    index_shard_ids: [[0]]
    indexing_batch_size: 1000000
    projection_size: 768
    overwrite: false
```
* `index_shard_ids`: IDs of passages included in each index. For example, [0, 1, 2] means build a single index over passages 0-2; [[0], [1,2]] means build an index for passage 0 and another index for passages 1-2. This argument is very useful when you want to index or search over different shards in parallel.
* `indexing_batch_size` defines the batch size used for indexing.
* `projection_size` tells the index the size of embedding.
* `overwrite`: if set to `false`, it will save the index and reuse the index if it exists; if set to `true`, the index will be reconstructed and overwrite the old index.

Note: we currently only support flat index. Please stay tuned for more index types.

## Distributed Datastore Construction
For datastores with over 1B tokens, we recommend using our distributed datastore construction pipeline, which could linearly accelerate your datastore construction. 

Below is an example slurm job script that parallelize the datastore construction using 16 GPUs.
```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint a40|l40
#SBATCH --mem 100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15

PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.datastore.embedding=true \
  tasks.datastore.index=true \
  datastore.embedding.num_shards=16 \
  datastore.embedding.shard_ids=[$SLURM_ARRAY_TASK_ID] \
  datastore.index.index_shard_ids=[[$SLURM_ARRAY_TASK_ID]]
```
The above script splits the raw data into 16 shards and builds embeddings and indices for these shards in parallel.

## Distributed Document Retrieval

Since the datastore is sharded, we can run document retrieval in parallel as well. For example, the below slurm script searches the top-K documents from each shard in parallel.

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint a40|l40
#SBATCH --mem 100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15

PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.eval.task_name=perplexity \
  tasks.eval.search=true \
  datastore.embedding.num_shards=16 \
  datastore.index.index_shard_ids=[[$SLURM_ARRAY_TASK_ID]]
```
When the 16 jobs finish, the top-K documents from each shard are saved separately. Next, we can merge the retrieved documents quickly on a single node:
```bash
index_list="[[0]"
for (( i=1; i<=$((16 - 1)); i++ )); do
index_list+=",[$i]"
done
index_list+="]"

PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.eval.task_name=perplexity \
  tasks.eval.search=true \
  datastore.embedding.num_shards=16 \
  datastore.index.index_shard_ids=$index_list
```

## Data Filtering
We provide functions to run post-hoc data filtering on the retrieved documents to either improve the retrieval quality or remove contamination data.
### Decontamination
We support two decontamination methods: the longest string overlap based decontamination and the Jaccard similarity based decontamination.

**Longest String Overlap Based Decontamination**

The longest string overlap based decontamination method removes a document from the retrieved top-K if it has a continuous overlapped string that exceeds `n` words with the gold answer. You can set it through

```bash
evaluation:
  decontamination: true
  contamination_threshold: 32
  decontamination_method: longest
```
where `n = 32` in the above case.

**Jaccard Similarity Based Decontamination**

Jaccard similarity based decontamination removes a document from the retrieved top-K if it has a 13-gram Jaccard similarity score that is higher than `t` (0 < `t` < 1) with the gold answer. You can set it through

```bash
evaluation:
  decontamination: true
  contamination_threshold: 0.8
  decontamination_method: jaccard
```
where `p = 0.8` in the above case.

### Deduplication
We provide a post-hoc deduplication method to remove the duplicates in the top-K retrieved documents before taking the top-k (k << K) for evaluation. We use MinHash for deduplication, which marks documents as deplicates if they have a 13-gram Jaccard similarity score over 0.8. For the set of documents that are marked duplicates, we keep the one with the highest retrieval score. You can enable it by setting `evaluation.search.post_process_only=true`.

## Subsampling
We also support running post-hoc subsampling on the retrieved top-K documents, which helps us simulate the sitatuions where the raw data is subsampled to a smaller size with a given random seed. See Appendix A in our paper for more details. 

To run subsampling on the retrieved top-K documents, turn on `evaluation.search.post_process_only`. Then set the subsampling probability through `evaluation.search.topk_subsample_p` and the random seed through `evaluation.search.subsample_seed` after running document retrieval.

## Evaluation
In this section, we provide more details of the evaluation. Here are some general evaluation configurations:
```bash
evaluation:
  domain: test_c4
  concate_k: 3
```
where `evaluation.domain` is the name of the evaluation task and `concate_k` is the number of documents that will be prepended in context.

### Perplexity Evaluation
Perplexity evaluation for retrieval-based LMs is different from perplexity evaluation for LMs. Specifically, given an input, the first part, i.e., the first half in our implementation, is used as the query to retrieve relevant documents, which thus cannot be used for perplexity computation. Therefore, only the second part, i.e., the second half, of the input is used to calculate perplexity. The retrieved top-k documents will be prepended in context before the query in a reversed order, i.e., the document with the highest retrieval score is the cloest to the query. The concatenated context is then fed into the LM and we compute the perplexity using the second half of the input as the target. 

Below is an example configuration that first concatenates all tokens in `eval_data` in one sentence using sequence packaging and then chunks the sentence into smaller chunks of `max_eval_data_seq_length` tokens with a stride of `eval_stride`. For each 1024-token chunk, the first 512 tokens are used as query, and the last 512 tokens are used for perplexity calculation.
```bash
evaluation
  data:
    eval_data: examples/test_c4.jsonl
    max_eval_data_seq_length: 1024
    eval_stride: 512
    merge: True
    num_eval_samples: null 
```

### Downstream Evaluation
We developped a package for retrieval-augmented-generation (RAG) evaluation---RAG-Evaluation-Harness. The package supports over 60 standard academic benchmarks and can be easily used with retrieval augmentation as shown in the [Quick Start](#quick-start). We refer the user to the [README of RAG-Evaluation-Harness](https://github.com/RulinShao/retrieval-scaling/blob/main/rag-evaluation-harness/README.md) for more details, such as accelerating the inference with VLLMs, getting a list of supported tasks, and adding new tasks.
