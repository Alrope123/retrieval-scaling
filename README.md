# Dense Retrieval Codebase

## Installation
```python
conda env create -f environment.yml
conda activate scaling
```

## Download Data
### Download vectors and passages
To download the single data source like pes2o:
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/s2orc your/local/directory/s2orc
```

To download the full data:
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/ your/local/directory
```

File structure:
```
s3://ai2-llm/pretraining-data/sources/ds-olmo-data/indices/
    c4+dclm/ # name for high quality source
        embeddings/ # contains vector files
        passages/ # contain passage files
    s2orc/ # name for pes2o
        .../
        .../
    ....
```

### Download search queries
```bash
aws s3 sync s3://ai2-llm/pretraining-data/sources/ds-olmo-data/queries your/local/query_directory
```

## Quick Start
To build an index with pes2o:
```bash
python -m ric.main_ric \
    --config-name s2orc_deduped_dense_retrieval.yaml \
    tasks.datastore.index=true \
    datastore.embedding.passages_dir=your/local/directory/s2orc/passages \
    datastore.index.passages_embeddings your/local/directory/s2orc/embeddings/facebook/contriever-msmarco/s2orc/*.pkl
```

To search with the queries for MMLU with the built index:
```bash
python -m ric.main_ric \
    --config-name s2orc_deduped_dense_retrieval.yaml \
    tasks.eval.task_name=lm-eval \
    tasks.eval.search=true \
    evaluation.data.eval_data=your/local/query_directory/mmlu:mc::retrieval_q.jsonl
```

To evaluate on the search results, refer to this repository: https://github.com/allenai/private-retrieval-lm/tree/main

## Hyperparameters for indexing and search
### Config file
- Config files: We define all the parameters for index building and searching in `ric/conf/*.yaml` files. At the runtime, you will specify the name of the config file (optionally you can specify the specific argument in the cli as well). The files with pattern `ric/conf/*_deduped_dense_retrieval.yaml` are the ones we used to build indices (e.g. `s2orc_deduped_dense_retrieval.yaml`).
- Config file overview: there are parameters for vector building and evaluation in the files as well, but you will not use most of them. Focus on `datastore.index` which contains the parameters for index building and searching. You also want to turn on the boolean values `tasks.datastore.index` when building index and `task.eval.search` when searching (probably though cli instead of modifying the file).

### Important parameters:
- `datastore.index.ncentroids`: number of clusters. Theoretically it is positively correlated with build speed and negatively correlated with search speed.
- `datastore.index.n_subquantizers`: number of quantizer.  Theoretically it is positively correlated with precision and resulting index size (linearly).
- `datastore.search.probe` : number of probes. Theoretical it's positively correlated with precision and search speed.

## Building an index aggregating all data sources
Since our implementation only support one single path for vectors and passages, I recommand  to create directories that contains symbolic links to the vector files / passage files. Run:
```
# Create symbolic link for vectors
mkdir your/local/directory/full/embeddings # your final directory to contains the vectors
for path in your/local/directory/s2orc/embeddings/facebook/contriever-msmarco/s2orc your/local/directory/pubmed/embeddings/...  ... ... ... ;
do
    find $path -name "*passages*.pkl" | while read file; do
        ln -s "$file" your/local/directory/full/embeddings/$(basename "$(dirname "$file")")--$(basename "$file")
    done
done

# Create symbolic link for passages
mkdir your/local/directory/full/passages # your final directory to contains the passages
for path in your/local/directory/s2orc/passages/s2orc your/local/directory/pubmed/passages/pubmed ... ... ...;
    do
    find $path -name "*.pkl" | while read file; do
        ln -s "$file" your/local/directory/full/passages/$(basename "$(dirname "$file")")--$(basename "$file")
    done
done
```
Then follow the steps mentioned above.