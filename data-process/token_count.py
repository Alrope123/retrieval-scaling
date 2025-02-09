import os
import zstandard
import json
import io
import numpy as np

from tqdm import tqdm
from dolma.core.paths import glob_path
from multiprocessing import Pool
import smart_open

NUM_SHARDS = 10

def count_file(file_name):
    n_docs, n_words = 0, 0
    with smart_open.open(file_name, 'rb') as fh:
        for line in fh:
            dp = json.loads(line)
            n_docs += 1
            n_words += len(dp["text"].split())
    return n_docs, n_words

def shard_file(file_pair):
    file_name, out_paths = file_pair
    
    # 62M blocks for shard 0-9
    # 12M blocks for shard 00-49
    assert len(out_paths)==NUM_SHARDS

    outs = [[] for _ in out_paths]
    with smart_open.open(file_name, 'r') as fh:
        for line in fh:
            idx = np.random.randint(len(out_paths))
            outs[idx].append(line)

    for lines, out_path in zip(outs, out_paths):
        with smart_open.open(out_path, "w") as f:
            for line in lines:
                f.write(line)

def main():
    #main_dclm()
    main_lb()

def main_lb():
    base_path = "s3://ai2-lucas-archival/pretraining-data/sources/libgen/lb_v1p0"
    data_paths = list(glob_path(base_path))
    print (len(data_paths))
    
    out_dir = "s3://ai2-lucas-archival/pretraining-data/sources/libgen/lb_v1p0_uniformly_sharded"
    out_paths = [[os.path.join(out_dir, str(shard_idx).zfill(2), "lb_v1p0-{}.jsonl.gz".format(str(file_idx).zfill(4))) for shard_idx in range(NUM_SHARDS)] for file_idx in range(len(data_paths))]
    with Pool() as p:
        with tqdm(total=len(data_paths), desc="Processing Files", smoothing=0) as pgr:
            for _ in p.imap_unordered(shard_file, zip(data_paths, out_paths)):
                pgr.update(1)

def main_dclm():
    base_path = "s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile_fw3/documents/*/*.json.zst"
    #base_path = "s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft1percentile/documents/*.json.zst"
    print (base_path)

    data_paths = list(glob_path(base_path))
    print (len(data_paths))
    
    out_dir = "s3://ai2-llm/pretraining-data/sources/ds-olmo-data/dclm_ft7percentile_fw3"

    # count n_docs, n_words
    '''
    n_docs, n_words = 0, 0
    with Pool() as p:
        with tqdm(total=len(data_paths), desc="Processing Files", smoothing=0) as pgr:
            for _n_docs, _n_words in p.imap_unordered(count_file, data_paths):
                n_docs += _n_docs
                n_words += _n_words
                pgr.update(1)
    print (n_docs, n_words)
    '''

    # shard paths
    out_paths = [[os.path.join(out_dir, str(shard_idx).zfill(2), "dclm-{}.jsonl.gz".format(str(file_idx).zfill(4))) for shard_idx in range(NUM_SHARDS)] for file_idx in range(len(data_paths))]
    with Pool() as p:
        with tqdm(total=len(data_paths), desc="Processing Files", smoothing=0) as pgr:
            for _ in p.imap_unordered(shard_file, zip(data_paths, out_paths)):
                pgr.update(1)
 

if __name__=='__main__':
    main()
