import os
import pickle
import json
from tqdm import tqdm
import re
import glob


def get_index_dir_and_embedding_paths(cfg, index_shard_ids=None):
    embedding_args = cfg.datastore.embedding
    index_args = cfg.datastore.index
    index_type = cfg.datastore.index.index_type

    # index passages
    index_shard_ids = index_shard_ids if index_shard_ids is not None else index_args.get('index_shard_ids', None)
    
    if index_shard_ids:
        index_shard_ids = [int(i) for i in sorted(index_shard_ids)]
        embedding_paths = [os.path.join(embedding_args.embedding_dir, embedding_args.prefix + f"_{shard_id:02d}.pkl")
                       for shard_id in index_shard_ids]

        # name the index dir with all shard ids included in this index, i.e., one index for multiple passage shards
        index_dir_name = '_'.join([str(shard_id) for shard_id in sorted(index_shard_ids)])
        index_dir = os.path.join(os.path.dirname(embedding_paths[0]), f'index_{index_type}/{index_dir_name}')
        
    else:
        embedding_paths = glob.glob(index_args.passages_embeddings)
        print(f"passages_embeddings: {index_args.passages_embeddings}")
        # put some domains to the back
        deprioritized_domains = ["massiveds-rpj_arxiv", "massiveds-rpj_github", "massiveds-rpj_book", "lb_full"]
        deprioritized_domains_index = {domain: i+1 for i, domain in enumerate(deprioritized_domains)}
        def sort_func(x):
            domain = x.split("/")[-1].split(f'{embedding_args.prefix}')[0].split('--')[0] 
            rank, shard_idx = x.split("/")[-1].split(f'{embedding_args.prefix}')[-1].split(".pkl")[0].split("_")
            if domain not in deprioritized_domains_index:
                depriortized = 0
            else:
                depriortized = deprioritized_domains_index[domain]
            return depriortized, domain, int(rank), int(shard_idx)
        embedding_paths = sorted(embedding_paths, key=sort_func)
        print("DEBUG: sorted embedding paths:")
        print("\n".join(embedding_paths))
        embedding_paths = embedding_paths if index_args.num_subsampled_embedding_files == -1 else embedding_paths[0:index_args.num_subsampled_embedding_files]
        
        index_dir = os.path.join(os.path.dirname(embedding_paths[0]), f'index_{index_type}')
    
    return index_dir, embedding_paths


def convert_pkl_to_jsonl(passage_dir):
    if os.path.isdir(passage_dir):
        filenames = os.listdir(passage_dir)
        pkl_files = [filename for filename in filenames if '.pkl' in filename]
        jsonl_files = [filename for filename in filenames if '.jsonl' in filename]
        print (f"Found {len(pkl_files)} pkl files and {len(jsonl_files)} jsonl files under {passage_dir}")
        if len(pkl_files)==len(jsonl_files):
            return
        print(f"Converting passages to JSONL data format: {passage_dir}")
    elif os.path.isfile(passage_dir):
        assert '.pkl' in passage_dir
        pkl_files = [passage_dir]
    else:
        print(f"{passage_dir} does not exist or is neither a file nor a directory.")
        raise AssertionError
        
    for file in tqdm(pkl_files):
        
        # Create the JSONL file name
        file_path = os.path.join(passage_dir, file)
        jsonl_file = file_path.replace('.pkl', '.jsonl')
        
        if os.path.exists(jsonl_file):
            continue
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Length of {file}: {len(data)}")

        # Save the data to the JSONL file
        with open(jsonl_file, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')
    print("All pickle files have been converted to JSONL files.")

def get_passage_pos_ids(passage_dir, pos_map_save_path):
    # DEBUG
    # if os.path.exists(pos_map_save_path):
    #     with open(pos_map_save_path, 'rb') as f:
    #         pos_id_map = pickle.load(f)
    #     return pos_id_map

    if os.path.isdir(passage_dir):
        filenames = os.listdir(passage_dir)
        jsonl_files = [filename for filename in filenames if '.jsonl' in filename]
        
        # raw_passages_{i}-{j}-of-{tot}.jsonl
        deprioritized_domains = ["massiveds-rpj_arxiv", "massiveds-rpj_github", "massiveds-rpj_book", "lb_full"]
        deprioritized_domains_index = {domain: i+1 for i, domain in enumerate(deprioritized_domains)}
        def sort_func(x):
            domain = x.split("/")[-1].split(f'raw_passages')[0].split('--')[0] 
            rank = int(x.split('passages_')[-1].split("-")[0])
            shard_idx = int(x.split("-of-")[0].split("-")[-1])
            if domain not in deprioritized_domains_index:
                depriortized = 0
            else:
                depriortized = deprioritized_domains_index[domain]
            return depriortized, domain, int(rank), int(shard_idx)
        jsonl_files = sorted(
            jsonl_files,
            key=sort_func)

        pos_id_map = {}
        print(f"DEBUG: Generating id2pos for {passage_dir}")
        print(f"Sorted JSONL files: {jsonl_files}")
        total = 0
        for shard_id, filename in enumerate(tqdm(jsonl_files)):
            #match = re.match(r"raw_passages-(\d+)-of-\d+\.jsonl", filename)
            #shard_id = int(match.group(1))
            file_path = os.path.join(passage_dir, filename)
            
            file_pos_id_map = {}
            with open(file_path, 'r') as file:
                position = file.tell()
                line = file.readline()
                doc_id = 0
                while line:
                    file_pos_id_map[doc_id] = [file_path, position]
                    doc_id += 1
                    position = file.tell()
                    line = file.readline()
            total += doc_id - 1
            # print(f"DEBUG: Building id2pos for {filename}: added {doc_id-1}, result in {total}")
            pos_id_map[shard_id] = file_pos_id_map
    
    elif os.path.isfile(passage_dir):
        # NOTE: deprecated feature, will be removed in future release.
        assert '.pkl' in passage_dir and os.path.exists(passage_dir.replace('.pkl', '.jsonl'))
        match = re.search(r"-(\d+)-of-\d+\.pkl", passage_dir)
        assert match, f"Cannot extract shard_id from {passage_dir}"
        shard_id = int(match.group(1))
        
        pos_id_map = {}
        file_path = passage_dir.replace('.pkl', '.jsonl')
        print(f"Generating id2pos for {file_path}")
        
        file_pos_id_map = {}
        with open(file_path, 'r') as file:
            position = file.tell()
            line = file.readline()
            doc_id = 0
            while line:
                file_pos_id_map[doc_id] = [file_path, position]
                doc_id += 1
                position = file.tell()
                line = file.readline()
        pos_id_map[shard_id] = file_pos_id_map
    
    else:
        print(f"{passage_dir} does not exist or is neither a file nor a directory.")
        raise AssertionError
    
    
    # Save the output map to a pickle file
    if pos_map_save_path is not None:
        with open(pos_map_save_path, 'wb') as f:
            pickle.dump(pos_id_map, f)
        print(f"Output map saved to {pos_map_save_path}")
    return pos_id_map
    


if __name__ == '__main__':
    passage_dir = '/gscratch/zlab/rulins/data/scaling_out/passages/pes2o_v3/16-shards'
    convert_pkl_to_jsonl(passage_dir)
    
    pos_map_save_path = '/gscratch/zlab/rulins/data/scaling_out/passages/pes2o_v3/16-shards/pos_map.pkl'
    get_passage_pos_ids(passage_dir, pos_map_save_path)
