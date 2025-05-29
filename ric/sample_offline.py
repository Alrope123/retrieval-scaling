import pickle
import numpy as np
import random
import argparse

def subsample(sample_size, embed_paths, output_path):
    """
    Subsample a given number of embeddings from multiple pickle files.

    Args:
        sample_size (int): Total number of embeddings to sample.
        embed_paths (list): List of paths to pickle files containing embeddings.

    Returns:
        np.ndarray: Array of sampled embeddings.
    """


    if sample_size <= 0:
        raise ValueError("Sample size must be a positive integer.")
    
    if not embed_paths:
        raise ValueError("Embed paths list cannot be empty.")
    
    np.random.seed(2025)  # For reproducibility

    print(f"Sampling {sample_size} examples from {len(embed_paths)} files...")
    per_shard_sample_size = sample_size // len(embed_paths)
    all_sampled_embs = []
    for embed_path in embed_paths:
        print(f"Loading pickle embedding from {embed_path}...")
        with open(embed_path, "rb") as fin:
            _, embeddings = pickle.load(fin)
        shard_size = len(embeddings)
        print(f"Finished loading, sampling {per_shard_sample_size} from {shard_size} for training...")
        random_samples = np.random.choice(np.arange(shard_size), size=[min(per_shard_sample_size, shard_size)], replace=False)
        sampled_embs = embeddings[random_samples]
        all_sampled_embs.extend(sampled_embs)
    all_sampled_embs = np.stack(all_sampled_embs).astype(np.float32)
    pickle.dump(all_sampled_embs, open(output_path, "wb"))
    print(f"Subsampled embeddings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample embeddings from multiple pickle files.")
    parser.add_argument("--sample_size", type=int, default=10000000, help="Total number of embeddings to sample.")
    parser.add_argument("--embed_paths", nargs='+', required=True, help="List of paths to pickle files containing embeddings.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the subsampled embeddings.")
    
    args = parser.parse_args()
    
    subsample(args.sample_size, args.embed_paths, args.output_path)