import json

# Input file path
#input_path = "/home/ubuntu/Jinjian/retrieval-scaling/data/retrieved_results/facebook/contriever-msmarco/s2orc_datastore-256_chunk_size/top_100_IVFPQ.2048.256.256/gpqa:0shot_cot::retrieval_q_retrieved_results.jsonl"
#input_path = "/home/ubuntu/Jinjian/retrieval-scaling/data2/retrieved_results/facebook/contriever-msmarco/s2orc_datastore-256_chunk_size/top_100_IVFPQ.2048.256.256/gpqa:0shot_cot::retrieval_q_retrieved_results.jsonl"
input_path = "/home/ubuntu/Jinjian/retrieval-scaling/faiss_flat_search/gpqa:0shot_cot::retrieval_q_retrieved_results.jsonl"
# Output file path
output_path = "entry_preview_faiss.txt"

def format_entry(entry, index):
    lines = [f"\nEntry {index + 1}:"]
    for key, value in entry.items():
        preview = value
        if isinstance(value, str) and len(value) > 150:
            preview = value[:150] + "..."
        elif isinstance(value, list) and len(value) > 3:
            preview = value[:3] + ["..."]
        lines.append(f"  {key}: {preview}")
    return "\n".join(lines)

def main():
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            if i >= 10:
                break
            entry = json.loads(line)
            formatted = format_entry(entry, i)
            outfile.write(formatted + "\n")

if __name__ == "__main__":
    main()
