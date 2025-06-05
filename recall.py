import os
import json
import numpy as np
from collections import defaultdict


file1 = "/home/ubuntu/Jinjian/retrieval-scaling/data/gpqa:0shot_cot::retrieval_q_retrieved_results_flat.jsonl"
file2 = "/home/ubuntu/Jinjian/retrieval-scaling/faiss_ivfpq_search_603_short_sub_16384_IVFPQ.16384.768.64/gpqa:0shot_cot::retrieval_q_retrieved_results.jsonl"
#file2 = "top_100_IVFFlat.2048.256/triviaqa_retrieved_results.jsonl"


with open(file1, "r") as f:
    lines1 = f.readlines()
with open(file2, "r") as f:
    lines2 = f.readlines()

assert len(lines1)==len(lines2)

matches = defaultdict(list)
for line1, line2 in zip(lines1, lines2):
    dp1 = json.loads(line1)
    dp2 = json.loads(line2)
    assert dp1["question"]==dp2["question"]
    
    for k in [3, 5, 10, 20, 100]:
        texts1 = set([p["retrieval text"] for p in dp1["ctxs"][:k]])
        texts2 = set([p["retrieval text"] for p in dp2["ctxs"][:k]])
        matches[k].append(len(texts1 & texts2) / len(texts1))

for k, _matches in matches.items():
    print (k, np.mean(_matches))