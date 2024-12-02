__author__ = "qiao"

"""
Conduct the first stage retrieval by the hybrid retriever with detailed trial information.
"""

from beir.datasets.data_loader import GenericDataLoader
import faiss
import json
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import numpy as np
import os
from rank_bm25 import BM25Okapi
import sys
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel


def get_bm25_corpus_index(corpus):
    corpus_path = os.path.join(f"trialgpt_retrieval/bm25_corpus_{corpus}.json")

    # if already cached then load, otherwise build
    if os.path.exists(corpus_path):
        corpus_data = json.load(open(corpus_path))
        tokenized_corpus = corpus_data["tokenized_corpus"]
        corpus_nctids = corpus_data["corpus_nctids"]
        trial_metadata = corpus_data.get("trial_metadata", {})
    else:
        tokenized_corpus = []
        corpus_nctids = []
        trial_metadata = {}

        with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
            for line in f.readlines():
                entry = json.loads(line)
                corpus_nctids.append(entry["_id"])

                # Save metadata for later inclusion in results
                trial_metadata[entry["_id"]] = {
                    "brief_title": entry["title"],
                    "phase": entry.get("phase", ""),
                    "drugs": entry.get("metadata", {}).get("drugs", "[]"),
                    "drugs_list": entry.get("metadata", {}).get("drugs_list", []),
                    "diseases": entry.get("metadata", {}).get("diseases", "[]"),
                    "diseases_list": entry.get("metadata", {}).get("diseases_list", []),
                    "enrollment": entry.get("metadata", {}).get("enrollment", "Unknown"),
                    "inclusion_criteria": entry.get("inclusion_criteria", ""),
                    "exclusion_criteria": entry.get("exclusion_criteria", ""),
                    "brief_summary": entry.get("text", ""),
                }

                # weighting: 3 * title, 2 * condition, 1 * text
                tokens = word_tokenize(entry["title"].lower()) * 3
                for disease in entry["metadata"]["diseases_list"]:
                    tokens += word_tokenize(disease.lower()) * 2
                tokens += word_tokenize(entry["text"].lower())

                tokenized_corpus.append(tokens)

        corpus_data = {
            "tokenized_corpus": tokenized_corpus,
            "corpus_nctids": corpus_nctids,
            "trial_metadata": trial_metadata,
        }

        with open(corpus_path, "w") as f:
            json.dump(corpus_data, f, indent=4)

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus_nctids, trial_metadata


def get_medcpt_corpus_index(corpus):
    corpus_path = f"trialgpt_retrieval/{corpus}_embeds.npy"
    nctids_path = f"trialgpt_retrieval/{corpus}_nctids.json"

    # if already cached then load, otherwise build
    if os.path.exists(corpus_path):
        embeds = np.load(corpus_path)
        corpus_nctids = json.load(open(nctids_path))
    else:
        embeds = []
        corpus_nctids = []

        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

        with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
            print("Encoding the corpus")
            for line in tqdm.tqdm(f.readlines()):
                entry = json.loads(line)
                corpus_nctids.append(entry["_id"])

                title = entry["title"]
                text = entry["text"]

                with torch.no_grad():
                    # tokenize the articles
                    encoded = tokenizer(
                        [[title, text]],
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=512,
                    ).to("cuda")

                    embed = model(**encoded).last_hidden_state[:, 0, :]

                    embeds.append(embed[0].cpu().numpy())

        embeds = np.array(embeds)

        np.save(corpus_path, embeds)
        with open(nctids_path, "w") as f:
            json.dump(corpus_nctids, f, indent=4)

    index = faiss.IndexFlatIP(768)
    index.add(embeds)

    return index, corpus_nctids


if __name__ == "__main__":
    # different corpora, "trec_2021", "trec_2022", "sigir"
    corpus = sys.argv[1]

    # query type
    q_type = sys.argv[2]

    # different k for fusion
    k = int(sys.argv[3])

    # bm25 weight
    bm25_wt = int(sys.argv[4])

    # medcpt weight
    medcpt_wt = int(sys.argv[5])

    # how many to rank
    N = 2000

    # loading all types of queries
    id2queries = json.load(open(f"dataset/{corpus}/id2queries.json"))

    # loading the indices
    bm25, bm25_nctids, trial_metadata = get_bm25_corpus_index(corpus)
    medcpt, medcpt_nctids = get_medcpt_corpus_index(corpus)

    # loading the query encoder for MedCPT
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    # then conduct the searches, saving top 1k
    output_path = f"results/qid2nctids_results_{q_type}_{corpus}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_N{N}.json"

    qid2nctids = {}

    print("Starting retrieval process...")
    with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
        for line in tqdm.tqdm(f.readlines(), desc="Processing queries"):
            entry = json.loads(line)
            query = entry["text"]
            qid = entry["_id"]

            # get the keyword list
            if q_type in ["raw", "human_summary"]:
                conditions = [id2queries[qid][q_type]]
            elif "turbo" in q_type:
                conditions = id2queries[qid][q_type]["conditions"]
            elif "Clinician" in q_type:
                conditions = id2queries[qid].get(q_type, [])

            if len(conditions) == 0:
                print(f"Skipping QID {qid}: No conditions found.")
                continue

            nctid2score = {}

            # Retrieve using BM25
            bm25_condition_top_nctids = []
            for condition in conditions:
                tokens = word_tokenize(condition.lower())
                top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=N)
                bm25_condition_top_nctids.append(top_nctids)

            # Retrieve using MedCPT
            with torch.no_grad():
                encoded = tokenizer(
                    conditions,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=256,
                ).to("cuda")

                embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
                scores, inds = medcpt.search(embeds, k=N)

            medcpt_condition_top_nctids = []
            for ind_list in inds:
                top_nctids = [medcpt_nctids[ind] for ind in ind_list]
                medcpt_condition_top_nctids.append(top_nctids)

            # Hybrid fusion scoring
            for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(
                zip(bm25_condition_top_nctids, medcpt_condition_top_nctids)
            ):
                if bm25_wt > 0:
                    for rank, nctid in enumerate(bm25_top_nctids):
                        if nctid not in nctid2score:
                            nctid2score[nctid] = 0
                        nctid2score[nctid] += (1 / (rank + k)) * (
                            1 / (condition_idx + 1)
                        )

                if medcpt_wt > 0:
                    for rank, nctid in enumerate(medcpt_top_nctids):
                        if nctid not in nctid2score:
                            nctid2score[nctid] = 0
                        nctid2score[nctid] += (1 / (rank + k)) * (
                            1 / (condition_idx + 1)
                        )

            nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
            top_nctids = []
            for nctid, _ in nctid2score[:N]:
                if nctid in trial_metadata:
                    trial_info = {**trial_metadata[nctid], "NCTID": nctid}
                else:
                    print(f"Metadata missing for NCTID: {nctid}")
                    trial_info = {
                        "NCTID": nctid,
                        "brief_title": "Unknown",
                        "phase": "Unknown",
                        "drugs": "[]",
                        "drugs_list": [],
                        "diseases": "[]",
                        "diseases_list": [],
                        "enrollment": "Unknown",
                        "inclusion_criteria": "Not Available",
                        "exclusion_criteria": "Not Available",
                        "brief_summary": "Metadata not available.",
                    }
                top_nctids.append(trial_info)

            qid2nctids[qid] = top_nctids

    if qid2nctids:
        print(f"Saving results to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(qid2nctids, f, indent=4)
        print(f"Results successfully saved to {output_path}")
    else:
        print("No results generated.")

