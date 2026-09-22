# datascience-HNSW

# Learning Approximate Nearest Neighbour (ANN) search graph for HNSW.

This project explores learning the graph structure used in Hierarchical Navigable Small Worlds (HNSW) for efficient ANN search. HNSW is a data structure designed to quickly find near neighbors for a given query in high-dimensional spaces. The goal is to investigate whether, given a set of ANN query examples and their outputs, we can learn the optimal graph structure for HNSW.

Start by learning about HNSW here: [Hierarchical Navigable Small Worlds (HNSW) | Pinecone](https://www.pinecone.io/learn/hnsw/).

## Dataset

We use pre-trained word vectors from the [GloVe project](https://nlp.stanford.edu/projects/glove/):

- Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB): `glove.6B.zip`
- Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB): `glove.42B.300d.zip`
- Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB): `glove.840B.300d.zip`
<!-- - Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB): `glove.twitter.27B.zip` -->

The data is available under the [Public Domain Dedication and License v1.0](http://www.opendatacommons.org/licenses/pddl/1.0/).

## Project Steps

1. Download and preprocess the GloVe word vectors.
2. Implement or use an HNSW library for ANN search.
3. Generate ANN queries and collect their outputs.
4. Explore methods to learn or optimize the HNSW graph structure based on query results.

## Adaptive Cluster Search Extension

[`adaptive_budget/`](adaptive_budget/) contains a small, dependency-free utility for clustered HNSW experiments. It selects how many clusters to search from the gap between the nearest centroid distances. The existing implementations remain unchanged; use it only in a new experiment after tuning thresholds on development queries and evaluating on held-out queries.

## References

- [Hierarchical Navigable Small World Graphs (original paper)](https://arxiv.org/abs/1603.09320)
- [Pinecone HNSW Guide](https://www.pinecone.io/learn/hnsw/)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
