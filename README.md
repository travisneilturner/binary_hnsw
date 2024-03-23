# Binary HNSW

## Introduction
This repo shows a very basic and suboptimal way to use binary vector embeddings in HNSW.  It is intended as a PoC to illustrate
that existing HNSW algorithms shouldn't require a lot of changes to work with binary embeddings. 

## Implementation
As mentioned, this implementation is not optimal or state-of-the-art.  It is about as basic as HNSW gets.  There is no heuristic
or probabilistic neighbor approximation for search/insert, it will do a full layer scan each and every time.  

## Future Work
I'd like to make it optimal and use proper approaches for limiting the scans to a subsdet of the layer based on heuristics / probability. 

