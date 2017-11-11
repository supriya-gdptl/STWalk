# STWalk

This repository contains implementation of "STWalk: Learning Trajectory Representations in Temporal Graphs".

STWalk uses structural properties of graphs at present and previous time-steps to capture the spatio-temporal behavior of nodes. 
There are two variants of STWalk:
 *  STWalk1: Learns the trajectory representation of node by considering spatial and temporal neighbors at the same time. Figure1 below shows the spatiotemporal graph generated for a single node u_t.
 *  STWalk2: It learns spatial representation and temporal representation separately and then combines them to generate trajectory representation.

## Requirement
*  python 3.4 (or later)
*  networkx 1.11
*  gensim 2.3.0

## To run the STWalk1 algorithm:
Please use *--dataset <dataset-name>* argument, where *dataset-name* can be one of the following: "dblp", "epinion", "ciao". By default the code will be executed on Epinion dataset.
The output will be saved in */epinion/output_stwalkone/* folder

```
cd code
python STWalk1.py --dataset epinion
```

## To run the STWalk2 algorithm:
The output will be saved in */epinion/output_stwalktwo/* folder

```
cd code
python STWalk2.py --dataset ciao
```

## Data
We experiment on three real-world datasets: DBLP, Epinion, Ciao datasets
*  Folder "dblp/input_graphs" contains DBLP co-authorship graphs. There are 45 from 1969 to 2011 excluding 1970, 1972, and 1974.
*  Folder "epinion/input_graphs" contains Epinion dataset graphs. Epinion is a popular product review site. Each node in a graph is reviewer and two reviewers share an edge if they have reviewed product from same category. We have considered 110 graphs from monthly data of March 2002 to April 2011.
*  Folder "ciao/input_graphs" contains Ciao dataset. Ciao is another popular product review site. We have considered 115 graphs from September 2001 to March 2011.


## Cite

Please cite our paper if you use this code in your work:

Paper title: STWalk: Learning Trajectory Representations in Temporal Graphs

Authors: Supriya Pandhre, Himangi Mittal, Manish Gupta, Vineeth N Balasubramanian
