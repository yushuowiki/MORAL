# MORAL

The PyTorch implementation of “Multi-type social patterns-based graph learning”.

## Abstract

Capturing the topology exhibited in social graphs is of paramount importance to graph learning, since node representations are heavily reliant on their neighbors in graph neural networks. Conventional methods often compute node representations by aggregating features of their multi-hop neighbors, while overlooking those subgraph structures, such as those multi-type social patterns that contain unique and rich semantic information. However, modeling all the multi-type social patterns in graphs is non-trivial, due to the expensive computational cost incurred by the involvement of multiple nodes in these patterns. In addition, the multi-type social pattern usually exhibits a long-tailed distribution over graph nodes, which makes it difficult for graph neural networks to accurately calibrate the pattern information for each node. In this paper, we propose a novel multi-type social patterns-based graph learning method (MORAL), which adaptively learns the multi-type social patterns in the graph through a structural attention mechanism, and balances its distribution state to improve the accuracy of low-degree nodes. Experimental results show that MORAL significantly improves the performance compared with other baselines, particularly on large graphs.

## Dependencies

- torch==1.12.1
- torch_geometric==2.1.0.post1
- scipy==1.9.3
- scikit_learn==1.1.3
- powerlaw==1.5
- numpy==1.23.1
- networkx==2.8.4

## Usage

Preprocess datasets by executing

```
python rawdata_process.py
python cal_motif.py
```

Train and evaluate the model by executing

```
python main.py
```

## Results

Results are shown in

```
records/record.log
```

Details of the training process are shown in

```
logs/training_log
```

