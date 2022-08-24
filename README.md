# HINT
Implementation for Hierarchical information matters: Text classification via tree based graph neural network
## Usage
Build dependency graph for dataset in `data/corpus/` as:

    python build_dependency_graph.py <dataset>
    
Provided datasets include `mr`,`ohsumed`,`R8`and`R52`. 

Build coding tree for dataset as:
    
    python build_coding_tree.py -d <dataset> -k <tree_deepth> -o <onehot> -a <add> -s <stop>
    example: python build_coding_tree.py -d mr -k 2 -o True -a False -s False

Start training and inference as:
    
     python main.py [--dataset DATASET] [--tree_deepth DEEPTH]
                    [--epochs EPOCHS] [--batch_size BATCHSIZE]
                    [--hidden_dim HIDDEN_DIM] [--learning_rate LEARNING_RATE]
                    [--final_dropout DROPOUT] [--input_dim INPUT_DIM]
                    [--num_mlp_layers MLP_LAYERS] [--l2rate L2RATE]
                    [--tree_pooling_type TREE_POOLING] [--mode MODE]
                    [--position_embedding PE] 
    example: python main.py -d mr -k 2 -b 4 -md dependency -pe onehot

## Citation

    @misc{zhang2022hint,
          title={Hierarchical information matters: Text classification via tree based graph neural network}, 
          author={Chong Zhang and He Zhu and Xingyu Peng and Junran Wu and Ke Xu},
          year={2022},
    }
