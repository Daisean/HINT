# TENT
Implementation for TENT: Text Classification Based on ENcoding Tree Learning
## Usage
Download stanford nlp model
    import stanfordnlp
    stanfordnlp.download('en_ewt')
You can use other dependency parsing model, find on https://stanfordnlp.github.io/stanfordnlp/models.html

Build dependency tree for dataset in `data/corpus/` as:

    python build_dependency_tree.py <dataset>
    
Provided datasets include `mr`,`ohsumed`,`R8`and`R52`. 

Build encoding tree for dataset as:
    
    python build_encoding_tree.py -d <dataset> -k <tree_deepth> -o <onehot> -a <add> -s <stop>
    example: python build_encoding_tree.py -d mr -k 2 -o True -a False -s False

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

    @misc{zhang2021tent,
          title={TENT: Text Classification Based on ENcoding Tree Learning}, 
          author={Chong Zhang and Junran Wu and He Zhu and Ke Xu},
          year={2021},
          eprint={2110.02047},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }
