# ELMRec

## Pretraining
        >> python pretrain.py ./data/{dataset}/ --cuda --batch_size 64 --checkpoint ./checkpoint/{dataset}/

## Inference
        >> python seq_reranker.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/
        >> python topn.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/
        >> python exp.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/

