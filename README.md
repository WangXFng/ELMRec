# ELMRec
## Paper
- Enhancing High-order Interaction Awareness in LLM-based Recommender Model, **EMNLP 2024 Main (Long)**.
- [**Xinfeng Wang**](https://wangxfng.github.io/), Jin Cui, Yoshimi Suzuki, Fumiyo Fukumoto.

## Note
- The checkpoints of the ELMRec model are uploaded on [Google Drive](https://drive.google.com/drive/folders/131UDvFsUbJHr-G1um4XUNpMfvIJ54FMj?usp=sharing).
- If you have any questions, please feel free to contact me at kaysenn@163.com.


## Train and test ELMRec
### (1) Install requirement 
        >> pip install -r  requirement.txt

### (2) Pretraining
        >> python pretrain.py ./data/{dataset}/ --cuda --batch_size 64 --checkpoint ./checkpoint/{dataset}/

### (3) Inference
        >> python seq_reranker.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/
        >> python topn.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/
        >> python exp.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/


## Code Reference
- [P5](https://github.com/jeykigung/P5)
- [POD](https://github.com/lileipisces/POD)


## Citation
If this repository helps you, please cite:

	@inproceedings{wang2024rdrec,
	  title={Enhancing High-order Interaction Awareness in LLM-based Recommender Model},
	  author={Wang, Xinfeng and Cui, Jin and Fukumoto, Fumiyo and Suzuki, Yoshimi},
	  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
	  year={2024}
	}
