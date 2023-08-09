# POSTER
The project is an official implementation of our paper [POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition](https://arxiv.org/pdf/2204.04083.pdf).


### Preparation
- create conda environment (we provide requirements.txt)

- Data Preparation

  Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset) dataset, and make sure it have a structure like following:
 
	```
	- data/raf-basic/
		 EmoLabel/
		     list_patition_label.txt
		 Image/aligned/
		     train_00001_aligned.jpg
		     test_0001_aligned.jpg
		     ...
	```

- Pretrained model weights
Dowonload pretrain weights (Image backbone and Landmark backbone) from [here](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq?usp=sharing). Put entire `pretrain` folder under `models` folder.

	```
	- models/pretrain/
		 ir50.pth
		 mobilefacenet_model_best.pth.tar
		     ...
	```

### Testing

Our best model can be download from [here](https://drive.google.com/drive/folders/1jeCPTGjBL8YgKKB9YrI9TYZywme8gymv?usp=sharing), put under `checkpoint ` folder. You can evaluate our model on RAD-DB dataset by running: 

```
python test.py --checkpoint checkpoint/rafdb_best.pth -p
```

### Training
Train on RAF-DB dataset:
```
python train.py --gpu 0,1 --batch_size 200
```
You may adjust batch_size based on your # of GPUs. Usually bigger batch size can get higher performance. We provide the log in  `log` folder. You may run several times to get the best results. 


## License

Our research code is released under the MIT license. See [LICENSE](LICENSE) for details. 



## Citations
If you find our work useful in your research, please consider citing:

```bibtex
@article{zheng2022poster,
  title={Poster: A pyramid cross-fusion transformer network for facial expression recognition},
  author={Zheng, Ce and Mendieta, Matias and Chen, Chen},
  journal={arXiv preprint arXiv:2204.04083},
  year={2022}
}
```


## Acknowledgments

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[JiaweiShiCV/Amend-Representation-Module](https://github.com/JiaweiShiCV/Amend-Representation-Module) 


