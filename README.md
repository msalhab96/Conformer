# Conformer

This is a PyTorch implementation of [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) paper

![conf_arch](https://user-images.githubusercontent.com/61272193/161316084-ab71af86-0c7c-4ffd-9273-d99a6cb91848.png)


# Train on your data
In order to train the model on your data follow the steps below 
### 1. data preprocessing 
* prepare your data and make sure the data is formatted in an CSV format as below 
```
audio_path,text,duration
file/to/file.wav,the text in that file,3.2 
```
* make sure the audios are MONO if not make the proper conversion to meet this condition

### 2. Setup development environment
* create enviroment 
```bash
python -m venv env
```
* activate the enviroment
```bash
source env/bin/activate
```
* install the required dependencies
```bash
pip install -r requirements.txt
```

### 3. Training 
* update the config file if needed
* train the model 
  * from scratch 
  ```bash
  python train.py
  ```
  * from checkpoint 
  ```
  python train.py checkpoint=path/to/checkpoint tokenizer.tokenizer_file=path/to/tokenizer.json
  ```
  
  
  # Referances
  ```
  @misc{https://doi.org/10.48550/arxiv.2005.08100,
  doi = {10.48550/ARXIV.2005.08100},
  
  url = {https://arxiv.org/abs/2005.08100},
  
  author = {Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and Pang, Ruoming},
  
  keywords = {Audio and Speech Processing (eess.AS), Machine Learning (cs.LG), Sound (cs.SD), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Conformer: Convolution-augmented Transformer for Speech Recognition},
  
  publisher = {arXiv},
  
  year = {2020},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

  ```
  
