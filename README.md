# nnproj

##Requirements
1. Pytorch (use built-in CTC)
2. torchaudio

Usage
1. ln -s /mnt/lustre/sjtu/home/zkz01/ctc/data_semi .
To get data directory. It has five languages
Catalan, French, Portuguese, basque Italian

2. bash run_mono.sh Catalan                 # To finetune/ train then test model on single language
bash run_multi.sh                           # To train model on multi languages. 


3. General Procedure:  
edit multi_config and train the model from mutli language first

add the model path to mono config's "resume" to train on specific language



Mono language config: conf/ctc_config.yaml

Multi language config: conf/multi_config.yaml
