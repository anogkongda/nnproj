# nnproj

Some source code is from 
https://github.com/Diamondfan/CTC_pytorch
It is a pytorch asr recipe for TIMIT.

##Requirements
1. Pytorch (use built-in CTC)
2. torchaudio

Usage
1. Download data from http://www.voxforge.org/ for Catalan, French, Portuguese and Italian
and 
Download data from https://openslr.org/76/ for Basque
Process these data by kaldi to
data_semi/French
data_semi/Catalan
...

For each language directory, such as French, we have
French/train, French/dev, French/test
French/text # for lm training
in train/dev/test, we have
feats.scp, feats_nolabel.scp, lab.txt




2. bash run_mono.sh Catalan                 # To finetune then test model on single language
bash run_multi.sh                           # To train model on multi languages. 


3. General Procedure:  
 train the model from mutli language first by "bash run_multi.sh"

add the model path to mono config's "resume" to train on specific language



Mono language config: conf/ctc_config.yaml

Multi language config: conf/multi_config.yaml
