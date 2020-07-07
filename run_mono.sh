#!/bin/bash
#SBATCH -J ctc_monol
#SBATCH -p gpu,2080ti
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o logs/mono.%j
#SBATCH --gres=gpu:1
#SBATCH --mem=16G


. path.sh

stage=1

feat_type='fbank'                          #fbank, mfcc, spectrogram
config_file='conf/ctc_config.yaml'
exp=`cat $config_file | sed "s/country/$1/g" | grep exp_name | awk '{print $2}' | sed "s/'//g"`

exp_dir=checkpoint/$exp
mkdir -p $exp_dir
cat $config_file | sed "s/country/$1/g" > $exp_dir/config.yaml
echo "Copy config Done"
config_file=$exp_dir/config.yaml

feat_dir=`cat $config_file | grep data_file | awk '{print $2}' | sed "s/'//g" | awk -F "/" '{print $1"/"$2}'`
#feat_dir=`cat $config_file | grep lm_path | awk '{print $2}' | sed "s/'//g" | awk -F "/" '{print $1"/"$2}'`
echo "feat_dir $feat_dir"
if [ ! -z $2 ]; then
    stage=$2
fi


if [ $stage -le 2 ]; then
    echo "Step 2: Acoustic Model(CTC) Training..."
    echo "Configfile: $config_file"
    CUDA_VISIBLE_DEVICE='0' python3 steps/train_ctc.py --conf $config_file || exit 1;
fi

if [ $stage -le 3 ]; then
    echo "Step 3: LM Model Training..."
    steps/train_lm.sh $feat_dir || exit 1;
fi

if [ $stage -le 4 ]; then
    echo "Step 4: Decoding..."
    CUDA_VISIBLE_DEVICE='1' python3 steps/test_ctc.py --conf $config_file >> $exp_dir/log|| exit 1;
fi

if [ $stage -le 5 ]; then
    echo "Step 5: Keyword Decoding..."
    [ -f $feat_dir/test/keyword ] || python3 utils/kw_test_gen.py --dir $feat_dir/test || exit 1;
    #python3 steps/test_kws.py --conf $config_file>>$exp_dir/log || exit 1;
    python3 steps/test_kws.py --conf $config_file || exit 1;
fi
