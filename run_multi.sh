#!/bin/bash
#SBATCH -J ctc_multi
#SBATCH -p gpu,2080ti
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o logs/multi.%j
#SBATCH --gres=gpu:1
#SBATCH --mem=16G


. path.sh

stage=0

feat_type='fbank'                          #fbank, mfcc, spectrogram
config_file='conf/multi_config.yaml'
exp=`cat $config_file | grep exp_name | awk '{print $2}' | sed "s/'//g"`
exp_dir=checkpoint/$exp
mkdir -p $exp_dir
cp $config_file $exp_dir/config.yaml
config_file=$exp_dir/config.yaml


feat_dir=`cat $config_file | grep lm_path | awk '{print $2}' | sed "s/'//g" | awk -F "/" '{print $1"/"$2}'`
echo "feat_dir $feat_dir"
if [ ! -z $1 ]; then
    stage=$1
fi


if [ $stage -le 2 ]; then
    echo "Step 2: Acoustic Model(CTC) Training..."
    echo "Configfile: $config_file"
    python3 steps/train_multi.py --conf $config_file || exit 1;
fi

if [ $stage -le -1 ]; then
    echo "Step 3: LM Model Training..."
    steps/train_lm.sh $feat_dir || exit 1;
fi

if [ $stage -le -1 ]; then
    echo "Step 4: Decoding..."
    CUDA_VISIBLE_DEVICE='0' python3 steps/test_ctc.py --conf $config_file || exit 1;
fi

if [ $stage -le -1 ]; then
    echo "Step 5: Keyword Decoding..."
    [ -f $feat_dir/test/keyword ] || python3 utils/kw_test_gen.py --dir $feat_dir/test || exit 1;
    python3 steps/test_kws.py --conf $config_file || exit 1;
fi
