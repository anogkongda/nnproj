#MAIN_ROOT=/mnt/lustre/sjtu/home/zkz01/espnet


export KALDI_ROOT=/mnt/lustre/sjtu/home/zkz01/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$KALDI_ROOT/tools/sctk/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

__conda_setup="$('/mnt/lustre/sjtu/home/zkz01/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/lustre/sjtu/home/zkz01/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/lustre/sjtu/home/zkz01/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/lustre/sjtu/home/zkz01/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
#alias copy-align="copy-int-vector"
#module load anaconda/3
#. /mnt/lustre/cm/shared/global/src/machinelearning/anaconda3/etc/profile.d/conda.sh
#module load cuda/10.1
#module load cudnn/7.6.1-cuda10.0
module load gcc/7.3.0
module load cmake/3.12.0
module load sox/14.4.2
module load cuda/10.0
module load cudnn/7.5.1-cuda10.0

#export NVM_DIR="$HOME/.nvm"
#[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
#[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
conda activate pytorch

#export PYTHONPATH=./asr:$PYTHONPATH
export PYTHONPATH=./:$PYTHONPATH
#export PYTHONPATH=./kws:$PYTHONPATH


#export OMP_NUM_THREADS=1
#export PYTHONIOENCODING=UTF-8
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
#export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH
