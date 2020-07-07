#!/bin/bash
data=data_multi/$1
target=data_semi/$1
portion=50

cp -r $data $target
newnum=`wc -l $data/train/feats.scp | awk -v port=$portion '{print int($1*port/100)}'`

rm $target/train/*
head -n $newnum $data/train/feats.scp > $target/train/feats.scp
head -n $newnum $data/train/lab.txt > $target/train/lab.txt
unlabelnum=`wc -l $data/train/feats.scp | awk -v newnum=$newnum '{print $1-newnum}'`
tail -n $unlabelnum $data/train/feats.scp >$target/train/feats_nolabel.scp



