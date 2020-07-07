#!/bin/bash
dir=$1
for ldir in `ls $dir`; do
    wer=`cat $dir/$ldir/log | grep Word| awk '{print $7}'`
    f1=`cat $dir/$ldir/log | grep "For keyword" | tail -n5 | awk 'BEGIN{f1=0}{f1+=$12;}END{print f1/5}'`
    language=`echo $ldir | awk -F "_" '{print $1}'`
    echo $language $wer $f1 >> $dir/log
    echo $language $wer $f1
done
