# suppose 1 is less than 2, make 2 consistent with 1



awk '{if(FNR==NR){list[$1]=$1}else{if($1 in list){print $0}}}' $1 $2 >$2.new
