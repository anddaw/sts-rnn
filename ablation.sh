#!/usr/bin/bash

# Calculate results needed for ablation study

set -o pipefail
set -o errexit
# set -o xtrace


calc_scores() {

    prefix="$(basename $dataset)-$ex_name"

    ex_outdir="$outdir/tmp_$prefix"
    mkdir -p $ex_outdir
    summary_file=$outdir/$prefix.tsv
    plot_file=$outdir/$prefix.pdf
    
    echo -e "name\toptions\tdataset\tpearson\tfscore\tparam" > $summary_file
    for i in $values; do
    	echo -n "##### $prefix $i"
    	real_options="$(sed "s/{}/$i/" <<< "$options")"
    	outfile="$ex_outdir/${prefix}_$i"
    	if [ -f "$outfile" ]; then
    	    echo " (Skippped)"
    	else
    	    echo ""
    	    ./train_and_eval.sh $outfile $config $dataset "$real_options"
    	fi
	
    	cat $outfile | sed "s/\$/\t$i/" >> $summary_file
	   
    done
    python evaluation/plot_results.py $summary_file $plot_file --title $ex_name
}


	   


outdir=results/ablation/cnn2d
config="config/cnn_2d.yaml"
### IMAGES
dataset='data/images'

# Sentence length 
ex_name=sent_length
options='-o sentence_length {}'
values="$(seq 4 1 10)"
calc_scores


# Channels layer 1
ex_name='channels_1'
options='-o sentence_length 4 -o channels_layer_1 {}'
values="1 $(seq 5 5 100) $(seq 50 20 220)"
calc_scores

# Channels layer 2
ex_name='channels_2'
options='-o sentence_length 4 -o channels_layer_1 20 -o channels_layer_2 {}'
values="$(seq 5 5 100)"
calc_scores


### MSRPAR
dataset='data/MSRpar'
# Sentence length 
ex_name='sent_length'
options='-o sentence_length {}'
values="$(seq 5 5 40)"
calc_scores

# Channels layer 1
ex_name='channels_1'
options='-o sentence_length 35 -o channels_layer_1 {}'
values="$(seq 1 2 10) $(seq 10 10 110)"
calc_scores


# Channels layer 2
ex_name='channels_2'
options='-o sentence_length 35 -o channels_layer_1 70 -o channels_layer_2 {}'
values="$(seq 5 5 100)"
calc_scores
