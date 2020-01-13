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
	if ! [ -z "$config" ]; then
	    real_options="-c $config $real_options"
	fi
    	outfile="$ex_outdir/${prefix}_$i"
    	if [ -f "$outfile" ]; then
    	    echo " (Skippped)"
    	else
    	    echo ""
    	    ./train_and_eval.sh $outfile $dataset "$real_options"
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

# Sentence length (single)
ex_name='sent_length_single'
options='-o sentence_length {} -o channels_layer_1 80 -o channels_layer_2 0'
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
values="$(seq 0 5 100)"
calc_scores


# Channels layer 1 (no layer 2)
ex_name='channels_1_single'
options='-o sentence_length 4 -o channels_layer_1 {} -o channels_layer_2 0'
values="$(seq 5 5 100)"
calc_scores

# Channels layer 1 (no layer 2)
ex_name='kernel_1_single'
options='-o sentence_length 4 -o channels_layer_1 80 -o channels_layer_2 0 -o kernel_size_layer_1 {}'
values="$(seq 1 6)"
calc_scores

# activation
ex_name='activation'
options='-o sentence_length 4 -o channels_layer_1 80 -o channels_layer_2 0 -o kernel_size_layer_1 3 -o activation_function {}'
values="tanh relu brak"
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
values="$(seq 0 5 100)"
calc_scores

# Channels layer 1 (no layer 2)
ex_name='channels_1_single'
options='-o sentence_length 35 -o channels_layer_1 {} -o channels_layer_2 0'
values="$(seq 5 5 100)"
calc_scores


# Channels layer 1 (no layer 2)
ex_name='kernel_1_single'
options='-o sentence_length 35 -o channels_layer_1 70 -o channels_layer_2 0 -o kernel_size_layer_1 {}'
values="$(seq 1 10)"
calc_scores

# activation
ex_name='activation'
options='-o sentence_length 35 -o channels_layer_1 80 -o channels_layer_2 0 -o kernel_size_layer_1 3 -o activation_function {}'
values="tanh relu brak"
calc_scores

# Sentence length (single)
ex_name='sent_length_single'
options='-o sentence_length {} -o channels_layer_1 80 -o channels_layer_2 0'
values="$(seq 5 5 55)"
calc_scores



### RESULTS

config=''
dataset='data/images'
ex_name='results_word2vec'
options='-c config/{}'
values="cnn_1d.yaml cnn_2d_short.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores

config=''
dataset='data/headlines'
ex_name='results_word2vec'
options='-c config/{}'
values="cnn_1d.yaml cnn_2d_short.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores

config=''
dataset='data/MSRpar'
ex_name='results_word2vec'
options='-c config/{}'
values="cnn_1d.yaml cnn_2d_long.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores

config=''
dataset='data/MSRvid'
ex_name='results_word2vec'
options='-c config/{}'
values="cnn_1d.yaml cnn_2d_long.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores


config=''
dataset='data/images'
ex_name='results_bert'
options='-c config/{} -o embeddings bert'
values="cnn_1d.yaml cnn_2d_short.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores

config=''
dataset='data/headlines'
ex_name='results_bert'
options='-c config/{} -o embeddings bert'
values="cnn_1d.yaml cnn_2d_short.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores

config=''
dataset='data/MSRpar'
ex_name='results_bert'
options='-c config/{} -o embeddings bert'
values="cnn_1d.yaml cnn_2d_long.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores

config=''
dataset='data/MSRvid'
ex_name='results_bert'
options='-c config/{} -o embeddings bert'
values="cnn_1d.yaml cnn_2d_long.yaml vrnn_linear.yaml vrnn_tree.yaml"
calc_scores

