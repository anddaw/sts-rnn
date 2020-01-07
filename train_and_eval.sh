#!/usr/bin/bash -e

export PYTHONPATH=$(dirname $(readlink -f $0))


outfile="$1"
config_file="$2"
dataset="$3"
options="$4"

if [ -f "$outfile" ]; then
   exit 1
fi

python train.py -c $config_file $options -o dataset $dataset
pred_file=best_model_preds.tsv

tmp_wa=$(mktemp)

if [ -f $dataset/test.wa ]; then
    python evaluation/insert_into_wa.py "$dataset/test.wa" "$pred_file" > "$tmp_wa"
    fscore=$(./evaluation/evalF1.pl "$dataset"/test.wa "$tmp_wa" \
		 | grep "F1 Score" | awk '{print $3}')
else
    fscore="-"
fi

pearson=$(python evaluation/evalPearson.py "$dataset/test_labels.txt" "$pred_file")

echo -e "$(basename $outfile)\t$(basename $config_file) $options\t$dataset\t$pearson\t$fscore" > $outfile






   



