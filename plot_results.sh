#!/usr/bin/bash

# Calculate results needed for ablation study

set -o pipefail
set -o errexit
set -o xtrace



# python evaluation/plot_results.py \
#        results/ablation/final/images-activation.tsv \
#        results/ablation/final/images-activation.pdf \
#        --title "Funkcja aktywacji (images)"



python evaluation/plot_results.py \
       results/ablation/final/images-sent_length_single.tsv \
       results/ablation/final/images-sent_length_single.pdf \
       --title "Rozmiar danych wejściowych (images)"

python evaluation/plot_results.py \
       results/ablation/final/MSRpar-sent_length_single.tsv \
       results/ablation/final/MSRpar-sent_length_single.pdf \
       --title "Rozmiar danych wejściowych (MSRpar)"



python evaluation/plot_results.py \
       results/ablation/final/images-channels_1_single.tsv \
       results/ablation/final/images-channels_1_single.pdf \
       --title "Liczba kanałów w warstwie splotowej (images)"

python evaluation/plot_results.py \
       results/ablation/final/MSRpar-channels_1_single.tsv \
       results/ablation/final/MSRpar-channels_1_single.pdf \
       --title "Liczba kanałów w warstwie splotowej (MSRpar)"

python evaluation/plot_results.py \
       results/ablation/final/images-kernel_1_single.tsv \
       results/ablation/final/images-kernel_1_single.pdf \
       --title "Rozmiar filtrów splotowych (images)"

python evaluation/plot_results.py \
       results/ablation/final/MSRpar-kernel_1_single.tsv \
       results/ablation/final/MSRpar-kernel_1_single.pdf \
       --title "Rozmiar filtrów splotowych (MSRpar)"

python evaluation/plot_results.py \
       results/ablation/final/images-channels_2.tsv \
       results/ablation/final/images-channels_2.pdf \
       --title "Liczba kanałów w 2. warstwie splotowej (images)"

python evaluation/plot_results.py \
       results/ablation/final/MSRpar-channels_2.tsv \
       results/ablation/final/MSRpar-channels_2.pdf \
       --title "Liczba kanałów w 2. warstwie splotowej (MSRpar)"





find results/ablation/final/ -name '*pdf' -exec cp {} ~/Projekty/mgr-txt/img \;

