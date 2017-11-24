################################################
# To reproduce our CIFAR result
################################################
python main.py --model condensenet -b 64 -j 8 cifar10 --epochs 300 --lr-type cosine --stages 14-14-14 --growth 8-16-32 --bottleneck 4 --group-1x1 4 --group-3x3 4 --condense-factor 4 --gpu 0 --resume
