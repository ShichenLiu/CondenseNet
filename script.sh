################################################
# To reproduce our CIFAR result
################################################
python main.py --model condensenet -b 64 -j 8 cifar10 --epochs 300 --lr-type cosine --stages 14-14-14 --growth 8-16-32 --bottleneck 4 --group-1x1 4 --group-3x3 4 --condense-factor 4 --gpu 0 --resume


################################################
# To reproduce our ImageNet result
################################################
# 1. Download models (e.g. 19M C=G=4)
wget "https://www.dropbox.com/s/sj26rm4so3uhdmg/converted_condensenet_4.pth.tar?dl=0"

# 2. Run the converted evaluate model
python main.py --model condensenet_converted -b 64 -j 4 /PATH/TO/IMAGENET --stages 4-6-8-10-8 --growth 8-16-32-64-128 --group-1x1 4 --group-3x3 4 --condense-factor 4 --evaluate-from /PATH/TO/converted_condensenet_4.pth.tar --gpu 0


################################################
# To train CondenseNets on ImageNet
################################################
python main.py --model condensenet -b 256 -j 20 /PATH/TO/IMAGENET --epochs 120 --stages 4-6-8-10-8 --growth 8-16-32-64-128 --group-1x1 4 --group-3x3 4 --condense-factor 4 --bottleneck 4 --resume --group-lasso 0.00001 --gpu 0,1,2,3


################################################
# To train PyTorch DenseNet-BC-100 (k=12)
################################################
python main.py --model densenet -b 64 -j 4 cifar10 --epochs 300 --lr-type cosine --stages 16-16-16 --growth 12-12-12 --bottleneck 4 --group-1x1 1 --group-3x3 1 --gpu 0 --resume
