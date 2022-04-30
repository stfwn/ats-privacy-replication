####################
## experiments.sh ##
####################

# These are all the experiments that were run to produce all the tables and
# figures in the report. Uncomment parts to reproduce.

#############################
# Table 1, figures 3, 4 & 5 #
#############################

# Training
# ********

# random_policy="19-1-18"
## a)
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --aug-list $random_policy
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --aug-list 3-1-7
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --aug-list 43-18-18
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --aug-list 3-1-7+43-18-18

## b)
# python main.py train --model convnet --dataset cifar100 -e 60 --bugged-loss
# python main.py train --model convnet --dataset cifar100 -e 60 --bugged-loss --aug-list $random_policy
# python main.py train --model convnet --dataset cifar100 -e 60 --bugged-loss --aug-list 21-13-3
# python main.py train --model convnet --dataset cifar100 -e 60 --bugged-loss --aug-list 7-4-15
# python main.py train --model convnet --dataset cifar100 -e 60 --bugged-loss --aug-list 7-4-15+21-13-3

## c)
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss --aug-list $random_policy
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss --aug-list 19-15-45
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss --aug-list 2-43-21
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss --aug-list 2-43-21+19-15-45

## d)
# python main.py train --model convnet --dataset fmnist -e 60 --bugged-loss
# python main.py train --model convnet --dataset fmnist -e 60 --bugged-loss --aug-list $random_policy
# python main.py train --model convnet --dataset fmnist -e 60 --bugged-loss --aug-list 42-28-42
# python main.py train --model convnet --dataset fmnist -e 60 --bugged-loss --aug-list 14-48-48
# python main.py train --model convnet --dataset fmnist -e 60 --bugged-loss --aug-list 14-48-48+42-28-42

# Attacks
# *******

# optimizer=inversed
# for img_idx in {0..5}
# do
#         ## a)
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer $optimizer --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list $random_policy 
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list 3-1-7
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list 43-18-18
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list 3-1-7+43-18-18

#         ## b)
#         python main.py attack --model convnet --dataset cifar100 --optimizer $optimizer --image-index $img_idx
#         python main.py attack --model convnet --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list $random_policy 
#         python main.py attack --model convnet --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list 21-13-3
#         python main.py attack --model convnet --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list 7-4-15
#         python main.py attack --model convnet --dataset cifar100 --optimizer $optimizer --image-index $img_idx --aug-list 7-4-15+21-13-3

#         ## c)
#         python main.py attack --model resnet20 --dataset fmnist --optimizer $optimizer --image-index $img_idx
#         python main.py attack --model resnet20 --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list $random_policy
#         python main.py attack --model resnet20 --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list 19-15-45
#         python main.py attack --model resnet20 --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list 2-43-21
#         python main.py attack --model resnet20 --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list 2-43-21+19-15-45

#         ## d)
#         python main.py attack --model convnet --dataset fmnist --optimizer $optimizer --image-index $img_idx
#         python main.py attack --model convnet --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list $random_policy
#         python main.py attack --model convnet --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list 42-28-42
#         python main.py attack --model convnet --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list 14-48-48
#         python main.py attack --model convnet --dataset fmnist --optimizer $optimizer --image-index $img_idx --aug-list 14-48-48+42-28-42
# done


###########
# Table 2 #
###########

# Training
# ********

# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --defense prune-70
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --defense prune-95
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --defense prune-99
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --defense laplacian-0.001
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --defense laplacian-0.01
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --defense gaussian-0.001
# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss --defense gaussian-0.01

# Attacks
# *******

# for img_idx in {0..5}
# do
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --defense prune-70 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --defense prune-95 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --defense prune-99 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --defense laplacian-0.001 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --defense laplacian-0.01 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --defense gaussian-0.001 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --defense gaussian-0.01 --image-index $img_idx
# done

###########
# Table 3 #
###########

# Training
# ********

# python main.py train --model resnet20 --dataset cifar100 -e 60 --bugged-loss

# Attacks
# *******

# img_idx=0
# hybrid=3-1-7+43-18-18
# for img_idx in {0..5}
# do
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer zhu --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-LBFGS-sim --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-adam-L1 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-adam-L2 --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-sgd-sim --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversefed-default --image-index $img_idx

#         python main.py attack --model resnet20 --dataset cifar100 --optimizer zhu --image-index $img_idx --aug-list $hybrid
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-LBFGS-sim --image-index $img_idx --aug-list $hybrid
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-adam-L1 --image-index $img_idx --aug-list $hybrid
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-adam-L2 --image-index $img_idx --aug-list $hybrid
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed-sgd-sim --image-index $img_idx --aug-list $hybrid
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversefed-default --image-index $img_idx --aug-list $hybrid
# done

###########
# Table 4 #
###########

# Training
# ********

## a)
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss --aug-list 3-1-7
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss --aug-list 43-18-18
# python main.py train --model resnet20 --dataset fmnist -e 50 --bugged-loss --aug-list 3-1-7+43-18-18

## b)
# python main.py train --model convnet --dataset fmnist -e 60 --bugged-loss
# python main.py train --model convnet --dataset fmnist -e 50 --bugged-loss --aug-list 21-13-3
# python main.py train --model convnet --dataset fmnist -e 50 --bugged-loss --aug-list 7-4-15
# python main.py train --model convnet --dataset fmnist -e 50 --bugged-loss --aug-list 7-4-15+21-13-3

# Attacks
# *******

## a) attacks
# for img_idx in {0..5}
# do
#         python main.py attack --model resnet20 --dataset fmnist --optimizer inversed --image-index $img_idx
#         python main.py attack --model resnet20 --dataset fmnist --optimizer inversed --image-index $img_idx --aug-list 3-1-7
#         python main.py attack --model resnet20 --dataset fmnist --optimizer inversed --image-index $img_idx --aug-list 43-18-18
#         python main.py attack --model resnet20 --dataset fmnist --optimizer inversed --image-index $img_idx --aug-list 3-1-7+43-18-18

#         python main.py attack --model convnet --dataset fmnist --optimizer inversed --image-index $img_idx
#         python main.py attack --model convnet --dataset fmnist --optimizer inversed --image-index $img_idx --aug-list 21-13-3
#         python main.py attack --model convnet --dataset fmnist --optimizer inversed --image-index $img_idx --aug-list 7-4-15
#         python main.py attack --model convnet --dataset fmnist --optimizer inversed --image-index $img_idx --aug-list 7-4-15+21-13-3
# done

###########################
# Extension: Extra attack #
###########################

# random_policy="19-1-18"
# for img_idx in {0..5}
# do
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversefed-default --image-index $img_idx
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversefed-default --image-index $img_idx --aug-list $random_policy
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversefed-default --image-index $img_idx --aug-list 3-1-7
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversefed-default --image-index $img_idx --aug-list 43-18-18
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversefed-default --image-index $img_idx --aug-list 3-1-7+43-18-18
# done

#######################
# Extension: ImageNet #
#######################

# python main.py train --model resnet20 --dataset tiny-imagenet200 --epochs=50 --tiny-subset-size=0.1 --bugged-loss

# checkpoint_path=logs/tiny-imagenet200-resnet20/training/none/lightning_logs/version_8726829/checkpoints/epoch=49.ckpt
# policies=0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28+29+30+31+32+33+34+35+36+37+38+39+40+41+42+43+44+45+46+47+48+49
# python main.py search --model resnet20 --dataset tiny-imagenet200 --checkpoint-path=$checkpoint_path --aug-list=$policies

# random_policy="19-1-18"
# python main.py train --model resnet20 --dataset tiny-imagenet200 -e 40 --bugged-loss
# python main.py train --model resnet20 --dataset tiny-imagenet200 -e 40 --bugged-loss --aug-list 3-1-7+43-18-18

# optimizer=inversed
# for img_idx in {0..5}
# do
#         python main.py attack --model resnet20 --dataset tiny-imagenet200 --optimizer inversed --image-index $img_idx
#         python main.py attack --model resnet20 --dataset tiny-imagenet200 --optimizer inversed --image-index $img_idx --aug-list 3-1-7+43-18-18
# done

############
# Figure 1 #
############

# checkpoint_path=logs/cifar100-resnet20/training/none/lightning_logs/version_8728662/checkpoints/epoch=49-step=1949.ckpt
# python main.py search --model resnet20 --dataset cifar100 --checkpoint-path=$checkpoint_path

############
# Figure 6 #
############

# checkpoint_path=logs/cifar100-resnet20/training/none/lightning_logs/version_8728662/checkpoints/epoch=49-step=1949.ckpt
# policies=0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28+29+30+31+32+33+34+35+36+37+38+39+40+41+42+43+44+45+46+47+48+49
# python main.py search --model resnet20 --dataset cifar100 --checkpoint-path=$checkpoint_path --aug-list=$policies

############
# Figure 7 #
############

# checkpoint_path=logs/cifar100-resnet20/training/none/lightning_logs/version_8728662/checkpoints/epoch=49-step=1949.ckpt
# policies=`python -c "import conf;print(' '.join(conf.selected_random_policies))"`
# for policy in policies
# do 
#         python main.py attack --model resnet20 --dataset cifar100 --optimizer inversed --image-index 0 --aug-list $policy --checkpoint-path $checkpoint_path --max-iterations=2500
#         python main.py search --model resnet20 --dataset cifar100 --optimizer inversed --image-index 0 --aug-list $policy --checkpoint-path $checkpoint_path
# done
