## Generate Poisons:

Feature Collision (FC) and Bullseye Polytope (BP) are under **transfer learning** setting. Gradient Matching (GM) is under **training from scratch** setting. We have 20 trials for each poison method. And after poison generation. We got 3 `.pickle` files for each trail.

### Feature Collision:

for f in {0..19..1};do \
    python3 ./poison_crafting/craft_poisons_fc.py \
        --poisons_path "poison_examples_transfer/fc_poisons/$f" \
        --pretrain_dataset CIFAR100 --dataset CIFAR10  \
        --poison_setups "poison_setups/cifar10_transfer_learning.pickle" \
        --setup_idx ${f} \
done

### Bullseye Polytope:

for f in {0..19..1};do \
    python3 ./poison_crafting/craft_poisons_bp.py \
        --poisons_path "poison_examples_transfer/bp_poisons/$f" \
        --pretrain_dataset CIFAR100 --dataset CIFAR10 \
        --poison_setups "poison_setups/cifar10_transfer_learning.pickle" \
        --setup_idx ${f} \
done

### Gradient Matching:

Load code from `git clone https://github.com/aks2203/poisoning-benchmark.git`

Run `python benchmark_gen.py` to get `benchmark_poison_brewing.sh`. This include the command for GM.

for f in {0..19..1};do \
  python brew_poison.py \
      --name bm_base \
      --benchmark poisons/poison_setups_from_scratch.pickle \
      --save benchmark \
      --vruns 0 \
      --ensemble 1 \
      --net ResNet18 \
      --eps 8 \
      --benchmark_idx ${f} \
done

In this project, I actually only run ine trial and use pre-generated 20 trial poisons do futher analysis.

## Testing:

### Feature Collision:

for i in {0..19}; do \
    python3 resnet_benchmark_test.py \
        --poisons_path poison_examples_transfer/fc_poisons/$i \
        --output results/benchmark_fc_$i \
        --dataset cifar10 \
done

### Bullseye Polytope:

for i in {0..19}; do \
    python3 resnet_benchmark_test.py \
        --poisons_path poison_examples_transfer/bp_poisons/$i \
        --output results/benchmark_bp_$i \
        --dataset cifar10 \
done

### Gradient Matching:

for i in {0..19}; do \
python benchmark_test_resnet18.py \
    --poisons_path poison_examples_gm/645_Project_v3/$i  \
    --output results/benchmark_gm_$i  \
    --dataset cifar10 \
done 

## Defense (EPIC)

### Feature Collision:

for f in {0..19}; do \
    python3 train_poison_epic.py \
        --dataset cifar10 \
        --arch resnet18 \
        --poisons_path ../poisoning-benchmark-master/poison_examples_transfer/fc_poisons/$f \
        --scenario transfer \
       --out results/fc/$f \
done

### Bullseye Polytope:

for f in {0..19}; do \
    python3 train_poison_epic.py \
        --dataset cifar10 \
        --arch resnet18 \
        --poisons_path ../poisoning-benchmark-master/poison_examples_transfer/bp_poisons/$f \
        --scenario transfer \
        --out results/bp/$f \
done

### Gradient Matching:

for i in {0..19}; do
  python train_poison_epic.py \
    --gpu-id 0 \
    --dataset cifar10 \
    --arch resnet18 \
    --poisons_path /home/r0liu015/645project_last/poisoning-benchmark/poison_examples_gm/645_Project_v3/$i \
    --epochs 40 \
    --batch-size 128 \
    --subset_size 0.1 \
    --subset_freq 10 \
    --drop_after 30 \
    --out results/epic_gm_eps8_run$i
done






