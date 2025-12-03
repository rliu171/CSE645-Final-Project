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
for f in {0..19..1};do
    python3 ./poison_crafting/craft_poisons_bp.py 
    --poisons_path "poison_examples_transfer/bp_poisons/$f" 
    --pretrain_dataset CIFAR100 --dataset CIFAR10  
    --poison_setups "poison_setups/cifar10_transfer_learning.pickle" 
    --setup_idx ${f}
done

### Gradient Matching:
Load code from `git clone https://github.com/aks2203/poisoning-benchmark.git`
Run `python benchmark_gen.py` to get `benchmark_poison_brewing.sh`. This include the command for GM.

for f in {0..19..1};do
  python brew_poison.py 
  --name bm_base 
  --benchmark poisons/poison_setups_from_scratch.pickle 
  --save benchmark 
  --vruns 0 
  --ensemble 1 
  --net ResNet18  
  --eps 8 
  --benchmark_idx 0 
done

In this project, I only run ine trial and use pre-generated 20 trial poisons do futher analysis.




