export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python -u main.py \
    --target_model_path /home/ubuntu/data/models/vicuna-7b-v1.5 \
    --defense_type None_defense \
    --attack AutoDAN \
    --instructions_path ./data/harmful_bench_1.csv \
    --save_result_path ./exp_results/target_align_autodan/ \
    --agent_recheck \
    --exp_name align_vicuna-7b_autodan

python -u main.py \
    --target_model_path /home/ubuntu/data/models/defense/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack AutoDAN \
    --instructions_path ./data/harmful_bench_1.csv \
    --save_result_path ./exp_results/target_align_autodan/ \
    --agent_recheck \
    --exp_name align_llama-7b_autodan