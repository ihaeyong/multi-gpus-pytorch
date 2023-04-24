export PYTHONPATH=/mnt/haeyong/multi-gpus
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1,$2"
python3 -um torch.distributed.launch \
        --nproc_per_node=2 --master_port=23423 \
        train_mnist.py
