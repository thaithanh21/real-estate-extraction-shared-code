mkdir -p log_dir/runs/$1
cp output/*.pkl log_dir/runs/$1
python3 train.py --version=$1 --batch_size=$2 --num_epoch=$3
