python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-papers100M --dataset ogbn-papers100M --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-papers100M --dataset ogbn-papers100M --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-papers100M --dataset ogbn-papers100M --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-papers100M --dataset ogbn-papers100M --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16


python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-arxiv --dataset ogbn-arxiv --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-arxiv --dataset ogbn-arxiv --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-arxiv --dataset ogbn-arxiv --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-arxiv --dataset ogbn-arxiv --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16


python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition reddit --dataset reddit --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition reddit --dataset reddit --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition reddit --dataset reddit --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition reddit --dataset reddit --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16


python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-products --dataset ogbn-products --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-products --dataset ogbn-products --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-products --dataset ogbn-products --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name graphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbn-products --dataset ogbn-products --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16


python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbl-citation2 --dataset ogbl-citation2 --hidden_dims 64 --num_layers 2 --epochs 50 --fanout 25 10 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16

python -m  partitioner.embeddings --mode cpu --model_name linkgraphsage --batch_size 8192 --partitioner k-means --num_parts 2 4 8 16 32 --dataset_to_partition ogbl-citation2 --dataset ogbl-citation2 --hidden_dims 64 --num_layers 3 --epochs 50 --fanout 15 10 5 --results_dir /mnt/reordered/emebddings/1 --max_vertex_balance 1.1 --max_training_balance 1.05 --out_size 16


