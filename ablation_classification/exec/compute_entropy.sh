cd ..
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:gaussiannoise --dataset mnist --seed 0
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:gaussiannoise --dataset emnist --seed 0
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:uniformnoise --dataset mnist --seed 0
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:uniformnoise --dataset emnist --seed 0
