cd ..
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:gaussiannoise_l:100._b:1. --dataset mnist --seed 0 --final
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:gaussiannoise_l:100._b:1. --dataset notmnist --seed 0 --final
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:gaussiannoise_l:100._b:1. --dataset kmnist --seed 0 --final
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:uniformnoise_l:50._b:10. --dataset mnist --seed 0 --final
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:uniformnoise_l:50._b:10. --dataset notmnist --seed 0 --final
python compute_entropy_values.py --folder cv --prefix train:mnist_repulsive:uniformnoise_l:50._b:10. --dataset kmnist --seed 0 --final
