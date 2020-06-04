cd ..

for i in 0 10 20 30 40
do
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+0)) --disable-cuda --id $(($i+0)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}" &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+1)) --disable-cuda --id $(($i+1)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}" &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+2)) --disable-cuda --id $(($i+2)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"   &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+3)) --disable-cuda --id $(($i+3)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"  &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+4)) --disable-cuda --id $(($i+4)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"  &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+5)) --disable-cuda --id $(($i+5)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"  &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+6)) --disable-cuda --id $(($i+6)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"  &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+7)) --disable-cuda --id $(($i+7)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"  &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+8)) --disable-cuda --id $(($i+8)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"  &
	sleep 1
	python train.py --train mnist --repulsive fashionmnist --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_0_10epochs.pt --lambda_repulsive 0.05 --bandwidth_repulsive 10 --seed $(($i+9)) --disable-cuda --id $(($i+9)) --save_folder "log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:${i}"  &
	wait
done
