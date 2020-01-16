cd ..
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 0 --id 0 --gpu 0 --comet &
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 1 --id 1 --gpu 0 --comet &
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 2 --id 2 --gpu 1 --comet &
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 3 --id 3 --gpu 1 --comet &
wait
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 4 --id 4 --gpu 0 --comet &
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 5 --id 5 --gpu 0 --comet &
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 6 --id 6 --gpu 1 --comet &
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 7 --id 7 --gpu 1 --comet &
wait
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 8 --id 8 --gpu 0 --comet &
python train_policy_color.py --train datasets/red_sphere --save_folder log/ensemble_train:red_sphere --seed 9 --id 9 --gpu 0 --comet &
wait
