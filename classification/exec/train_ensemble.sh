cd ..

for e in 0 10 20 30 40
do
	for i in {0..9}
	do	
		python train.py --train mnist --save_folder log/deep_ensemble_seed:$e --seed $((e+i)) --id $((e+i)) &
	done
	wait
done
