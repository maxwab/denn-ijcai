cd ..

for e in 0 10 20 30 40
do
	for i in {0..9}
	do	
		python train.py --train mnist --save_folder "log/prior_bs-50.0_seed:${e}" --id $((e+i)) --seed $((e+i)) --beta 50.0 --bootstrapping &
	done
	wait
done
