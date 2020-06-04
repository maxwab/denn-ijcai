cd ..

# Train the ensemble
for e in 0 10 20 30 40
do
	for i in {0..9}
	do	
		python train.py --train mnist --save_folder log/anchoring-5.0_seed:${e} --id $((e+i)) --seed $((e+i)) --lambda_anchoring 5.0 &
	done
	wait
done

