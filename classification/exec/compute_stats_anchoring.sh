# In this file we create the statistics for each value of beta.
cd ..

for e in 0 10 20 30 40
do
	python gen_all_stats-anchoring.py --folder "log/anchoring-5.0_seed:${e}" --dataset mnist --final --save
	python gen_all_stats-anchoring.py --folder "log/anchoring-5.0_seed:${e}" --dataset notmnist --final --save
	python gen_all_stats-anchoring.py --folder "log/anchoring-5.0_seed:${e}" --dataset kmnist --final --save
done
