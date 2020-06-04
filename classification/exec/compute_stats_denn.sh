cd ..

for i in 0 10 20 30 40
do
	python gen_all_stats-denn.py --dataset mnist --final --id $i
	python gen_all_stats-denn.py --dataset notmnist --final --id $i
	python gen_all_stats-denn.py --dataset kmnist --final --id $i
done
