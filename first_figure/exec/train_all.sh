cd ..

# Train the ensemble
for i in {1..50}; do
    python main_ensemble.py --verbose --id $i --seed $i &
done
wait

# Train DENN method (ours)

# First train the reference function
python main_repulsive.py --verbose --dropout_rate 0.0 --seed 0 --save_folder 'log/ref'
for i in {1..50}; do
    python main_repulsive.py --verbose --batch_size_repulsive 10 --dropout_rate 0.0 --id $i --seed $i --repulsive 'log/ref/models/repulsive_lambda:0.003_5000epochs.pt' --lambda_repulsive 0.003 --save_folder "log/repulsive" &
done
wait


