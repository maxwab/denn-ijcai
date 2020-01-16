cd ..
# Train the ref/modelserence function
python main_repulsive.py --verbose --dropout_rate 0.0 --seed 0 --save_folder 'log/ref'


# Train the repulsive functions
for lv in 0. 0.0001 0.001 0.01
do
    echo "Computing for lambda repulsive = $lv"
    for i in {1..50}
    do
        python main_repulsive.py --verbose --batch_size_repulsive 10 --dropout_rate 0.0 --id $i --seed $i --repulsive 'log/ref/models/repulsive_lambda:0.003_5000epochs.pt' --lambda_repulsive $lv --save_folder "log/lv$lv" &
    done
    wait
done
