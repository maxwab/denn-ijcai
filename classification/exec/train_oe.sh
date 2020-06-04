cd ..
for i in {0..4} 
do
    python oe_scratch.py --save "log/oe_retrained_lambda:100_seed:${i}" --learning_rate 0.001 --batch_size 64 --oe_batch_size 128 --lambda_oe 100 --seed $i &
done
wait
