cd ..
counter=0
for REPULSIVE in 'uniformnoise' 'gaussiannoise'
do
    for lval in 0.001 0.005 0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.
    do
        for bval in 0.01 0.1 1. 10.
        do
            for i in {5..9}
            do
                python train.py \
                    --train mnist \
                    --repulsive $REPULSIVE \
                    --reference_net log/reference/models/repulsive_train:mnist_repulsive:None_lambda:0.0_bandwidth:0.0_10epochs.pt \
                    --lambda_repulsive $lval \
                    --bandwidth_repulsive $bval \
                    --seed $i \
                    --noise_factor 1.0 \
                    --disable-cuda \
                    --id $i  \
                    --save_folder "cv/train:mnist_repulsive:${REPULSIVE}_l:${lval}_b:${bval}" &
                # Increment counter and wait if we have reacher full capacity of the node
                counter=$(($counter + 1))
                if [ $(($counter % 24)) = 0 ]
                then
                    wait
                fi
            done
        done
    done
done
wait
