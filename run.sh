cd "$(dirname "$0")" # change working directory to directory of this script

bash utils/reset_plots.sh # delete current plots

# obtain new results
for env in Bertrand BertrandDiff
do
    for N in 2 3 4
    do
        for lr in 0.05 0.1 0.5
        do
            for k in 1 2 10
            do
                exp_name=${env}_N-${N}_lr-${lr}_k-${k}
                python run.py --env_name $env --N $N --lr $lr --k $k --exp_name $exp_name --n_steps 100000
            done
        done
    done
done