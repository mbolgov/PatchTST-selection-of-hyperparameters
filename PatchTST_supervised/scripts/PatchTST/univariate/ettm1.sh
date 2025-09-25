if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

seq_len=336
patch_len=16
stride=8
model_name=PatchTST

root_path_name=./dataset/
data_path_name=ETTm1.csv
data_name=ETTm1

if [ ! -d "./logs/LongForecasting/univariate/$data_name" ]; then
    mkdir ./logs/LongForecasting/univariate/$data_name
fi

random_seed=2021
for pred_len in 96 192 336 720
do
    if [ ! -d "./logs/LongForecasting/univariate/$data_name/$pred_len" ]; then
        mkdir ./logs/LongForecasting/univariate/$data_name/$pred_len
    fi

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $pred_len'_'$seq_len'_'$patch_len'_'$stride \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len $patch_len\
      --stride $stride\
      --des 'Exp' \
      --train_epochs 50\
      --patience 20\
      --lradj 'warmup_decay'\
      --pct_start 0.4\
      --itr 1 --batch_size 128 --learning_rate 0.00001 >logs/LongForecasting/univariate/$data_name/$pred_len/$model_name'_fS_'$seq_len'_'$patch_len'_'$stride.log
done
