
exps=("halfcheetah")
encode_dim_exp=("16 24 8")

# for (( exp_index = 0 ; exp_index < ${#exps[@]} ; exp_index++ ))
for exp_index in `seq 0 ${#exps[@]}`
do
	exp_name=${exps[$exp_index]}
	encode_dim_buf=${encode_dim_exp[$exp_index]}
		
	for encode_dim in $encode_dim_buf
	do
		echo $exp_name $encode_dim
		python LSTM_PAD_type1.py --exp $exp_name --encode_dim $encode_dim
	done
done