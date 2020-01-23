#!/bin/bash

pip install --upgrade wandb
wandb login c3e78e152c40028f867ae8dc709b15fbc2d4b7a5

printf "\nStarting: "
date

# Ensure that our file encodings are correct and that we have permission to execute.
for file in $PWD
do 
	# Make sure all python files are good to go.
	if [[ $file == *.py ]]
	then
		dos2unix $file 2> /dev/null
		chmod 777 $file
	fi

	# Remove old error logs.
	if [[ $file == *err.txt ]]
	then 
    	rm $file
	fi

	# Remove old outputs.
	if [[ $file == *out.txt ]]
	then 
    	rm $file
	fi
done

# Run our python code.
# argv -> 1. LOSS 2. GPU 3. LR 4. COVD 5. MP
nohup python -u train_cls.py 'gmm_nll_loss' 2 1e-4 1e-5 0.10 >out2.txt 2>err2.txt &
sleep 5
nohup python -u train_cls.py 'gmm_nll_loss' 1 1e-4 1e-5 0.05 >out1.txt 2>err1.txt &
sleep 5
nohup python -u train_cls.py 'gmm_nll_loss' 0 1e-4 1e-5 0.01 >out0.txt 2>err0.txt &

wait

printf "\nTerminating: "
date

# Let the user know if they need to check for any outputted errors or warnings.
if [ -s "err.txt" ]
then 
   echo "Check err.txt for logged errors."
fi