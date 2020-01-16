#!/bin/bash

printf "\nStarting: "
date

# Ensure that our file encodings are correct and that we have permission to execute.
for file in $PWD
do 
	# Make sure all python files are good to go.
	if [[ $file == *.py ]]
	then
		dos2unix $file #2> /dev/null
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
nohup python -u train_cls.py 'gmm_nll_loss' 2 >outGMM.txt 2>errGMM.txt &
sleep 5
nohup python -u train_cls.py 'chamfer_loss' 1 >outCHAMFER.txt 2>errCHAMFER.txt &
sleep 5
nohup python -u train_cls.py 'sorted_mse_loss' 0 >outMSE.txt 2>errMSE.txt &

wait

printf "\nTerminating: "
date

# Let the user know if they need to check for any outputted errors or warnings.
if [ -s "err.txt" ]
then 
   echo "Check err.txt for logged errors."
fi