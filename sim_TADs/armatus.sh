for noise in 4 8 12 16 20;
do
	for sim in 1 2 3 4 5;
	do
		for gamma in `seq 0 0.5 5`;
		do
		echo $noise, $sim, $gamma
		armatus -i matrices/simHiC_countMatrix_noise${noise}_sim${sim}.txt.gz -g $gamma -o yielded/armatus_gamma${gamma}_noise${noise}_sim${sim} -r 40000
		done
	done
done
