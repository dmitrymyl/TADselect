"""
example run:
python2 script_TADbit.py infile.txt outfile.txt S2 chr2L 20000 8

From examples folder:

~/anaconda3/envs/tadbit/bin/python ../TADselect/script_TADbit.py ../data/test_S2.20000.chr2L.txt tmp/tadbit_output.txt S2 chr2L 20000 8
"""


from sys import argv

infile = argv[1] # txt matrix
output = argv[2]
exp    = argv[3]
ch     = argv[4]
resolution = int(argv[5]) # in bp
nth = int(argv[6]) # 8

from pytadbit import Chromosome

my_chrom = Chromosome(name=ch,
                      centromere_search=False)
my_chrom.add_experiment(exp,
                        exp_type='Hi-C',
                        identifier=exp,
                        hic_data=infile,
                        resolution=resolution)

my_chrom.find_tad(exp, n_cpus=nth)

experiment = my_chrom.experiments[exp]

experiment.write_tad_borders(savedata=output, density=True)