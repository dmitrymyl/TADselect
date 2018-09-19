import TADcalling

for i in range(5):
    for noise in [0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24]:
        TADcalling.simulate_HiC(mtx_size=4523,
                                min_tadsize=3,
                                max_tadsize=50,
                                min_intertadsize=0,
                                max_intertadsize=2,
                                tads_filename='../data/simTADs.iter{}.noise{}.txt'.format(i + 1, noise * 100),
                                mtx_filename='../data/simMTX.iter{}.noise{}.cool'.format(i + 1, noise * 100),
                                noise=noise)