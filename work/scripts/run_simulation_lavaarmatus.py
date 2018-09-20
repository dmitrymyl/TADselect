import TADcalling
import pickle

for i in range(5):
    for noise in [0, 4, 8, 12, 16, 20, 24]:
        for modification in ["", '.filtered', '.nodiag', '.log']:
            print("iteration {} noise {} modification {}".format(i, noise, modification))
            exp_class = TADcalling.Experiment(['Sim_rep1'], ['../data/simulations/simMTX.iter{}.noise{}{}.cool'.format(i, noise, modification)],
                                              'cool', assembly=None, resolution=1, balance=False, chr='Sim',
                                              callername='lavaarmatus', optimisation='simulated',
                                              track_file='../data/true_segmentations/simTADs.iter{}.noise{}.txt'.format(i, noise))
            exp_class.iterative_approach(iterations=7, filename='../data/results/plots/sim.lavaarmatus.iter{}.noise{}{}.png'.format(i, noise, modification))
            pickle_jar = {'best_gamma': exp_class.best_gamma[0],
                          'ranges': exp_class.history['ranges'],
                          'segmentations': exp_class.caller._segmentations,
                          'label': exp_class.caller._metadata['labels'],
                          'mtx_size': TADcalling.InteractionMatrix('../data/simulations/simMTX.iter{}.noise{}{}.cool'.format(i, noise, modification),
                                                                   input_type='cool', balance=False, ch='Sim')._mtx.shape[0]}
            with open('../data/results/pickles/sim.lavaarmatus.iter{}.noise{}{}.pickle'.format(i, noise, modification)) as f:
                pickle.dump(pickle_jar, f)

            exp_class.optimisation_data.to_csv('../data/results/optimisations/sim.lavaarmatus.iter{}.noise{}{}.tsv'.format(i, noise, modification), sep='\t')
            exp_class.background_data.to_csv('../data/results/backgrounds/sim.lavaarmatus.iter{}.noise{}{}.tsv'.format(i, noise, modification), sep='\t')
            exp_class.caller.update_benchmark_df()
            exp_class.caller._benchmark_df.to_csv('../data/results/benchmarks/sim.lavaarmatus.iter{}.noise{}{}.tsv'.format(i, noise, modification), sep='\t')
