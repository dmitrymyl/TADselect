from . import CallerClasses, DataClasses
from .utils import *
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

caller_dict = {'armatus': CallerClasses.ArmatusCaller,
               'lavaburst': CallerClasses.LavaburstCaller,
               'lavaarmatus': CallerClasses.LavaArmatusCaller,
               'lavamodularity': CallerClasses.LavaModularityCaller,
               'lavavariance': CallerClasses.LavaVarianceCaller,
               'lavacorner': CallerClasses.LavaCornerCaller,
               'insulation': CallerClasses.InsulationCaller,
               'directionality': CallerClasses.DirectionalityCaller,
               'hicseg': CallerClasses.HiCsegCaller,
               'hicseg_p': CallerClasses.HiCsegPCaller,
               'hicseg_g': CallerClasses.HiCsegGCaller,
               'hicseg_b': CallerClasses.HiCsegBCaller,
               'mrtadfinder': CallerClasses.MrTADFinderCaller,
               'hicexplorer': CallerClasses.HiCExplorerCaller}

default_funcs = {'simulated': ['TPR TADs', 'TPR boundaries', 'PPV TADs', 'PPV boundaries'],
                 'convergence': ['JI TADs', 'OC TADs', 'JI boundaries', 'OC boundaries'],
                 'border_events': ['P-value']}

set_scale_factors = {'armatus': {'narrow': 2, 'wide': 5, 'factor': 5},
                     'lavaburst': {'narrow': 2, 'wide': 5, 'factor': 5},
                     'lavaarmatus': {'narrow': 2, 'wide': 5, 'factor': 5},
                     'lavamodularity': {'narrow': 2, 'wide': 5, 'factor': 5},
                     'insulation': {'narrow': 2, 'wide': 5, 'factor': 5},
                     'directionality': {'narrow': 1, 'wide': 3, 'factor': 4}}


class Experiment(object):

    def __init__(self, datasets_labels, datasets_files, data_format, chr, callername, track_file=None, scaling=False, **kwargs):

        mode = kwargs.get('mode', 'iterative')
        background_method = kwargs.get('background_method', 'size')
        optimisation = kwargs.get('optimisation', 'convergence')
        resolution = kwargs.get('resolution', 1)

        if mode not in ('iterative', 'user'):
            raise Exception("Mode not understood: %s" % mode)
        else:
            self.mode = mode

        if background_method not in ('size', 'dispersion'):
            raise Exception("Background method not understood: %s" % background_method)
        else:
            self.background_method = background_method

        if optimisation not in ('convergence', 'simulated', 'border_events', 'fitting-average'):
            raise Exception("optimisation method not understood: %s" % optimisation)
        else:
            self.optimisation = optimisation

        if callername not in caller_dict.keys():
            raise Exception("Caller not understood: %s" % caller)
        else:
            self.caller = caller_dict[callername](datasets_labels, datasets_files, data_format, chr=chr, **kwargs)
            self.callername = callername

        if scaling:
            data_scale = self.caller._metadata['resolution']
        else:
            data_scale = 1

        if track_file:
            self.track = DataClasses.load_BED(track_file, scale=data_scale, chrm=chr)[self.caller._metadata['chr']]
        else:
            self.track = None

        self.default_ranges = {'armatus': pd.Series([np.arange(-5, 5.5, 0.5)], index=['gamma']),
                               'lavaburst': pd.Series([np.arange(-5, 5.5, 0.5)], index=['gamma']),
                               'lavaarmatus': pd.Series([np.arange(-5, 5.5, 0.5)], index=['gamma']),
                               'lavavariance': pd.Series([np.arange(-5, 5.5, 0.5)], index=['gamma']),
                               'lavacorner': pd.Series([np.arange(-5, 5.5, 0.5)], index=['gamma']),
                               'lavamodularity': pd.Series([np.arange(5, 15.5, 0.5)], index=['gamma']),
                               'insulation': pd.Series([np.arange(1, 11, 1) * self.caller._metadata['resolution'],
                                                        [0.1, 0.2, 0.5, 0.7]], index=['window', 'cutoff']),
                               'directionality': pd.Series([np.arange(1, 101, 10) * self.caller._metadata['resolution'],
                                                            [0.1, 0.2, 0.5, 0.7]], index=['window', 'cutoff']),
                               'hicseg': None,
                               'mrtadfinder': None,
                               'hicexplorer': None}

        self.scale_factors = set_scale_factors[callername]

        self.history = {'ranges': [deepcopy(self.default_ranges[self.callername])],
                        'best_gamma': list(),
                        'best_func': list(),
                        'iteration': 0}  # dictionary for history events
        self.optimisation_data = pd.DataFrame()  # handles TPR, PPV, JI, OC...
        self.background_data = pd.DataFrame()  # handles mean size or dispersions

        # min and max mean size of TADs in bins, user is able to redefine them.
        self.profile = {'size': [2, 100],
                        'dispersion': [0.05, 0.95],
                        'midpoint': 500000 / self.caller._metadata['resolution']}

    @staticmethod
    def data_generation(caller, obtained_gamma_range, mode):
        """
        Generates data from segmentations of caller specific to a certain mode.
        :param caller: a CallerClasses class with generated segmentations
        :param obtained_gamma_range: a pd.Series with range of parameter
        :param mode: simulated, convergence, border_events, sizes
        """
        if obtained_gamma_range.shape[0] == 1:
            gamma_range = list(deepcopy(obtained_gamma_range[0]))
            arr_shape = len(gamma_range)
        else:
            gamma_range = list(product(*obtained_gamma_range))
            arr_shape = [len(dim) for dim in obtained_gamma_range]

        if mode in ('simulated', 'border_events', 'fitting-average'):
            label = caller._metadata['labels'][0]
            segmentations = [caller._segmentations[label][gamma] for gamma in gamma_range]
            return [segmentations], arr_shape, gamma_range

        elif mode == 'convergence':
            label_rep1 = list(filter(lambda i: "rep1" in i, caller._metadata['labels']))[0]
            label_rep2 = list(filter(lambda i: "rep2" in i, caller._metadata['labels']))[0]
            segmentations_rep1 = [caller._segmentations[label_rep1][gamma] for gamma in gamma_range]
            segmentations_rep2 = [caller._segmentations[label_rep2][gamma] for gamma in gamma_range]
            return [segmentations_rep1, segmentations_rep2], arr_shape, gamma_range

        elif mode == 'sizes':
            segmentation_sizes = [[caller._segmentations[label][gamma].sizes for gamma in gamma_range]
                                  for label in caller._metadata['labels']]
            return segmentation_sizes, arr_shape, gamma_range

        else:
            raise Exception("Mode for data generation not understood: %s" % mode)

    @staticmethod
    def background_calc(segmentation_sizes, arr_shape, background_method='size'):
        """
        Calculates background function: mean size of TADs or
        dispersion of sizes for each segmentation.
        :param segmentation_sizes: nested list of segmentation sizes
        :param arr_shape: shape of data = [gamma1, gamma2, ...]
        :param background_method: a background function to be calculated
        """
        if background_method == 'size':
            return np.array([np.reshape([np.mean(i) for i in labelled], arr_shape) for labelled in segmentation_sizes])
        elif background_method == 'dispersion':
            return np.array([np.reshape([np.std(i) for i in labelled], arr_shape) for labelled in segmentation_sizes])
        else:
            raise Exception("Background method not understood: %s" % background_method)

    @staticmethod
    def optimised_calc(segmentation_list, arr_shape, optimisation, **kwargs):
        """
        Calculates function to be optimised: TPRs and FDRs for
        simulated segmentations, convergence between two replica
        or p-values of distances to the closest genome features based
        on track file.
        :param segmentation_list: contains one or two nested lists with segmentations
        :param arr_shape: shape of data = [gamma1, gamma2, ...]
        :param optimisation: an optimisation mode to be calculated
        :param track: a GenomicRanges instance of some genomic track to be used
        """
        offset = kwargs.get('offset', 0)
        track = kwargs.get('track', None)
        if optimisation == 'simulated':
            track = kwargs.get('track', None)
            return list(map(np.array, (np.reshape([segmentation.count_coef(track, coef=function, offset=offset)
                                                   for segmentation in segmentation_list[0]], arr_shape)
                                       for function in ('TPR TADs', 'PPV TADs', 'TPR boundaries', 'PPV boundaries'))))

        elif optimisation == 'convergence':
            return list(map(np.array, [np.reshape([segmentation_list[0][i].count_coef(segmentation_list[1][i], coef=function, offset=offset)
                                                   for i in range(len(segmentation_list[0]))], arr_shape)
                                       for function in ('JI TADs', 'OC TADs', 'JI boundaries', 'OC boundaries')]))

        elif optimisation == 'border_events':
            distances = list()
            for segmentation in segmentation_list[0]:
                dists = segmentation.dist_closest(track, mode='bin-boundariwise')
                if dists is not None:
                    distances.append(dists)
                else:
                    distances.append(np.inf)
            #distances = [segmentation.dist_closest(track, mode='bin-boundariwise').flatten()
            #             for segmentation in segmentation_list[0]]
            return [np.array([-np.mean(np.abs(i)) for i in distances])]

        elif optimisation == 'fitting-average':
            midpoint = kwargs.get('average', 100)
            return [np.reshape([-np.abs(midpoint - np.mean(segmentation.sizes))
                               for segmentation in segmentation_list[0]], arr_shape)]
        else:
            raise Exception("optimisation method not understood: %s" % optimisation)

    @staticmethod
    def max_coord(target_arr, background_arr, mask_list, threshold):
        """
        Find coordinates of maximum values in target_arr
        based on mask from background_arr values.
        """
        if target_arr.shape[0] != background_arr.shape[1]:
            raise Exception("Shapes of target and background arrays are inconsistent: {} vs {}".format(target_arr.shape, background_arr.shape))
        mask = (background_arr <= mask_list[0]) | (background_arr >= mask_list[1])
        if len(background_arr.shape) > 1 and background_arr.shape[0] > 1:
            mask = np.multiply(*mask)
        else:
            mask = mask[0]
        v1 = target_arr.copy()
        print(np.sum(np.gradient(v1.flatten())), np.sum(v1), 0.9 * v1.flatten().shape[0], v1.shape)
        if np.abs(np.sum(np.gradient(v1.flatten()))) < threshold and np.sum(v1) > 0.9 * v1.flatten().shape[0]:
            coord = v1.flatten().shape[0] // 2
        else:
            v1[mask] = np.NINF
            v1_rev = v1.flatten()[::-1]
            v1_rev.shape = v1.shape
            coord = len(v1_rev.flatten()) - v1_rev.argmax() - 1
        return np.unravel_index(coord, v1.shape)

    @staticmethod
    def select_function(optimising_list):
        if len(optimising_list) == 1:
            return 0
        elif len(optimising_list) > 1:
            gradient_list = [np.gradient(func) for func in optimising_list]
            convoluted_list = [np.sum(np.abs(grad)) for grad in gradient_list]
            rank_arr = np.argsort(-np.array(convoluted_list))
            best_func = np.argmin(rank_arr)
            return best_func
        else:
            raise Exception("optimising list is of unexpected length: %d" % len(optimising_list))

    def make_newrange(self, oldrange, optimising_list, background_arr, background_method, **kwargs):
        """
        Return new range of gammas based on old range, values of optimised
        function, background function and background method.
        """
        mode = kwargs.get('mode', 'primary')
        threshold = kwargs.get('threshold', 0.001)
        scale_factors = kwargs.get('scale_factors', {'narrow': 2, 'wide': 5, 'factor': 5})
        narrow = scale_factors['narrow']
        wide = scale_factors['wide']
        factor = scale_factors['factor']

        if mode == 'gradient-selection':
            best_func = Experiment.select_function(optimising_list)
            target_arr = optimising_list[best_func]
        elif mode == 'sum-maximization':
            target_arr = sum(optimising_list)
            best_func = -1
        elif mode == 'primary':
            best_func = 0
            target_arr = optimising_list[0]
        else:
            raise Exception("Mode not understood: %s" % mode)

        if background_method == 'dispersion':
            raise Exception("Dispersion not implemented!")

        elif background_method == 'size':
            range_list = list()
            print(oldrange, target_arr, background_arr)
            max_index = Experiment.max_coord(target_arr, background_arr, self.profile['size'], threshold=threshold)
            best_gamma = list()
            if not isinstance(max_index, list) and not isinstance(max_index, tuple):
                max_index = [max_index]
            for dim in range(oldrange.shape[0]):
                max_loc = max_index[dim]
                best_gamma.append(oldrange[dim][max_loc])
                old_step = oldrange[dim][1] - oldrange[dim][0]
                if 0.1 >= (max_loc / target_arr.shape[dim]):
                    left = oldrange[dim][0] - wide * old_step
                    right = oldrange[dim][max_loc] + narrow * old_step
                    step = old_step

                elif (max_loc / target_arr.shape[dim]) >= 0.9:
                    left = oldrange[dim][max_loc] - narrow * old_step
                    right = oldrange[dim][-1] + wide * old_step
                    step = old_step

                else:
                    left = oldrange[dim][max_loc] - narrow * old_step
                    right = oldrange[dim][max_loc] + narrow * old_step
                    step = old_step / factor

                range_list.append(np.arange(left, right + step, step).round(5).tolist())
            return pd.Series(data=[i for i in np.array(range_list)], index=oldrange.index), tuple(best_gamma), best_func

        else:
            raise Exception("Background method not understood: %s" % background_method)

    @staticmethod
    def chain_multiindex(arr1, arr2):
        """
        Takes 1-dim arr1 and n-dim arr2 and returns
        pd.Multiindex of product(arr1, arr2).
        In case n > 1, flattens items of
        product(arr1, arr2).
        """
        prod = list(product(arr1, arr2))
        try:
            iter(prod[0][1])
            v2 = [[item[0]] + list(item[1]) for item in prod]
            names = ['label'] + ['gamma{}'.format(i + 1) for i in range(len(v2[0]) - 1)]
            return pd.MultiIndex.from_tuples(v2, names=names)
        except TypeError:
            return pd.MultiIndex.from_tuples(prod, names=['label', 'gamma1'])

    @staticmethod
    def plot_tads(mtx, tads, bgn=0, end=250, fname=None, plot_size=None):
        """
        Plot given matrix and segmentaion in given bin coordinates.
        Optionally saves figure to file and redefine figure size.
        :param mtx: Cooler matrix
        :param tads: GenomicRanges segmentation
        :param bgn: the first bin of the matrix to be plotted
        :param end: the last of the matrix to be plotted
        :param fname: if given, the plot will be saved there
        :param plot_size: if given (type list), the plot will be of that size
        """
        tads_color = 'blue'

        # plot tuning
        if plot_size:
            plt.figure(figsize=plot_size)
        sns.heatmap(mtx[bgn:end, bgn:end], cmap='Reds')
        plt.xticks([])
        plt.yticks([])

        for i in tads:
            tad_bgn = i[0] - bgn
            tad_end = i[1] - bgn
            plt.plot([tad_bgn, tad_end], [tad_bgn, tad_bgn], color=tads_color)
            plt.plot([tad_end, tad_end], [tad_bgn, tad_end], color=tads_color)

        if fname:
            plt.savefig(fname)

    @staticmethod
    def plot_one_dim(caller, obtained_gamma_range, optimisation_data, background_data, best_gamma, filename=None):
        """
        Plots three-panel plot: optimisation data, background data for given range of gammas
        and best segmentation concerning both data.
        :param caller:
        :param obtained_gamma_range: 1-dim pd.Series
        :param optimisation_data:
        :param background_data:
        :param best_gamma:
        """
        mtx_1 = cooler.Cooler(caller._metadata['files_cool'][0]).matrix(balance=caller._metadata['balance'],
                                                                        as_pixels=False).fetch(caller._metadata['chr'],
                                                                                               caller._metadata['chr'])
        label = caller._metadata['labels'][0]
        best_segmentation = caller._segmentations[label][best_gamma[0]].data
        plt.rcParams['figure.figsize'] = 10, 10
        plt.subplot(221)
        if 'Distance' not in optimisation_data.columns.values:
            plt.ylim(0, 1.01)            
        plt.plot(optimisation_data.loc[label].loc[obtained_gamma_range[0]], alpha=0.7)
        plt.legend(labels=optimisation_data.loc[label].columns)
        plt.subplot(222)
        plt.plot(background_data.loc[label].loc[obtained_gamma_range[0]])
        plt.legend(labels=background_data.loc[label].columns)
        plt.subplot(223)
        Experiment.plot_tads(mtx_1, best_segmentation, bgn=1000, end=1100)
        plt.title('Best segmentation with gamma{}'.format(best_gamma))
        if filename:
            plt.savefig(filename)

    @staticmethod
    def plot_two_dim(caller, obtained_gamma_range, optimisation_data, background_data, best_gamma, background_method, filename=None):
        """
        Plots six-panel plot: optimisation data, background data for given range of gammas
        and best segmentation concerning both data in slice of the first two gammas.
        :param caller:
        :param obtained_gamma_range: n-dim pd.Series
        :param optimisation_data:
        :param background_data:
        :param best_gamma:
        :param background_method:
        """
        mtx_1 = cooler.Cooler(caller._metadata['files_cool'][0]).matrix(balance=caller._metadata['balance'],
                                                                        as_pixels=False).fetch(caller._metadata['chr'],
                                                                                               caller._metadata['chr'])
        label = caller._metadata['labels'][0]
        best_segmentation = caller._segmentations[label][best_gamma].data
        heatmap_source = optimisation_data.unstack(level=0).unstack(level=0).groupby('gamma2').aggregate(np.mean)
        plt.rcParams['figure.figsize'] = 15, 10

        if 'Distance' in optimisation_data.columns.values:
            vmin = None
            vmax = 0
        else:
            vmin = 0
            vmax = 1
        for i, func in zip((1, 2, 4, 5), optimisation_data.columns):
            plt.subplot(2, 3, i)
            sns.heatmap(heatmap_source[func][label].loc[obtained_gamma_range[1],
                                                        obtained_gamma_range[0]],
                        cmap='Reds', center=0.5, vmin=vmin, vmax=vmax)
            plt.title(func)

        background_source = background_data.unstack(level=0).unstack(level=0).groupby('gamma2').aggregate(np.mean)
        plt.subplot(2, 3, 3)
        sns.heatmap(background_source['size'][label].loc[obtained_gamma_range[1],
                                                         obtained_gamma_range[0]], cmap='Reds')
        plt.title(background_method)
        plt.subplot(2, 3, 6)
        Experiment.plot_tads(mtx_1, best_segmentation, bgn=1000, end=1100)
        plt.title('Best segmentation with gamma{}'.format(best_gamma))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        if filename:
            plt.savefig(filename)

    @staticmethod
    def append_data(data1, data2):
        """
        Adds data2 to data1 and return new DataFrame.
        In case of duplicate indices those of data2
        will be removed.
        """
        buff = data1.append(data2)
        return buff[~buff.index.duplicated(keep='first')]

    def call(self, **kwargs):
        """
        Perform one segmentation call in given range of gammas.
        Estimate new range of gammas, best gamma in current range.
        Plot optimisation and background data as well as the
        best current segmentation.
        """
        mode = kwargs.get('mode', 'primary')
        regime = kwargs.get('regime', 'silent')
        offset = kwargs.get('offset', 0)
        background_method = kwargs.get('background_method', self.background_method)

        self.caller.call(self.history['ranges'][-1])

        segmentation_sizes, arr_shape, gamma_arr = Experiment.data_generation(self.caller, self.history['ranges'][-1], 'sizes')
        background_arr = Experiment.background_calc(segmentation_sizes, arr_shape, background_method=background_method)
        segmentation_list, arr_shape, gamma_arr = Experiment.data_generation(self.caller, self.history['ranges'][-1], self.optimisation)
        optimising_list = Experiment.optimised_calc(segmentation_list, arr_shape, self.optimisation, track=self.track, average=self.profile['midpoint'], offset=offset)
        new_range, best_gamma, best_func_id = self.make_newrange(self.history['ranges'][-1], optimising_list,
                                                                 background_arr, self.background_method, mode=mode,
                                                                 scale_factors=self.scale_factors)
        # assign new gammas
        self.best_gamma = best_gamma
        self.history['iteration'] += 1
        # save values in history
        best_func = best_func_id
        for key, value in zip(('ranges', 'best_gamma', 'best_func'), (new_range, best_gamma, best_func)):
            self.history[key].append(deepcopy(value))
        # print result
        if regime != 'silent':
            print("new range is:\n{}".format(self.history['ranges'][-1]))
            print("best gamma value is: {}".format(best_gamma))
            print("function used for optimisation is: {}".format(best_func))

        # Prepare dataframes of obtained values to handle them.
        if self.optimisation == 'simulated':
            columns = ('TPR TADs', 'PPV TADs', 'TPR boundaries', 'PPV boundaries')
            opt_index = Experiment.chain_multiindex(self.caller._metadata['labels'], gamma_arr)
        elif self.optimisation == 'convergence':
            columns = ('JI TADs', 'OC TADs', 'JI boundaries', 'OC boundaries')
            opt_index = Experiment.chain_multiindex([self.caller._metadata['labels'][0]], gamma_arr)
        elif self.optimisation == 'fitting-average':
            columns = ['Difference']
            opt_index = Experiment.chain_multiindex(self.caller._metadata['labels'], gamma_arr)
        elif self.optimisation == 'border_events':
            columns = ['Distance']
            opt_index = Experiment.chain_multiindex(self.caller._metadata['labels'], gamma_arr)
        else:
            raise Exception("optimisation not understood: %s" % self.optimisation)

        optimisation_handler = {func: pd.Series(optimising_list[i].flatten(), index=opt_index)
                                for func, i in zip(columns, range(len(columns)))}
        self.optimisation_data = Experiment.append_data(self.optimisation_data, pd.DataFrame(optimisation_handler))

        background_index = Experiment.chain_multiindex(self.caller._metadata['labels'], gamma_arr)
        background_handler = {background_method: pd.Series(background_arr.flatten(), index=background_index)}
        self.background_data = Experiment.append_data(self.background_data, pd.DataFrame(background_handler))

        if regime != 'silent':
            if self.history['ranges'][-2].shape[0] == 1:  # if gamma is 1-dim
                Experiment.plot_one_dim(self.caller,
                                        self.history['ranges'][-2],
                                        self.optimisation_data,
                                        self.background_data,
                                        self.best_gamma)

            elif self.history['ranges'][-2].shape[0] > 1:  # if gamma is n-dim
                Experiment.plot_two_dim(self.caller,
                                        self.history['ranges'][-2],
                                        self.optimisation_data,
                                        self.background_data,
                                        self.best_gamma,
                                        background_method)

    def iterative_approach(self, **kwargs):
        threshold = kwargs.get('threshold', 0.01)
        iterations = kwargs.get('iterations', 5)
        cutoff = kwargs.get('cutoff', 0.95)
        plt_filename = kwargs.get('filename', 'sample.png')
        offset = kwargs.get('offset', 0)
        step = self.history['ranges'][-1][0][1] - self.history['ranges'][-1][0][0]
        best_value = -1
        while step > threshold and self.history['iteration'] < iterations and best_value < cutoff:
            self.call(offset=offset)
            best_value = self.optimisation_data.loc[self.caller._metadata['labels'][0]].loc[self.best_gamma][0]
            print("best gamma value is: {}".format(self.best_gamma))
            print("new range is:\n{}".format(self.history['ranges'][-1]))

        if self.history['ranges'][-2].shape[0] == 1:  # if gamma is 1-dim
            Experiment.plot_one_dim(self.caller,
                                    self.history['ranges'][-2],
                                    self.optimisation_data,
                                    self.background_data,
                                    self.best_gamma,
                                    filename=plt_filename)

        elif self.history['ranges'][-2].shape[0] > 1:  # if gamma is n-dim
            Experiment.plot_two_dim(self.caller,
                                    self.history['ranges'][-2],
                                    self.optimisation_data,
                                    self.background_data,
                                    self.best_gamma,
                                    self.background_method,
                                    filename=plt_filename)


class ExperimentNoGamma(object):

    def __init__(self, datasets_labels, datasets_files, data_format, chr, callername, key=None, track_file=None, scaling=False, **kwargs):

        mode = kwargs.get('mode', 'iterative')
        background_method = kwargs.get('background_method', 'size')
        optimisation = kwargs.get('optimisation', 'convergence')
        self._key = key

        if mode not in ('iterative', 'user'):
            raise Exception("Mode not understood: %s" % mode)
        else:
            self.mode = mode

        if background_method not in ('size', 'dispersion'):
            raise Exception("Background method not understood: %s" % background_method)
        else:
            self.background_method = background_method

        if optimisation not in ('convergence', 'simulated', 'border_events', 'fitting-average'):
            raise Exception("optimisation method not understood: %s" % optimisation)
        else:
            self.optimisation = optimisation

        if callername not in caller_dict.keys():
            raise Exception("Caller not understood: %s" % callername)
        else:
            self.caller = caller_dict[callername](datasets_labels, datasets_files, data_format, chr=chr, **kwargs)
            self.callername = callername

        if scaling:
            scale = self.caller._metadata['resolution']
        else:
            scale = 1

        if track_file:
            self.track = DataClasses.load_BED(track_file, scale=scale)[self.caller._metadata['chr']]
        else:
            self.track = None

        self.results = {'optimisation': None, 'background': None}
        self.profile = {'size': [2, 100],
                        'dispersion': [0.05, 0.95],
                        'midpoint': 500000 / self.caller._metadata['resolution']}

    @staticmethod
    def data_generation(caller, mode, key=None):
        """
        Generates data from segmentations of caller specific to a certain mode.
        :param caller: a CallerClasses class with generated segmentations
        :param mode: simulated, convergence, border_events, sizes
        """
        if mode in ('simulated', 'border_events'):
            label = caller._metadata['labels'][0]
            segmentations = [caller._segmentations[label]]
            return segmentations

        elif mode == 'convergence':
            label_rep1 = list(filter(lambda i: "rep1" in i, caller._metadata['labels']))[0]
            label_rep2 = list(filter(lambda i: "rep2" in i, caller._metadata['labels']))[0]
            segmentations_rep1 = caller._segmentations[label_rep1][key]
            segmentations_rep2 = caller._segmentations[label_rep2][key]
            return [segmentations_rep1, segmentations_rep2]

        elif mode == 'sizes':
            segmentation_sizes = [[caller._segmentations[label][key].sizes]
                                  for label in caller._metadata['labels']]
            return segmentation_sizes
        else:
            raise Exception("Mode for data generation not understood: %s" % mode)

    @staticmethod
    def background_calc(segmentation_sizes, background_method='size'):
        """
        Calculates background function: mean size of TADs or
        dispersion of sizes for each segmentation.
        :param segmentation_sizes: nested list of segmentation sizes
        :param background_method: a background function to be calculated
        """
        if background_method == 'size':
            return np.array([[np.mean(i) for i in labelled] for labelled in segmentation_sizes])
        elif background_method == 'dispersion':
            return np.array([[np.std(i) for i in labelled] for labelled in segmentation_sizes])
        else:
            raise Exception("Background method not understood: %s" % background_method)

    @staticmethod
    def optimised_calc(segmentation_list, optimisation, **kwargs):
        """
        Calculates function to be optimised: TPRs and FDRs for
        simulated segmentations, convergence between two replica
        or distances to the closest genome features based
        on track file.
        :param segmentation_list: contains one or two nested lists with segmentations
        :param optimisation: an optimisation mode to be calculated
        :param track: a GenomicRanges instance of some genomic track to be used
        """

        if optimisation == 'simulated':
            track = kwargs.get('track', None)
            return list(map(np.array, ([segmentation.count_coef(track, coef=function)
                                        for segmentation in segmentation_list[0]]
                                       for function in ('TPR TADs', 'PPV TADs', 'TPR boundaries', 'PPV boundaries'))))

        elif optimisation == 'convergence':
            return list(map(np.array, [segmentation_list[0].count_coef(segmentation_list[1], coef=function)
                                       for function in ('JI TADs', 'OC TADs', 'JI boundaries', 'OC boundaries')]))

        elif optimisation == 'border_events':
            distances = [segmentation.dist_closest(track, mode='bin-boundariwise').flatten()
                         for segmentation in segmentation_list[0]]
            return [np.array([-np.mean(np.abs(i)) for i in distances])]

        elif optimisation == 'fitting-average':
            midpoint = kwargs.get('average', 100)
            return [[-np.abs(midpoint - np.mean(segmentation.sizes)) for segmentation in segmentation_list[0]]]
        else:
            raise Exception("optimisation method not understood: %s" % optimisation)

    def call(self, **kwargs):
        """
        Perform one segmentation call, estimate and save
        background and optimisation values.
        """

        background_method = kwargs.get('background_method', self.background_method)

        self.caller.call()

        segmentation_sizes = ExperimentNoGamma.data_generation(self.caller, 'sizes', key=self._key)
        self.results['background'] = ExperimentNoGamma.background_calc(segmentation_sizes, background_method=background_method)
        segmentation_list = ExperimentNoGamma.data_generation(self.caller, self.optimisation, key=self._key)
        self.results['optimisation'] = ExperimentNoGamma.optimised_calc(segmentation_list, self.optimisation, track=self.track, average=self.profile['midpoint'])    