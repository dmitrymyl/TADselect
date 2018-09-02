"""
Classes for TAD calling for various tools
"""

#TODO: remove min_tad_size from everywhere to GRanges and Experiment

# Universal imports
from .utils import *
from .logger import TADcalling_logger # as logger
from .DataClasses import GenomicRanges, load_BED
from .InteractionMatrix import InteractionMatrix
from copy import deepcopy
from functools import partial
import numpy as np
import pandas as pd
import cooler
import glob
import shutil

# Class-specific imports
import tadtool.tad
import lavaburst
from .templates import hicseg_template

ACCEPTED_FORMATS = ['cool', 'txt', 'txt.gz', 'sparse', 'mr_sparse', 'hic', 'h5']


class BasicCallerException(Exception):
    pass


class BaseCaller(object):

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        """
        Initialize base TAD caller

        :param datasets_labels (:obj:`list`): short names of files that are needed to be processed
        :param datasets_files (:obj:`list`): names of files with intra-chromosomal chromatin interaction maps
        :param data_format (:obj:`str`): format of input files ('cool', 'txt', 'txt.gz')
        :param kwargs:
            * assembly (:obj:`str`): name of assembly, default is '-'
            * chromosome (:obj:`str`): name of chromosome, default is 'chr1'
            * size (:obj:`int`): size of dataset in bins, default is 0
            * resolution (:obj:`int`): size of bin in basepairs, default is 1000
            * balance (:obj:`bool`): obtain balanced map from cool file or not (default True)
            * metadata (:obj:`dict`): optional metadata dict, default is empty dict

            Object of BaseCaller type or derivative has two attributes:
            self._metadata -- metadata, dict with 'assembly', 'chr','resolution', 'labels', 'balance' keys
            self._segmentations -- dictionary with segmentations for all the files
        """

        TADcalling_logger.debug("Initializing from files: %s", str(datasets_files))

        assert len(datasets_labels) == len(datasets_files)

        metadata = kwargs.get('metadata', {})
        if not isinstance(metadata, dict):
            raise TypeError("Metadata should be a dictionary not a %s" % str(type(metadata)))

        metadata['assembly'] = kwargs.get('assembly', '-')
        metadata['chr'] = kwargs.get('chr', 'chr1')
        metadata['size'] = kwargs.get('size', 0)
        metadata['resolution'] = kwargs.get('resolution', 1000)
        metadata['labels'] = datasets_labels
        metadata['balance'] = kwargs.get('balance', True)
        metadata['params'] = ['params']
        metadata['method'] = 'Base'

        assert data_format in ACCEPTED_FORMATS
        metadata['data_formats'] = [data_format]

        for f in datasets_files:
            assert os.path.isfile(f)

        metadata['files_{}'.format(data_format)] = datasets_files

        self._metadata = metadata

        self._segmentations = {x: {} for x in datasets_labels}

    def convert_files(self, data_format, **kwargs):
        """
        TODO @agal Remove all the levels of data processing to the InteractionMatrix class
        TODO @agal Remove everything to old "tune" function

        Converts input files into required format.
        :param data_format: format of resulting files ('cool', 'txt', 'txt.gz', 'hic', 'h5')
        :param kwargs: Optional parameters for data coversion
        :return: None
        """

        assert data_format in ACCEPTED_FORMATS

        original_format = kwargs.get('original_format', self._metadata['data_formats'][0])

        resulting_files = []
        file_holder = 'files_{}'.format(original_format)
        param_dict = {
            'balance': self._metadata['balance'],
            'ch': self._metadata['chr'],
            'res': self._metadata['resolution']
        }
        for k in ['remove_intermediary_files', 'juicer_path']:
            if k in kwargs.keys():
                param_dict[k] = kwargs.get(k)

        for f in self._metadata[file_holder]:

            mtxObj = InteractionMatrix(f, original_format, read=False)

            output_fname = mtxObj.convert_without_reading(input_filename=f,
                                                          output_filename="{}.{}".format(f, data_format),
                                                          input_format=original_format,
                                                          output_format=data_format, **param_dict)
            resulting_files.append(output_fname)

        self._metadata['files_{}'.format(data_format)] = resulting_files
        self._metadata['data_formats'].append(data_format)

    def _load_segmentations(self, input, params, label=None):
        """
        Basic function of BaseCaller to load the segmentation (from file or from variable)
        :param input dict (ordered by metadata['labels']) with segmentations (2d np.ndarray) or names of files
        :param params: set of parameters
        :return:
        """
        if label:
            self._segmentations[label][params] = input[label]
        else:
            for x in self._metadata['labels']:
                self._segmentations[x][params] = input[x]

    def call(self, params):
        """
        Basic function of BaseCaller to call the segmentation (and write it to file or to RAM)
        :param params: set of calling parameters
        :return: dict (ordered by metadata['labels']) with segmentations (2d np.ndarray) or names of files
        """
        TADcalling_logger.debug("Calling %s with params: %s" % (self.__class__.__name__, str(params)))
        segmentations = {x: GenomicRanges(np.empty([0, 0], dtype=int), data_type="segmentation")
                         for x in self._metadata['labels']}
        self._load_segmentations(segmentations, params)

    def segmentation2df(self):
        """
        Converter of all the segmentations to Pandas dataframe.
        :return:
        """
        params_names = self._metadata['params']
        self._df = pd.DataFrame(columns=['bgn', 'end', 'length', 'label', 'caller'] + params_names)
        for x in self._segmentations.keys():
            for y in self._segmentations[x].keys():
                length = self._segmentations[x][y].length
                if length == 0 or self._segmentations[x][y].data.shape[1] == 0:
                    continue  # it would be better to store zero segmentations in order to compare
                dct = {
                    'bgn': self._segmentations[x][y].data[:, 0],
                    'end': self._segmentations[x][y].data[:, 1],
                    'length': (self._segmentations[x][y].data[:, 1]-self._segmentations[x][y].data[:, 0]),
                    'label': [x for i in range(length)],
                    'caller': [self._metadata['caller'] for i in range(length)]
                }
                if isinstance(y, int) or isinstance(y, float) or isinstance(y, np.float_) or isinstance(y, np.int_):
                    y = [y]
                for param, value in zip(params_names, y):
                    dct.update({param: value})

                df = pd.DataFrame(dct)

                self._df = pd.concat([self._df, df]).copy()

        self._df.bgn = pd.to_numeric(self._df.bgn)
        self._df.end = pd.to_numeric(self._df.end)
        self._df.length = pd.to_numeric(self._df.length)

        for param in params_names:
            try:
                self._df[param] = pd.to_numeric(self._df[param])
            except Exception as e:
                pass

        return self._df


class LavaburstCaller(BaseCaller):

    # Example run:
    # from CallerClasses import *
    # lc = LavaburstCaller(['S2'], ['../data/S2.20000.cool'], 'cool',
    # assembly='dm3', resolution=20000, balance=True, chr='chr2L')
    # lc.call({'gamma':[0.1,0.9]})
    # df = lc.segmentation2df()
    #

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(LavaburstCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['gamma', 'method']
        self._metadata['caller'] = 'Lavaburst'
        self._metadata['method'] = kwargs.get('method', 'armatus')

    def call(self, params_data={}, **kwargs):
        """
        Lavaburst segmentation calling for a set of parameters.
        :param params_dict: dictionary of parameters, containing gammas and methods
        :param kwargs:
        :return:
        """

        TADcalling_logger.debug("Calling %s with params: %s" % (self.__class__.__name__, str(params_data)))

        params_dict = dict()
        params_dict['gamma'] = params_data.get('gamma', np.arange(0, 10, 1))
        params_dict['method'] = params_data.get('method', [self._metadata['method']])

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            output_dct = {}

            mtx_orig = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'],
                                                                                     self._metadata['chr'])
            mtxObj = InteractionMatrix(mtx_orig)
            mtx = mtxObj.fill_nans(0).remove_diagonal(1, 0).filter_extreme().log_transform(2).subtract_min().as_array()

            for gamma in params_dict['gamma']:
                for method in params_dict['method']:
                    segmentation = self._call_single(mtx, gamma, method=method, **kwargs)
                    output_dct = {label: deepcopy(segmentation)}
                    if len(params_dict['method']) > 1:
                        self._load_segmentations(output_dct, (gamma, method), label=label)
                    else:
                        self._load_segmentations(output_dct, (gamma), label=label)

    def _call_single(self, mtx, gamma, good_bins='default',
                     method='armatus', max_intertad_size=3, max_tad_size=10000, **kwargs):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param method: 'modularity', 'variance', 'corner' or 'armatus' (defalut)
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """

        # TODO @agal move this step to one level up
        if np.any(np.isnan(mtx)):
            TADcalling_logger.warning("NaNs in dataset, pease remove them first.")

        # TODO @agal move this step to one level up
        if np.diagonal(mtx).sum() > 0:
            TADcalling_logger.warning(
                "Note that diagonal is not removed. you might want to delete it to avoid noisy and not stable results. ")

        if method == 'modularity':
            score = lavaburst.scoring.modularity_score
        elif method == 'armatus':
            score = lavaburst.scoring.armatus_score
        elif method == 'corner':
            score = lavaburst.scoring.corner_score
        elif method == 'variance':
            score = lavaburst.scoring.variance_score
        else:
            raise BasicCallerException('Algorithm not understood: {}'.format(method))

        if good_bins == 'default':
            good_bins = mtx.astype(bool).sum(axis=0) > 0

        S = score(mtx, gamma=gamma, binmask=good_bins)
        model = lavaburst.model.SegModel(S)

        segments = model.optimal_segmentation()

        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


LavaArmatusCaller = partial(LavaburstCaller, method='armatus')
LavaModularityCaller = partial(LavaburstCaller, method='modularity')
LavaVarianceCaller = partial(LavaburstCaller, method='variance')
LavaCornerCaller = partial(LavaburstCaller, method='corner')


class ArmatusCaller(BaseCaller):

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(ArmatusCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['gamma']
        self._metadata['caller'] = 'Armatus'

    def call(self, params_dict={}, **kwargs):

        TADcalling_logger.debug("Calling %s with params: %s" % (self.__class__.__name__, str(params_dict)))

        params_dict['gamma'] = params_dict.get('gamma', np.arange(0, 10, 1))

        if 'files_txt.gz' not in self._metadata.keys():
            self.convert_files('txt.gz')

        for gamma in params_dict['gamma']:
            output_dct = {}
            for label, f in zip(self._metadata['labels'], self._metadata['files_txt.gz']):
                segmentation = self._call_single(f, gamma, **kwargs)
                output_dct[label] = deepcopy(segmentation)
            self._load_segmentations(output_dct, (gamma))

    def _call_single(self, mtx_name, gamma, good_bins='default',
                     max_intertad_size=3, max_tad_size=10000, caller_path='armatus', **kwargs):

        command = "{} -i {} -g {} -j -o buff -r 1".format(caller_path, mtx_name, gamma)
        run_command(command)
        segments = np.loadtxt("buff.consensus.txt", ndmin=2, dtype=object)
        segments = np.array(segments[:, 1:], dtype=int)
        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class InsulationCaller(BaseCaller):

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(InsulationCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['window', 'cutoff']
        self._metadata['caller'] = 'Insulation'

    def call(self, params_dict={}, **kwargs):

        TADcalling_logger.debug("Calling %s with params: %s" % (self.__class__.__name__, str(params_dict)))

        params_dict['window'] = params_dict.get('window', [1])
        params_dict['cutoff'] = params_dict.get('cutoff', [0])

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False)\
                .fetch(self._metadata['chr'], self._metadata['chr'])

            for window in params_dict['window']:
                for cutoff in params_dict['cutoff']:
                    segmentation = self._call_single(mtx, window, cutoff, **kwargs)
                    output_dct = {label: deepcopy(segmentation)}

                    self._load_segmentations(output_dct, (window, cutoff), label=label)

    def _call_single(self, mtx, window, cutoff, max_intertad_size=3, max_tad_size=10000, **kwargs):
        """
        TODO: annotate, is size in bp or in genomic bins?
        TODO add unified approach, where we use bp or bins for all algorithms!
        BP: will reflect biology, but can get confused when fitting parameters.
        Bins: do not reflect biology but only properties of given matrix.
        I'd rather use bp.
        :param mtx: ndarray Hi-C matrix
        :param window: window size in bp
        :param cutoff:
        :param good_bins:
        :param max_intertad_size:
        :param max_tad_size:
        :return:
        """

        if np.any(np.isnan(mtx)):
            TADcalling_logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            TADcalling_logger.warning(
                "Note that diagonal is not removed. You might want to delete it to avoid noisy and not stable results.")

        regions = [tadtool.tad.GenomicRegion(chromosome='', start=i, end=i) for i in range(mtx.shape[0])]

        IS = tadtool.tad.insulation_index(mtx, regions, window_size=window / self._metadata['resolution'])
        tads = tadtool.tad.call_tads_insulation_index(IS, cutoff, regions=regions)
        segments = np.array([[some_tad.start - 1, some_tad.end] for some_tad in tads], dtype=int)

        if len(segments) > 0:
            v = segments[:, 1] - segments[:, 0]
            mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)
            segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class DirectionalityCaller(BaseCaller):

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(DirectionalityCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['window', 'cutoff']
        self._metadata['caller'] = 'Directionality'

    def call(self, params_dict={}, **kwargs):

        TADcalling_logger.debug("Calling %s with params: %s" % (self.__class__.__name__, str(params_dict)))

        params_dict['window'] = params_dict.get('window', [1])
        params_dict['cutoff'] = params_dict.get('cutoff', [0])

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False)\
                .fetch(self._metadata['chr'], self._metadata['chr'])

            for window in params_dict['window']:
                for cutoff in params_dict['cutoff']:
                    segmentation = self._call_single(mtx, window, cutoff, **kwargs)
                    output_dct = {label: deepcopy(segmentation)}

                    self._load_segmentations(output_dct, (window, cutoff), label=label)

    def _call_single(self, mtx, window, cutoff, max_intertad_size=3, max_tad_size=10000, **kwargs):

        if np.any(np.isnan(mtx)):
            TADcalling_logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            TADcalling_logger.warning(
                "Note that diagonal is not removed. You might want to delete it to avoid noisy and not stable results.")

        regions = [tadtool.tad.GenomicRegion(chromosome='', start=i, end=i) for i in range(mtx.shape[0])]

        ii = tadtool.tad.directionality_index(mtx, regions, window_size=window / self._metadata['resolution'])
        tads = tadtool.tad.call_tads_directionality_index(ii, cutoff, regions=regions)
        segments = np.array([[some_tad.start - 1, some_tad.end] for some_tad in tads], dtype=int)

        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class HiCsegCaller(BaseCaller):
    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(HiCsegCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['distr_model']
        self._metadata['caller'] = 'HiCseg'

    def call(self, params_dict={}, **kwargs):

        TADcalling_logger.debug("Calling %s with params: %s" % (self.__class__.__name__, str(params_dict)))

        params_dict['distr_model'] = params_dict.get('distr_model', ["P"])

        if 'files_txt' not in self._metadata.keys():
            self.convert_files('txt')

        for label, f in zip(self._metadata['labels'], self._metadata['files_txt']):
            for distr_model in params_dict['distr_model']:
                segmentation = self._call_single(f, distr_model, **kwargs)
                output_dct = {label: deepcopy(segmentation)}

                self._load_segmentations(output_dct, (distr_model), label=label)

    def _call_single(self, mtx_name, distr_model,
                     max_intertad_size=3, max_tad_size=10000, binary_path='Rscript'):

        hicseg_script = hicseg_template.format(mtx_name, distr_model, 1, 'output_hicseg.txt')
        with open('hicseg_script.R', "w") as hicseg_script_file:
            hicseg_script_file.write(hicseg_script)

        command = "{} hicseg_script.R".format(binary_path)
        run_command(command)
        segments = np.loadtxt("output_hicseg.txt", ndmin=2, dtype=int)
        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class MrTADFinderCaller(BaseCaller):

    def call(self, **kwargs):

        TADcalling_logger.debug("Calling %s" % (self.__class__.__name__))

        caller_path = kwargs.get('caller_path', '../MrTADFinder/run_MrTADFinder.jl')
        output_dct = {}

        if 'files_mr_sparse' not in self._metadata.keys():
            self.convert_files('mr_sparse')

        for label, f in zip(self._metadata['labels'], self._metadata['files_mr_sparse']):
            segmentation = self._call_single(f, caller_path=caller_path)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, gamma) # TODO: fix, no gamma defined for MrTADFinder

    def _call_single(self, mtx_name, max_intertad_size=3, max_tad_size=10000,
                     binary_path='julia', **kwargs):
        # check usage in command below
        # clone MrTADFinder from my fork at https://github.com/dmitrymyl/MrTADFinder.git
        caller_path = kwargs.get('caller_path', '../MrTADFinder/run_MrTADFinder.jl')
        res = self._metadata['resolution']

        mtx_prefix = '.'.join(mtx_name.split('.')[:-2])
        command = "{} {} {} {}.genome_bin.txt {}.all_bins.txt res={} 1 buff_mrtadfinder.txt"\
            .format(binary_path, caller_path, mtx_name, mtx_prefix, mtx_prefix, res)
        run_command(command)
        mr_df = pd.read_csv('buff_mrtadfinder.txt')
        segments = np.array(mr_df.loc[:, "domain_st_bin":"domain_ed_bin"].values, ndmin=2, dtype=int)
        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class ArrowheadCaller(BaseCaller):


    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(ArrowheadCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['windowSize']
        self._metadata['caller'] = 'Arrowhead'

    def call(self, params_dict={}, **kwargs):

        if 'files_hic' not in self._metadata.keys():
            raise BasicCallerException("No hic file present for caller. Please, perform valid conversion!")

        params_dict['windowSize'] = params_dict.get('windowSize', [2000])

        remove_intermediates = kwargs.get('remove_intermediates', False)

        new_folder = False
        if not os.path.isdir("tmp"):
            os.mkdir('tmp')
            new_folder = True

        for windowSize in params_dict['windowSize']:
            output_dct = {}
            for label, f in zip(self._metadata['labels'], self._metadata['files_hic']):

                outmask = "tmp/{}.arrowhead.{}.tmp".format(label, windowSize)
                segmentation = self._call_single(f,
                                             outmask,
                                             windowSize,
                                             **kwargs)

                output_dct[label] = deepcopy(segmentation)

                if remove_intermediates:
                    shutil.rmtree(outmask)

            self._load_segmentations(output_dct,
                                     (windowSize))

        if new_folder and remove_intermediates:
            shutil.rmtree("tmp")
        return self._segmentations

    def _call_single(self, infile, outmask,
                     windowSize=2000,
                     caller_path='./juicer_tools.1.8.9_jcuda.0.8.jar',
                     java_path='java',
                     **kwargs):

        command = """{java} -Xmx2g -jar {caller_path} \
         arrowhead -m {windowSize} \
         -c {ch} \
         -r {resolution} \
         -k NONE \
         --ignore_sparsity \
         {infile_hic} \
         {output_directory}""".format(caller_path=caller_path,
                                      java=java_path,
                                      windowSize=windowSize,
                                      ch=self._metadata['chr'],
                                      resolution=self._metadata['resolution'],
                                      infile_hic=infile,
                                      output_directory=outmask)

        run_command(command)
        try:
            output_file = glob.glob(outmask + "/*")[0]
            segments = \
                pd.read_csv(output_file, sep='\t').sort_values('x1')[['x1', 'x2']].values/self._metadata['resolution']
        except Exception as e:
            segments = []

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class HiCExplorerCaller(BaseCaller):

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(HiCExplorerCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['minDepth', 'maxDepth', 'step', 'thresholdComparisons', 'delta', 'correction']
        self._metadata['caller'] = 'HiCExplorer'

    def call(self, params_dict={}, **kwargs):

        if 'files_h5' not in self._metadata.keys():
            raise BasicCallerException("No h5 file present for caller. Please, perform valid conversion!")

        params_dict['minDepth'] = params_dict.get('minDepth', [3*self._metadata['resolution']]) # At least 3 resolutions
        params_dict['maxDepth'] = params_dict.get('maxDepth', [5*self._metadata['resolution']]) # At least 5 resolutions
        params_dict['step'] = params_dict.get('step', [ self._metadata['resolution'] ]) # At least resolution
        params_dict['thresholdComparisons'] = params_dict.get('thresholdComparisons', [0.05])
        params_dict['delta'] = params_dict.get('delta', [0.01])
        params_dict['correction'] = params_dict.get('correction', ['fdr'])

        remove_intermediates = kwargs.get('remove_intermediates', False)

        for minDepth in params_dict['minDepth']:
            for maxDepth in params_dict['maxDepth']:
                for step in params_dict['step']:
                    for thresholdComparisons in params_dict['thresholdComparisons']:
                        for delta in params_dict['delta']:
                            for correction in params_dict['correction']:
                                output_dct = {}
                                for label, f in zip(self._metadata['labels'], self._metadata['files_h5']):
                                    outmask = \
                                        "tmp/{label}.{minDepth}.{maxDepth}.{step}.{tC}.{delta}.{correction}.hicexplorer.tmp".format(label=label,
                                                                                                                                    minDepth=minDepth,
                                                                                                                                    maxDepth=maxDepth,
                                                                                                                                    step=step,
                                                                                                                                    tC=thresholdComparisons,
                                                                                                                                    delta=delta,
                                                                                                                                    correction=correction)
                                    segmentation = self._call_single(f,
                                                                     outmask,
                                                                     minDepth,
                                                                     maxDepth,
                                                                     step,
                                                                     thresholdComparisons,
                                                                     delta,
                                                                     correction,
                                                                     **kwargs)
                                    output_dct[label] = deepcopy(segmentation)
                                self._load_segmentations(output_dct,
                                                         (minDepth, maxDepth, step,
                                                          thresholdComparisons, delta, correction))
                                if remove_intermediates:
                                    toremove=glob.glob(outmask+'*')
                                    for f in toremove: os.remove(f)


        return self._segmentations

    def _call_single(self, infile, outmask, minDepth, maxDepth, step,
                     thresholdComparisons=0.05,
                     delta=0.01,
                     correction="fdr",
                     nthreads=1,
                     caller_path='hicFindTADs',
                     **kwargs):

        command = """{caller_path} -m {infile_h5} \
                --outPrefix {out_prefix} \
                --minDepth {minDepth} \
                --maxDepth {maxDepth} \
                --step {step} \
                --thresholdComparisons {th} \
                --delta {delta} \
                --correctForMultipleTesting {correction} \
                -p {nth}""".format(caller_path=caller_path,
                                infile_h5=infile,
                                out_prefix=outmask,
                                minDepth=minDepth,
                                maxDepth=maxDepth,
                                step=step,
                                th=thresholdComparisons,
                                delta=delta,
                                correction=correction,
                                nth=nthreads)

        run_command(command)

        try:
            output_file = glob.glob(outmask+"*_domains.bed")[0]
            segments = pd.read_csv(output_file, sep='\t', header=None)[[1, 2]].values/self._metadata['resolution']

        except Exception as e:
            segments = []

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class TADtreeCaller(BaseCaller):

#TODO @agal: Reading resulting tree files

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(TADtreeCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['max_TAD_size', 'max_tree_depth', 'boundary_index_p', 'boundary_index_q', 'gamma']
        self._metadata['caller'] = 'TADtree'

    def call(self, params_dict={}, **kwargs):

        if 'files_txt' not in self._metadata.keys():
            raise BasicCallerException("No txt file present for caller. Please, perform valid conversion!")

        params_dict['max_TAD_size'] = params_dict.get("max_TAD_size", [50])    #S = 50  # max. size of TAD (in bins)
        params_dict['max_tree_depth'] = params_dict.get("max_tree_depth", [25])    #M = 25  # max. number of TADs in each tad-tree
        params_dict['boundary_index_p'] = params_dict.get("boundary_index_p", [3])     #p = 3  # boundary index parameter
        params_dict['boundary_index_q'] = params_dict.get("boundary_index_q", [12])    #q = 12 # boundary index parameter
        params_dict['gamma'] = params_dict.get("gamma", [500])                 #gamma = 500  # balance between boundary index and squared error in score function


        for max_TAD_size in params_dict['max_TAD_size']:
            for max_tree_depth in params_dict['max_tree_depth']:
                for boundary_index_p in params_dict['boundary_index_p']:
                    for boundary_index_q in params_dict['boundary_index_q']:
                        for gamma in params_dict['gamma']:

                            output_dct = {}

                            for label, f in zip(self._metadata['labels'], self._metadata['files_txt']):

                                outmask = \
                                    "tmp/{label}.{S}.{M}.{bp}.{bq}.{gamma}.TADtree.tmp".format(
                                        label=label,
                                        S=max_TAD_size,
                                        M=max_tree_depth,
                                        bp=boundary_index_p,
                                        bq=boundary_index_q,
                                        gamma=gamma)

                                towrite = \
                                    "S = {}\nM = {}\np = {}\nq = {}\ngamma = {}\n\ncontact_map_path = {}\ncontact_map_name = {}\nN = {}\noutput_directory = {}".\
                                        format(max_TAD_size, max_tree_depth, boundary_index_p, boundary_index_q, gamma,
                                               f, label, 400, outmask)

                                with open(outmask+".control_file", 'w') as control_file:
                                    control_file.write(towrite)

                                segmentation = self._call_single(outmask+".control_file", **kwargs)
                                output_dct[label] = deepcopy(segmentation)

                            self._load_segmentations(output_dct, (max_TAD_size,
                                                                  max_tree_depth,
                                                                  boundary_index_p,
                                                                  boundary_index_q,
                                                                  gamma))

        return self._segmentations


    def _call_single(self, control_file,
                     caller_path="../bins/TADtree/TADtree.py",
                     python_path='python2',
                     **kwargs):

        command = "{python} {caller_path} {control_file}".format(python=python_path,
            caller_path=caller_path,
            control_file=control_file)

        run_command(command)

        # TODO: @agal add parser pf tree files.
        # Note that code works forever for some reason. Maybe we don't want to use this algo.

        return None


class TADbitCaller(BaseCaller):

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(TADbitCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = []
        self._metadata['caller'] = 'TADbut'

    def call(self, params_dict={}, nth=1, **kwargs):

        if 'files_txt' not in self._metadata.keys():
            raise BasicCallerException("No txt file present for caller. Please, perform valid conversion!")

        output_dct = {}

        for label, f in zip(self._metadata['labels'], self._metadata['files_txt']):

            outmask = "tmp/{label}.TADbit.tmp".format(label=label)

            segmentation = self._call_single(f, outmask, label=label, nth=nth, **kwargs)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, ())

        return self._segmentations

    def _call_single(self, infile, outfile,
                     label='tmp',
                     nth=1,
                     caller_path="../TADcalling/script_TADbit.py",
                     python_path='~/anaconda3/envs/tadbit/bin/python',
                     **kwargs):

#~/anaconda3/envs/tadbit/bin/python ../TADcalling/script_TADbit.py ../data/test_S2.20000.chr2L.txt tmp/tadbit_output.txt S2 chr2L 20000 8

        command = "{python} {caller_path} {infile} {outfile} {exp} {ch} {resolution} {nth} &>/dev/null".format(
            python=python_path,
            caller_path=caller_path,
            infile=infile,
            outfile=outfile,
            exp=label,
            ch=self._metadata['chr'],
            resolution=self._metadata['resolution'],
            nth=nth
        )

        run_command(command)

        segments = pd.read_csv('tmp/example.output.txt', sep=' *')[['start', 'end']].values

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')
