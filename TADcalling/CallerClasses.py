"""
Classes for TAD calling for various tools
"""

from .utils import *
from .logger import logger
from .DataClasses import GenomicRanges, load_BED
from .templates import hicseg_template
from copy import deepcopy
import numpy as np
import pandas as pd
import cooler
import lavaburst
import tadtool.tad

ACCEPTED_FORMATS = ['cool', 'txt', 'txt.gz', 'sparse']


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

        logger.debug("Initializing from files: %s", str(datasets_files))

        assert len(datasets_labels) == len(datasets_files)

        metadata = kwargs.get('metadata', {})
        if not isinstance(metadata, dict):
            raise TypeError("Metadata should be a dictionary not a %s" % str(type(metadata)))

        metadata['assembly'] = kwargs.get('assembly', '-')
        metadata['chr'] = kwargs.get('chr', 'chr1')
        metadata['size'] = kwargs.get('size', 0)
        metadata['resolution'] = metadata.get('resolution', 1000)
        metadata['labels'] = datasets_labels
        metadata['balance'] = kwargs.get('balance', True)

        assert data_format in ACCEPTED_FORMATS
        metadata['data_formats'] = [data_format]
        metadata['files_{}'.format(data_format)] = datasets_files

        self._metadata = metadata

        self._segmentations = {x: {} for x in datasets_labels}

    def tune_matrix(self, input_mtx, remove_diagonal=True, fill_nans=True, undercut=True):
        """
        Tunes 2d matrix according to kwargs. Returns tuned matrix.
        :param input_mtx: input 2d matrix
        :param remove_diagonal: whether to remove diagonal or not (default True)
        :param fill_nans: fill NaNs with 0 if presented (default True)
        :param undercut: perform other tuning operations
        """
        mtx = input_mtx.copy()

        if fill_nans:
            mtx[np.isnan(mtx)] = 0

        if remove_diagonal:
            np.fill_diagonal(mtx, 0)

        if undercut:
            mn = np.percentile(mtx[mtx > 0], 1)
            mx = np.percentile(mtx[mtx > 0], 99)
            mtx[mtx <= mn] = mn
            mtx[mtx >= mx] = mx
            mtx = np.log(mtx)
            mtx = mtx - np.min(mtx)

        return mtx

    def convert_file(self, input_filename=None, input_format=None,
                     mtx=None, output_format='txt', **kwargs):
        """
        Convert one file or matrix into required format.
        """
        tune = kwargs.get('tune', False)
        balance = kwargs.get('balance', True)
        as_pixels = kwargs.get('as_pixels', False)
        ch = kwargs.get('ch', 'chr1')
        res = kwargs.get('res', 20000)

        if input_filename and input_format:
            if input_format == 'cool' and 'txt' in output_format:
                output_prefix = '.'.join(input_filename.split('.')[:-1])
                output_filename = output_prefix + '.{}.txt'.format(ch)

                c = cooler.Cooler(input_filename)
                mtx = c.matrix(balance=balance, as_pixels=as_pixels).fetch(ch, ch)

                if tune:
                    mtx = self.tune_matrix(mtx)

                np.savetxt(output_filename, mtx, delimiter='\t')

                if 'gz' in output_format:
                    subprocess.call('gzip {}'.format(output_filename), shell=True)
                    output_filename += '.gz'

                return output_filename

            elif input_format == 'cool' and 'sparse' in output_format:
                output_prefix = '.'.join(input_filename.split('.')[:-1])
                output_filename = output_prefix + '.{}.sparse.txt'.format(ch)

                c = cooler.Cooler(input_filename)
                mtx = c.matrix(balance=True, as_pixels=as_pixels).fetch(ch, ch)
                mtx.loc[:, "bin1_id":"bin2_id"] += 1

                mtx.loc[:, 'bin1_id':'count'].to_csv(output_filename, header=False, index=False, sep='\t')

                max_bin = mtx.loc[:, 'bin1_id':'bin2_id'].max().max()

                with open(output_prefix + ".{}.genome_bin.txt".format(ch), 'w') as outfile:
                    outfile.write("1\tchr1\t0\t{}".format(max_bin - 1))

                with open(output_prefix + ".{}.all_bins.txt".format(ch), 'w') as outfile:
                    for i in range(max_bin):
                        outfile.write("0\t{}\t{}\n".format(i * res + 1, (i + 1) * res))

                return output_filename

            elif 'txt' in input_format and output_format == 'cool':
                # TODO implement this option
                # https://github.com/hms-dbmi/higlass/issues/100#issuecomment-302183312
                logger.error('Option currently not importmented!')

        elif mtx:
            if tune:
                mtx = self.tune_matrix(mtx)

                if 'txt' in output_format:
                    np.savetxt(output_filename, mtx, delimiter='\t')

                    if 'gz' in output_format:
                        subprocess.call('gzip {}'.format(output_filename), shell=True)
                        output_filename += '.gz'

                elif output_format == 'cool':
                    logger.error('Option currently not importmented!')

            return output_filename

        else:
            raise Exception("Neither input filename nor matrix are presented.")

    def convert_files(self, data_format, **kwargs):
        """
        Converts input files into required format.
        :param data_format: format of resulting files ('cool', 'txt', 'txt.gz')
        :param kwargs: Optional parameters for data coversion
        :return: None
        """
        tune = kwargs.get('tune', False)
        as_pixels = kwargs.get('as_pixels', False)

        assert data_format in ACCEPTED_FORMATS

        original_format = self._metadata['data_formats'][0]

        resulting_files = []
        file_holder = 'files_{}'.format(original_format)
        param_dict = {'tune': tune, 'balance': self._metadata['balance'],
                      'as_pixels': as_pixels, 'ch': self._metadata['chr'],
                      'res': self._metadata['resolution']}
        for f in self._metadata[file_holder]:
            output_fname = self.convert_file(input_filename=f, input_format=original_format,
                                             output_format=data_format, **param_dict)
            resulting_files.append(output_fname)

        self._metadata['files_{}'.format(data_format)] = resulting_files
        self._metadata['data_formats'].append(data_format)

    def _load_segmentations(self, input, params):
        """
        Basic function of BaseCaller to load the segmentation (from file or from variable)
        :param input dict (ordered by metadata['labels']) with segmentations (2d np.ndarray) or names of files
        :param params: set of parameters
        :return:
        """
        logger.debug("Calling dump BaseCaller segmentation loading")

        for x in self._metadata['labels']:
            self._segmentations[x][params] = input[x]

    def call(self, params):
        """
        Basic function of BaseCaller to call the segmentation (and write it to file or to RAM)
        :param params: set of calling parameters
        :return: dict (ordered by metadata['labels']) with segmentations (2d np.ndarray) or names of files
        """
        logger.debug("Calling dump BaseCaller segmentation call with params: {}".format(str, params))
        segmentations = {x: GenomicRanges(np.empty([0, 0], dtype=int), data_type="segmentation") for x in self._metadata['labels']}
        self._load_segmentations(segmentations, params)
        return  # what?

    def segmentation2df(self):
        """
        Converter of all segmentations to Pandas dataframe.
        :return:
        """
        self._df = pd.DataFrame(columns=['bgn', 'end', 'label', 'params'])
        for x in self._segmentations.keys():
            for y in self._segmentations[x].keys():
                lngth = self._segmentations[x][y].data.shape[0]
                df = pd.DataFrame({
                    'bgn': self._segmentations[x][y].data[:, 0],
                    'end': self._segmentations[x][y].data[:, 1],
                    'label': [x for i in range(lngth)],
                    'params': [str(y) for i in range(lngth)]
                })
                self._df = pd.concat([self._df, df]).copy()
        self._df.bgn = pd.to_numeric(self._df.bgn)
        self._df.end = pd.to_numeric(self._df.end)
        return self._df


class LavaburstCaller(BaseCaller):

    # Example run:
    # from CallerClasses import *
    # lc = LavaburstCaller(['S2'], ['../data/S2.20000.cool'], 'cool', assembly='dm3', resolution=20000, balance=True, chr='chr2L')
    # lc.call(0.9)
    # lc.call(1.9)
    # df = lc.segmentation2df()
    #

    def call(self, gamma, tune=True, method='armatus', **kwargs):

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        output_dct = {}

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'], self._metadata['chr'])

            if tune:
                mtx = self.tune_matrix(mtx)

            segmentation = self._call_single(mtx, gamma, method=method, **kwargs)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, (gamma))

        return self._segmentations

    def _call_single(self, mtx, gamma, good_bins='default', method='armatus', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param method: 'modularity', 'variance', 'corner' or 'armatus' (defalut)
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """

        if np.any(np.isnan(mtx)):
            logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            logger.warning(
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


class ArmatusCaller(BaseCaller):

    def call(self, gamma, tune=True, **kwargs):

        output_dct = {}

        if 'files_txt.gz' not in self._metadata.keys():
            self.convert_files('txt.gz', tune=tune)

        for label, f in zip(self._metadata['labels'], self._metadata['files_txt.gz']):
            segmentation = self._call_single(f, gamma, **kwargs)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, (gamma))

        return self._segmentations

    def _call_single(self, mtx_name, gamma, good_bins='default', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """

        subprocess.run("armatus -i {} -g {} -j -o buff -r 1".format(mtx_name, gamma), shell=True)
        segments = np.loadtxt("buff.consensus.txt", ndmin=2, dtype=object)
        segments = np.array(segments[:, 1:], dtype=int)
        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class InsulationCaller(BaseCaller):

    def call(self, window, cutoff, tune=True, **kwargs):

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        output_dct = {}

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'], self._metadata['chr'])

            if tune:
                mtx = self.tune_matrix(mtx)

            segmentation = self._call_single(mtx, window, cutoff, **kwargs)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, (window, cutoff)) # TODO: make it available to deploy two parameters

        return self._segmentations

    def _call_single(self, mtx, window, cutoff, good_bins='default', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param method: 'modularity', 'variance', 'corner' or 'armatus' (defalut)
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """

        if np.any(np.isnan(mtx)):
            logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            logger.warning(
                "Note that diagonal is not removed. you might want to delete it to avoid noisy and not stable results. ")

        regions = [tadtool.tad.GenomicRegion(chromosome='', start=i, end=i) for i in range(mtx.shape[0])]

        ii = tadtool.tad.insulation_index(mtx, regions, window_size=window)
        tads = tadtool.tad.call_tads_insulation_index(ii, cutoff, regions=regions)
        segments = np.array([[some_tad.start - 1, some_tad.end] for some_tad in tads], dtype=int)

        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class DirectionalityCaller(BaseCaller):

    def call(self, window, cutoff, tune=True, **kwargs):

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        output_dct = {}

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'], self._metadata['chr'])

            if tune:
                mtx = self.tune_matrix(mtx)

            segmentation = self._call_single(mtx, window, cutoff, **kwargs)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, (window, cutoff)) # TODO: make it available to deploy two parameters

        return self._segmentations

    def _call_single(self, mtx, window, cutoff, good_bins='default', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param method: 'modularity', 'variance', 'corner' or 'armatus' (defalut)
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """

        if np.any(np.isnan(mtx)):
            logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            logger.warning(
                "Note that diagonal is not removed. you might want to delete it to avoid noisy and not stable results. ")

        regions = [tadtool.tad.GenomicRegion(chromosome='', start=i, end=i) for i in range(mtx.shape[0])]

        ii = tadtool.tad.directionality_index(mtx, regions, window_size=window)
        tads = tadtool.tad.call_tads_directionality_index(ii, cutoff, regions=regions)
        segments = np.array([[some_tad.start - 1, some_tad.end] for some_tad in tads], dtype=int)

        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class HiCsegCaller(BaseCaller):

    def call(self, tune=True, distr_model="P", **kwargs):

        output_dct = {}

        if 'files_txt' not in self._metadata.keys():
            self.convert_files('txt', tune=tune)

        for label, f in zip(self._metadata['labels'], self._metadata['files_txt']):
            segmentation = self._call_single(f, distr_model, **kwargs)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, (distr_model))

        return self._segmentations

    def _call_single(self, mtx_name, distr_model, good_bins='default', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """
        hicseg_script = hicseg_template.format(mtx_name, distr_model, 1, 'output_hicseg.txt')
        with open('hicseg_script.R', "w") as hicseg_script_file:
            hicseg_script_file.write(hicseg_script)
        subprocess.run("Rscript hicseg_script.R", shell=True)
        segments = np.loadtxt("output_hicseg.txt", ndmin=2, dtype=int)
        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')


class MrTADFinderCaller(BaseCaller):

    def call(self, resolution, tune=False, **kwargs):

        output_dct = {}

        if 'files_sparse' not in self._metadata.keys():
            self.convert_files('sparse', tune=tune, as_pixels=True)

        for label, f in zip(self._metadata['labels'], self._metadata['files_sparse']):
            segmentation = self._call_single(f, resolution, **kwargs)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, (gamma))

        return self._segmentations

    def _call_single(self, mtx_name, res, good_bins='default', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """
        # check usage in coomand below
        # clone MrTADFinder from my fork at https://github.com/dmitrymyl/MrTADFinder.git
        mtx_prefix = '.'.join(mtx_name.split('.')[:-2])
        subprocess.run("julia ../MrTADFinder/run_MrTADFinder.jl {} {}.genome_bin.txt {}.all_bins.txt res={} 1 buff_mrtadfinder.txt".format(mtx_name, mtx_prefix, mtx_prefix, res), shell=True)
        mr_df = pd.read_csv('buff_mrtadfinder.txt')
        segments = np.array(mr_df.loc[:, "domain_st_bin":"domain_ed_bin"].values, ndmin=2, dtype=int)
        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')
