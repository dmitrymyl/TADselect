"""
Classes for TAD calling for various tools
"""

# Universal imports
from .utils import *
from .logger import logger
from .DataClasses import GenomicRanges, load_BED
from copy import deepcopy
import numpy as np
import pandas as pd
import cooler

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

        logger.debug("Initializing from files: %s", str(datasets_files))

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

        logger.info("Converting file: {} from {} to {}".format(input_filename, input_format, output_format))

        tune = kwargs.get('tune', False)
        balance = kwargs.get('balance', True)
        #as_pixels = kwargs.get('as_pixels', False)
        chromosome = kwargs.get('chr', self._metadata['chr'])
        resolution = kwargs.get('res', 20000)
        remove_intermediary_files = kwargs.get('remove_intermediary_files', True)
        output_filename = kwargs.get('output_filename', 'tmp.txt')

        if input_filename and input_format:
            if input_format == 'cool' and 'txt' in output_format:
                output_prefix = '.'.join(input_filename.split('.')[:-1])
                output_filename = output_prefix + '.{}.txt'.format(chromosome)

                c = cooler.Cooler(input_filename)
                mtx = c.matrix(balance=balance, as_pixels=False).fetch(chromosome, chromosome)

                if tune:
                    mtx = self.tune_matrix(mtx)

                np.savetxt(output_filename, mtx, delimiter='\t')

                if 'gz' in output_format:
                    command = 'gzip {}'.format(output_filename)
                    run_command(command)
                    output_filename += '.gz'

                return output_filename

            elif input_format == 'cool' and output_format == 'sparse':

                # Very bad section, needs to be fixed!

                output_prefix = '.'.join(input_filename.split('.')[:-1])
                output_filename = output_prefix + '.{}.sparse.txt'.format(chromosome)

                c = cooler.Cooler(input_filename)
                mtx_df = c.matrix(balance=self._metadata['balance'], as_pixels=True, join=True,
                                  ignore_index=False).fetch(chromosome, chromosome)
                if self._metadata['balance']:
                    mtx_df.loc[:, 'count'] = mtx_df.loc[:, 'balanced']
                    mtx_df = mtx_df.drop('balanced', axis=1)
                    mtx_df = mtx_df.dropna()
                mtx_df.to_csv(output_filename, index=False, sep='\t', header=False)

                return output_filename

            elif input_format == 'cool' and 'mr_sparse' in output_format:
                output_prefix = '.'.join(input_filename.split('.')[:-1])
                output_filename = output_prefix + '.{}.mr_sparse.txt'.format(chromosome)

                c = cooler.Cooler(input_filename)
                mtx = c.matrix(balance=True, as_pixels=True).fetch(chromosome, chromosome)
                mtx.loc[:, "bin1_id":"bin2_id"] += 1

                mtx.loc[:, 'bin1_id':'count'].to_csv(output_filename, header=False, index=False, sep='\t')

                max_bin = mtx.loc[:, 'bin1_id':'bin2_id'].max().max()

                with open(output_prefix + ".{}.genome_bin.txt".format(chromosome), 'w') as outfile:
                    outfile.write("1\tchr1\t0\t{}".format(max_bin - 1))

                with open(output_prefix + ".{}.all_bins.txt".format(chromosome), 'w') as outfile:
                    for i in range(max_bin):
                        outfile.write("0\t{}\t{}\n".format(i * resolution + 1, (i + 1) * resolution))

                return output_filename

            elif 'txt' in input_format and output_format == 'cool':
                # TODO implement this option
                # https://github.com/hms-dbmi/higlass/issues/100#issuecomment-302183312
                logger.error('Option currently not importmented!')

            elif input_format == 'cool' and output_format == 'h5':
                output_prefix = '.'.join(input_filename.split('.')[:-1])
                output_filename = output_prefix + '.h5'
                command = "hicExport --inFile {} --outFileName {} --inputFormat cool --outputFormat h5".format(input_filename, output_filename)
                run_command(command)
                return output_filename

            elif input_format == 'cool' and output_format == 'hic':

                binary_path = kwargs.get('binary_path', 'java')
                juicer_path = kwargs.get('juicer_path', './juicer_tools.1.8.9_jcuda.0.8.jar')
                genome      = kwargs.get('genome', 'dm3')

                output_prefix = '.'.join(input_filename.split('.')[:-1])
                outfile_hic = "{}.{}.hic".format(output_prefix, chromosome)

                outfile_txt = outfile_hic + '.txt'
                outfile_tmp = outfile_hic + '.tmp'

                with open(outfile_tmp, 'w'):
                    pass

                outfile_tmp = self.convert_file(input_filename=input_filename, input_format='cool', output_format='sparse')

                command1 = "awk '{{print 0, $1, $2, 0, 0, $4, $5, 1, $7}}' {} > {}".format(outfile_tmp, outfile_txt)
                command2 = "gzip -f {}".format(outfile_txt)
                command3 = "{} -Xmx2g -jar {} pre -r {} -c {} {}.gz {} {}".format(binary_path, juicer_path, resolution,
                                                                                  chromosome, outfile_txt, outfile_hic, genome)

                run_command(command1)
                run_command(command2)
                run_command(command3)

                if remove_intermediary_files:
                    os.remove(outfile_txt + '.gz')

        elif mtx:
            if tune:
                mtx = self.tune_matrix(mtx)

                if 'txt' in output_format:
                    np.savetxt(output_filename, mtx, delimiter='\t')

                    if 'gz' in output_format:
                        command = 'gzip {}'.format(output_filename)
                        run_command(command)
                        output_filename += '.gz'

                elif output_format == 'cool':
                    logger.error('Option currently not importmented!')

            return output_filename

        else:
            raise Exception("Neither input filename nor matrix are presented.")

    def convert_files(self, data_format, **kwargs):
        """
        Converts input files into required format.
        :param data_format: format of resulting files ('cool', 'txt', 'txt.gz', 'hic', 'h5')
        :param kwargs: Optional parameters for data coversion
        :return: None
        """
        tune = kwargs.get('tune', False)
        #as_pixels = kwargs.get('as_pixels', False)

        assert data_format in ACCEPTED_FORMATS

        original_format = kwargs.get('original_format', self._metadata['data_formats'][0])

        resulting_files = []
        file_holder = 'files_{}'.format(original_format)
        param_dict = {
            'tune': tune,
            'balance': self._metadata['balance'],
            #'as_pixels': as_pixels,
            'ch': self._metadata['chr'],
            'res': self._metadata['resolution']
        }
        for k in ['remove_intermediary_files', 'juicer_path']:
            if k in kwargs.keys():
                param_dict[k] = kwargs.get(k)

        for f in self._metadata[file_holder]:
            output_fname = self.convert_file(input_filename=f,
                                             input_format=original_format,
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
        #return segmentations

    def segmentation2df(self):
        """
        Converter of all segmentations to Pandas dataframe.
        :return:
        """
        params_names = self._metadata['params']
        self._df = pd.DataFrame(columns=['bgn', 'end', 'label', 'caller'] + params_names)
        for x in self._segmentations.keys():
            for y in self._segmentations[x].keys():
                length = self._segmentations[x][y].length
                if length == 0 or self._segmentations[x][y].data.shape[1] == 0:
                    continue  # it would be better to store zero segmentations in order to compare
                dct = {
                    'bgn': self._segmentations[x][y].data[:, 0],
                    'end': self._segmentations[x][y].data[:, 1],
                    'label': [x for i in range(length)],
                    'caller': [self._metadata['caller'] for i in range(length)]
                }
                if isinstance(y, int) or isinstance(y, float):
                    y = [y]
                for param, value in zip(params_names, y):
                    dct.update({param: value})

                df = pd.DataFrame(dct)

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

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(LavaburstCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['gamma', 'method']
        self._metadata['caller'] = 'Lavaburst'


    def call(self, params_dict={}, tune=True, **kwargs):
        """
        Lavaburst segmentation calling for a set of parameters.
        :param params_dict: dictionary of parameters, containing gammas and methods
        :param tune:
        :param kwargs:
        :return:
        """

        params_dict['gamma'] = params_dict.get('gamma', np.arange(0,10,1))
        params_dict['method'] = params_dict.get('method', ['armatus'])

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            output_dct = {}

            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'],
                                                                                     self._metadata['chr'])
            if tune:
                mtx = self.tune_matrix(mtx)

            for gamma in params_dict['gamma']:
                for method in params_dict['method']:
                    segmentation = self._call_single(mtx, gamma, method=method, **kwargs)
                    output_dct = {label: deepcopy(segmentation)}
                    self._load_segmentations(output_dct, (gamma, method))

        #eturn self._segmentations

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

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        super(ArmatusCaller, self).__init__(datasets_labels, datasets_files, data_format, **kwargs)
        self._metadata['params'] = ['gamma']
        self._metadata['caller'] = 'Armatus'

    def call(self, params_dict={}, tune=True, **kwargs):

        params_dict['gamma'] = params_dict.get('gamma', np.arange(0,10,1))

        if 'files_txt.gz' not in self._metadata.keys():
            self.convert_files('txt.gz', tune=tune)

        for gamma in params_dict['gamma']:
            output_dct = {}
            for label, f in zip(self._metadata['labels'], self._metadata['files_txt.gz']):
                segmentation = self._call_single(f, gamma, **kwargs)
                output_dct[label] = deepcopy(segmentation)
            self._load_segmentations(output_dct, (gamma))

        #return self._segmentations

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

    def call(self, params_dict={}, tune=True, **kwargs):

        params_dict['window'] = params_dict.get('window', [1])
        params_dict['cutoff'] = params_dict.get('cutoff', [0])

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'], self._metadata['chr'])

            if tune:
                mtx = self.tune_matrix(mtx)

            for window in params_dict['window']:
                for cutoff in params_dict['cutoff']:
                    segmentation = self._call_single(mtx, window, cutoff, **kwargs)
                    output_dct = {label: deepcopy(segmentation)}

                    self._load_segmentations(output_dct, (window, cutoff))

        #return self._segmentations

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
            logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            logger.warning(
                "Note that diagonal is not removed. you might want to delete it to avoid noisy and not stable results. ")

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

    def call(self, params_dict={}, tune=True, **kwargs):

        params_dict['window'] = params_dict.get('window', [1])
        params_dict['cutoff'] = params_dict.get('cutoff', [0])

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'], self._metadata['chr'])

            if tune:
                mtx = self.tune_matrix(mtx)

            for window in params_dict['window']:
                for cutoff in params_dict['cutoff']:
                    segmentation = self._call_single(mtx, window, cutoff, **kwargs)
                    output_dct = {label: deepcopy(segmentation)}

                    self._load_segmentations(output_dct, (window, cutoff))

        #return self._segmentations

    def _call_single(self, mtx, window, cutoff, max_intertad_size=3, max_tad_size=10000, **kwargs):

        if np.any(np.isnan(mtx)):
            logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            logger.warning(
                "Note that diagonal is not removed. you might want to delete it to avoid noisy and not stable results. ")

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

    def call(self, params_dict={}, tune=True, **kwargs):

        params_dict['distr_model'] = params_dict.get('distr_model', ["P"])

        if 'files_txt' not in self._metadata.keys():
            self.convert_files('txt', tune=tune)

        for label, f in zip(self._metadata['labels'], self._metadata['files_txt']):
            for distr_model in params_dict['distr_model']:
                segmentation = self._call_single(f, distr_model, **kwargs)
                output_dct = {label: deepcopy(segmentation)}

                self._load_segmentations(output_dct, (distr_model))

        #return self._segmentations

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

    def call(self, tune=False, **kwargs):

        caller_path = kwargs.get('caller_path', '../MrTADFinder/run_MrTADFinder.jl')
        output_dct = {}

        if 'files_mr_sparse' not in self._metadata.keys():
            self.convert_files('mr_sparse', tune=tune)

        for label, f in zip(self._metadata['labels'], self._metadata['files_mr_sparse']):
            segmentation = self._call_single(f, caller_path=caller_path)
            output_dct[label] = deepcopy(segmentation)

        self._load_segmentations(output_dct, (gamma))

        #return self._segmentations

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

class HiCExplorerCaller(BaseCaller):

    def call(self, params_dict={}, **kwargs):

        if 'files_h5' not in self._metadata.keys():
            raise BasicCallerException("No h5 file present for caller. Please, perform valid conversion!")

        params_dict['minDepth'] = params_dict.get('minDepth', [5])
        params_dict['maxDepth'] = params_dict.get('minDepth', [10])
        params_dict['step']     = params_dict.get('step', [1500])
        params_dict['thresholdComparisons'] = params_dict.get('thresholdComparisons', [0.05])
        params_dict['delta']    = params_dict.get('delta', [0.01])
        params_dict['correction']    = params_dict.get('correction', ['fdr'])

        for minDepth in params_dict['minDepth']:
            for maxDepth in params_dict['maxDepth']:
                for step in params_dict['step']:
                    for thresholdComparisons in params_dict['thresholdComparisons']:
                        for delta in params_dict['delta']:
                            for correction in params_dict['correction']:
                                output_dct = {}
                                for label, f in zip(self._metadata['labels'], self._metadata['files_h5']):
                                    outmask = "tmp/{}.tmp".format(label)
                                    segmentation = self._call_single(f, outmask, minDepth, maxDepth, step, thresholdComparisons, delta, correction, **kwargs)
                                    output_dct[label] = deepcopy(segmentation)
                                self._load_segmentations(output_dct, (minDepth, maxDepth, step, thresholdComparisons, delta, correction))

        return self._segmentations

    def _call_single(self, infile, outmask, minDepth, maxDepth,
                     step, thresholdComparisons=0.05, delta=0.01,
                     correction="fdr", nthreads=1, caller_path='hicFindTADs', **kwargs):

        command = """{} -m {} \
                --outPrefix {} \
                --minDepth {} \
                --maxDepth {} \
                --step {} \
                --thresholdComparisons {} \
                --delta {} \
                --correctForMultipleTesting {} \
                -p {}""".format(caller_path, infile, outmask, minDepth, maxDepth,
                                step, thresholdComparisons, delta, correction, nthreads)

        run_command(command)


        #return GenomicRanges(np.array(segments, dtype=int), data_type='segmentation')
