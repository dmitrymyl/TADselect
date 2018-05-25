"""
Classes for TAD calling for various tools
"""

from .utils import *
from .logger import logger
from .DataClasses import GenomicRanges, load_BED
from copy import deepcopy

ACCEPTED_FORMATS = ['cool', 'txt', 'txt.gz']

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

        self._segmentations = {x:{} for x in datasets_labels}

    def convert_files(self, data_format, **kwargs):
        """
        Converting Input files into required format.
        :param data_format: format of resulting files ('cool', 'txt', 'txt.gz')
        :param kwargs: Optional parameters for data coversion
        :return: None
        """

        assert data_format in ACCEPTED_FORMATS

        original_format = self._metadata['data_formats'][0]
        ch = self._metadata['chr']

        resulting_files = []
        if original_format == 'cool' and 'txt' in data_format:
            for f in self._metadata['files_cool']:
                output_prefix = '.'.join(f.split('.')[:-1])
                output_file = output_prefix + '.{}.txt'.format(ch)

                c = cooler.Cooler(f)
                mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(ch, self._metadata['chr'])

                np.savetxt(output_file, mtx, delimiter='\t')

                if 'gz' in data_format:
                    subprocess.call('gzip {}'.format(output_name), shell=true)
                    output_name += ".gz"

                resulting_files.append(output_name)

        elif 'txt' in original_format and data_format == 'cool':
            # TODO implement this option
            # https://github.com/hms-dbmi/higlass/issues/100#issuecomment-302183312
            logger.error('Option currently not importmented!')

        self._metadata['files_{}'.format(data_format)] = resulting_files
        self._metadata['data_formats'].append(data_format)

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

    def call(self, gamma, **kwargs):

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        output_dct = {}

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'], self._metadata['chr'])

            mtx[np.isnan(mtx)] = 0
            np.fill_diagonal(mtx, 0)
            mn = np.percentile(mtx[mtx > 0], 1)
            mx = np.percentile(mtx[mtx > 0], 99)
            mtx[mtx <= mn] = mn
            mtx[mtx >= mx] = mx
            mtx = np.log(mtx)
            mtx = mtx - np.min(mtx)

            segmentation = self._call_single(mtx, gamma, **kwargs)

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

    def call(self, gamma, **kwargs):

        if 'files_cool' not in self._metadata.keys():
            raise BasicCallerException("No cool file present for caller. Please, perform valid conversion!")

        output_dct = {}

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f)
            mtx = c.matrix(balance=self._metadata['balance'], as_pixels=False).fetch(self._metadata['chr'], self._metadata['chr'])

            mtx[np.isnan(mtx)] = 0
            np.fill_diagonal(mtx, 0)
            mn = np.percentile(mtx[mtx > 0], 1)
            mx = np.percentile(mtx[mtx > 0], 99)
            mtx[mtx <= mn] = mn
            mtx[mtx >= mx] = mx
            mtx = np.log(mtx)
            mtx = mtx - np.min(mtx)

            segmentation = self._call_single(mtx, gamma, **kwargs)

            output_dct[label] = segmentation.copy()

        self._load_segmentations(output_dct, (gamma))

        return self._segmentations


    def _call_single(self, mtx, gamma, good_bins='default', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """

        if np.any(np.isnan(mtx)):
            logger.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            logger.warning(
                "Note that diagonal is not removed. you might want to delete it to avoid noisy and not stable results. ")

        # TODO: implement CLI via subprocess. Convert matrix to txt.gz, store segmentation in buff.consensus.txt and
        # extract it with np.loadtxt, then delete chr column
        # subprocess.run("/armatus -i <input.txt.gz> -g <gamma> -j -o <output buff.txt> -r <resolution>")
        # segments = np.loadtxt("buff.txt", ndmin=2, dtype=object)
        # segments = np.array(segments[:, 1:], dtype=int)
        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return np.array(segments)