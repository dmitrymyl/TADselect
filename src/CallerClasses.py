"""
Classes for TAD calling for various tools
"""

from utils import *

ACCEPTED_FORMATS = ['cool', 'txt', 'txt.gz']

class BasicCallerException(Exception):
    pass

class BaseCaller(object):

    def __init__(self, datasets_labels, datasets_files, data_format, **kwargs):
        """
        Initialize base TAD caller

        :param datasets_labels (:obj:`list`): short names of files that need to be processed
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
            self._metadata -- metadata, fict with 'assembly', 'chr','resolution', 'labels', 'balance' keys
            self._segmentations -- dictionary with segmentations for all the files
        """

        assert len(datasets_labels)==len(datasets_files)

        metadata = kwargs.get('metadata', {})
        if not isinstance(metadata, dict):
            raise TypeError("Metadata should be a dictionary not a %s" % str(type(metadata)))

        metadata['assembly'] = kwargs.get('assembly', '-')
        metadata['chr'] = kwargs.get('chromosome', 'chr1')
        metadata['size'] = kwargs.get('size', 0)
        metadata['resolution'] = metadata.get('resolution', 1000)
        metadata['labels'] = datasets_labels
        metadata['balance'] = kwargs.get('balance', True)

        assert data_format in ACCEPTED_FORMATS
        metadata['data_formats'] = [data_format]
        metadata['files_{}'.format(data_format)] = datasets_files

        self._metadata = metadata

        self._segmentations = {x:{} for x in dataset_labels}

    def convert_files(self, data_format, **kwargs):
        """
        Converting nput files into required format.
        :param data_format: format of resulting files ('cool', 'txt', 'txt.gz')
        :param kwargs: Optional parameters for data coversion
        :return: None
        """

        assert data_format in ACCEPTED_FORMATS

        original_format = self._metadata['data_formats'][0]

        resulting_files = []
        if original_format=='oool' and 'txt' in data_format:
            for f in self._metadata['files_cool']:
                output_name = '.'.join(x.split('.')[:-1])+'txt'
                c = cooler.Cooler(f, balance=self._metadata['balance'])
                mtx = c.fetch(self._metadata['chr'])
                np.savetxt(output_name, mtx, delimiter='\t')
                resulting_files.append(output_name)
            if 'gz' in data_format:
                subprocess.call('gzip {}'.format(output_name), shell=true)
        elif 'txt' in original_format and data_format=='cool':
            # TODO implement this option
            logger.ERROR('Option currently not importmented!')

        self._metadata['files_{}'.format(data_format)] = resulting_files
        self._metadata['data_formats'].append(data_format)

    def call(self, params):
        """
        Basic function of BaseCaller to call the segmentation (and writ eit to file or to memory
        :param params: set of calling parameters
        :return: dict (ordered by metadata['labels']) with segmentations (2d np.ndarray) or names of files
        """
        logger.DEBUG("Calling dump BaseCaller segmentation call with params: {}".format(str, params))
        return {x: np.empty([0,0]) for x in self._metadata['labels']}


    def load_segmentation(self, input, params):
        """
        Basic function of BaseCaller to load the segmentation (from file or from variable)
        :param input dict (ordered by metadata['labels']) with segmentations (2d np.ndarray) or names of files
        :param params: set of parameters
        :return:
        """
        logger.DEBUG("Calling dump BaseCaller segmentation loading")

        for x in self._metadata['labels']:
            self._segmentations[x][params] = input[self._metadata['labels']]

    def segmentation2df(self):
        """
        Converter of all segmentations to Pandas dataframe.
        :return:
        """
        self._df = pd.DataFrame(columns=['bgn', 'end', 'label', 'params'])
        for x in self._segmentations.keys():
            for y in self._segmentations[x].keys():
                lngth = len(self._segmentations[x][y])
                df = pd.DataFrame({
                    'bgn': self._segmentations[x][y][:,0],
                    'end': self._segmentations[x][y][:,1],
                    'label': [x for x in range(lngth)],
                    'params':[str(y) for y in range(lngth)]
                })
                self._df.append(df)

        return self._df

class LavaburstCaller(BaseCaller):

    def call(self, gamma, **kwargs):

        if not 'files_cool' in self._metadata.keys():
            raise BasicCallerException("No oool file present for caller. Please, perform valid conversion!")

        output_dct = {}

        for label, f in zip(self._metadata['labels'], self._metadata['files_cool']):
            c = cooler.Cooler(f, balance=self._metadata['balance'])
            mtx = c.fetch(self._metadata['chr'])
            segmentation = self._call_single(mtx, gamma, **kwargs)

            output_dct[label] = {}
            output_dct[label][(gamma)] = segmentation.copy()

         return output_dct

    def _call_single(self, mtx, gamma, good_bins='default', method='armatus', max_intertad_size=3, max_tad_size=10000):
        """
        Produces single segmentation (TADs calling) of mtx with one gamma with the algorithm provided.
        :param gamma: parameter for segmentation calling
        :param good_bins: bool vector with length of len(mtx) with False corresponding to masked bins, 'default' is that
            good bins are all columns/rows with sum > 0
        :param method: 'modularity' (default) or 'armatus'
        :param max_intertad_size: max size of segmentation unit that is considered as interTAD
        :return:  2D numpy array where segments[:,0] are segment starts and segments[:,1] are segments end, each row corresponding to one segment
        """

        if np.any(np.isnan(mtx)):
            logging.warning("NaNs in dataset, pease remove them first.")

        if np.diagonal(mtx).sum() > 0:
            logger.warning(
                "Note that diagonal is not removed. you might want to delete it to avoid noisy and not stable results. ")

        if method == 'modularity':
            score = lavaburst.scoring.modularity_score
        elif method == 'armatus':
            score = lavaburst.scoring.armatus_score
        else:
            # TODO add all the options from lavaburst package
            raise BasicCallerException('Algorithm not understood: {}'.format(method))

        if good_bins == 'default':
            good_bins = mtx.astype(bool).sum(axis=0) > 0

        S = score(mtx, gamma=gamma, binmask=good_bins)
        model = lavaburst.model.SegModel(S)

        segments = model.optimal_segmentation()

        v = segments[:, 1] - segments[:, 0]
        mask = (v > max_intertad_size) & (np.isfinite(v)) & (v < max_tad_size)

        segments = segments[mask]

        return np.array(segments)

    def load_segmentation(self, input):
        for x in input.keys():
            for params in input[x].keys():
                self._segmentations[x][params] = input[x][params]