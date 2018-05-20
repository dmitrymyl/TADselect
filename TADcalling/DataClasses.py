"""
Classes for data handling like TAD segmentation
or genomic tracks.
"""

from .utils import *
from .logger import logger

class GenomicRanges(object):
    """
    Basic class for any genomic ranges.
    """

    def __init__(self, name, data_type='simulation', **kwargs):
        """
        Initialize base genomic ranges object.
        :param name (:obj:`str`): specific name of given dataset
        :param data_type (:obj: `str`): type of data (simulated segmentation (default), genomic track)
        :param kwargs:
            * assembly (:obj:`str`): name of assembly, default is '-'
            * chromosome (:obj:`str`): name of chromosome, default is 'chr1'
            * size (:obj:`int`): size of dataset in bins, default is 0
            * resolution (:obj:`int`): size of bin in basepairs, default is 1000
            * metadata (:obj:`dict`): optional metadata dict, default is empty dict
            Object of BaseCaller type or derivative has two attributes:
            self._metadata -- metadata, fict with 'assembly', 'chr','resolution' keys
        """
        self.name = name
        self.data_type = data_type
        metadata = kwargs.get('metadata', {})
        if not isinstance(metadata, dict):
            raise TypeError("Metadata should be a dictionary not a %s" % str(type(metadata)))

        metadata['assembly'] = kwargs.get('assembly', '-')
        metadata['chr'] = kwargs.get('chr', 'chr1')
        metadata['size'] = kwargs.get('size', 0)
        metadata['resolution'] = metadata.get('resolution', 1000)
        self._metadata = metadata

class TADSegmentation(GenomicRanges):
    """
    Class for TAD segmentation.
    """
    pass

class GenomicTrack(GenomicRanges)
