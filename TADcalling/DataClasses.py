"""
Classes for data handling like TAD segmentation
or genomic tracks.
"""

from .utils import *
from .logger import logger
import numpy as np


class GenomicRanges(object):
    """
    Basic class for any genomic ranges.
    """
    # TODO: define input method. Maybe inheriting from np.array?

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

    def _TAD_bins(arr1, arr2):
        """
        Returns TADs as as objects from their coordinates.
        """
        _vector_str = np.vectorize(str)
        return npchar.add(_vector_str(arr1), npchar.add(",", _vector_str(arr2)))

    def _TAD_boundaries(arr1, arr2):
        """
        Returns TAD unique boundaries.
        """
        return np.unique(np.append(arr1, arr2))

    def _jaccard_index(arr1, arr2):
        """
        Calculate Jaccard index between two 1-dimensional
        arrays containing genomic ranges.
        """
        intersection = np.isin(arr1, arr2)
        return sum(intersection) / (arr1.shape[0] + arr2.shape[0] - sum(intersection))

    def _overlap_coef(arr1, arr2):
        """
        Calculate Overlap coefficient between two 1-dimensional
        arrays containing genomic ranges.
        """
        intersection = np.isin(arr1, arr2)
        return sum(intersection) / min(arr1.shape[0], arr2.shape[0])

    # TODO: redefine coefs with np.intersect1d, np.union1d, np.setdiff1d. Set routines in numpy.

    def count_coef(self, other, coef="JI TADs"):
        """
        Calculate coefficient between two genomic ranges.
        :param type: JI TADs, JI boundaries, OC TADs, OC boundaries, TPR, FDR.

        In JI and OC other might be any genomic range, in FDR and TPR other is
        true segmentation obtained from simulation.
        """
        if coef == 'JI TADs':
            return _jaccard_index(_TAD_bins(self[:, 0], self[:, 1]), _TAD_bins(other[:, 0], other[:, 1]))

        elif coef == 'JI boundaries':
            return jaccard_index(_TAD_boundaries(self[:, 0], self[:, 1]), _TAD_boundaries(other[:, 0], other[:, 1]))

        elif coef == 'OC TADs':
            return _overlap_coef(TAD_bins(self[:, 0], self[:, 1]), TAD_bins(other[:, 0], other[:, 1]))

        elif coef == 'OC boundaries':
            return _overlap_coef(_TAD_boundaries(self[:, 0], self[:, 1]), _TAD_boundaries(other[:, 0], other[:, 1]))

        elif coef == 'TPR TADs':
            return sum(np.isin(_TAD_bins(self), _TAD_bins(other))) / _TAD_bins(other).shape[0]

        elif coef == 'FDR TADs':
            return sum(~np.isin(_TAD_bins(self), _TAD_bins(other))) / _TAD_bins(self).shape[0]

        elif coef == 'TPR boundaries':
            return sum(np.isin(_TAD_boundaries(self), _TAD_boundaries(other))) / _TAD_boundaries(other).shape[0]

        elif coef == 'FDR boundaries':
            return sum(~np.isin(_TAD_boundaries(self), _TAD_boundaries(other))) / _TAD_boundaries(self).shape[0]

        else:
            raise Exception('Coefficient not understood: {}'.format(coef))

    # TODO: define JI and OC for boundaries with offset as in @agal
    # TODO: count number of shared TADs.
    # TODO: implement bedtools intersect with 70% coverage. Easy in while loop
    # by O(n+m), but want to to implement in numpy.