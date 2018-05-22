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
    # TODO: implement load from bed file with 3 or 4 columns - make a function

    def __init__(self, arr_object, arr_count=None, data_type='simulation', **kwargs):
        """
        Initialize base genomic ranges object.
        :param arr_object (:obj:`collection`): any array-like object
        :param arr_count (:obj: `collection`): any array-like object of the same length as arr_object
        :param data_type (:obj: `str`): type of data (simulated segmentation (default), genomic track)
        :param kwargs:
            * assembly (:obj:`str`): name of assembly, default is '-'
            * chromosome (:obj:`str`): name of chromosome, default is 'chr1'
            * size (:obj:`int`): size of dataset in bins, default is 0
            * resolution (:obj:`int`): size of bin in basepairs, default is 1000
            * metadata (:obj:`dict`): optional metadata dict, default is empty dict
        """
        self.data = np.array(arr_object, dtype=int, ndmin=2)
        if self.data.shape[1] != 2:
            raise Exception("Shape of arr_object is not n x 2")
        self.length = self.data.shape[0]
        if arr_count:
            self.count = arr_count
            if self.count.shape[0] != self.length:
                raise Exception("Size of count array and amount of ranges are not equal.")
        else:
            # maybe fill with zeros?
            self.count = np.full_like(self.data[:, 0], 1, dtype=int)
        self.data_type = data_type
        metadata = kwargs.get('metadata', {})
        if not isinstance(metadata, dict):
            raise TypeError("Metadata should be a dictionary not a %s" % str(type(metadata)))

        metadata['assembly'] = kwargs.get('assembly', '-')
        metadata['chr'] = kwargs.get('chr', 'chr1')
        metadata['size'] = kwargs.get('size', 0)
        metadata['resolution'] = metadata.get('resolution', 1000)
        self._metadata = metadata
    
    @staticmethod
    def TAD_bins(arr):
        """
        Returns TADs as as objects from their coordinates.
        """
        _vector_str = np.vectorize(str)
        return npchar.add(_vector_str(arr[:, 0]), npchar.add(",", _vector_str(arr[:, 1])))
    
    @staticmethod
    def TAD_boundaries(arr):
        """
        Returns TAD unique boundaries.
        """
        return np.unique(np.append(arr[:, 0], arr[:, 1]))
    
    @staticmethod
    def jaccard_index(arr1, arr2):
        """
        Calculate Jaccard index between two 1-dimensional
        arrays containing genomic ranges.
        """
        intersection = np.isin(arr1, arr2)
        return sum(intersection) / (arr1.shape[0] + arr2.shape[0] - sum(intersection))
    
    @staticmethod
    def overlap_coef(arr1, arr2):
        """
        Calculate Overlap coefficient between two 1-dimensional
        arrays containing genomic ranges.
        """
        intersection = np.isin(arr1, arr2)
        return sum(intersection) / min(arr1.shape[0], arr2.shape[0])

    # Redefine coefs with np.intersect1d, np.union1d, np.setdiff1d? Set routines in numpy.

    def count_coef(self, other, coef="JI TADs"):
        """
        Calculate coefficient between two genomic ranges.
        :param type: JI TADs, JI boundaries, OC TADs, OC boundaries, TPR, FDR.

        In JI and OC other might be any genomic range, in FDR and TPR other is
        true segmentation obtained from simulation.
        """
        if coef == 'JI TADs':
            return GenomicRanges.jaccard_index(GenomicRanges.TAD_bins(self.data), GenomicRanges.TAD_bins(other.data))

        elif coef == 'JI boundaries':
            return GenomicRanges.jaccard_index(GenomicRanges.TAD_boundaries(self.data), GenomicRanges.TAD_boundaries(other.data))

        elif coef == 'OC TADs':
            return GenomicRanges.overlap_coef(GenomicRanges.TAD_bins(self.data), GenomicRanges.TAD_bins(other.data))

        elif coef == 'OC boundaries':
            return GenomicRanges.overlap_coef(GenomicRanges.TAD_boundaries(self.data), GenomicRanges.TAD_boundaries(other.data))

        elif coef == 'TPR TADs':
            return sum(np.isin(GenomicRanges.TAD_bins(self.data), GenomicRanges.TAD_bins(other.data))) / GenomicRanges.TAD_bins(other.data).shape[0]

        elif coef == 'FDR TADs':
            return sum(~np.isin(GenomicRanges.TAD_bins(self.data), GenomicRanges.TAD_bins(other.data))) / GenomicRanges.TAD_bins(self.data).shape[0]

        elif coef == 'TPR boundaries':
            return sum(np.isin(GenomicRanges.TAD_boundaries(self.data), GenomicRanges.TAD_boundaries(other.data))) / GenomicRanges.TAD_boundaries(other.data).shape[0]

        elif coef == 'FDR boundaries':
            return sum(~np.isin(GenomicRanges.TAD_boundaries(self.data), GenomicRanges.TAD_boundaries(other.data))) / GenomicRanges.TAD_boundaries(self.data).shape[0]

        else:
            raise Exception('Coefficient not understood: {}'.format(coef))

    # TODO: define JI and OC for boundaries with offset as in @agal
    # TODO: count number of shared TADs.
    # TODO: implement bedtools intersect with 70% coverage. Easy in while loop
    # by O(n+m), but want to implement in numpy.