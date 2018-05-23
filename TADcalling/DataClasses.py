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
        buff = np.array(arr_object, dtype=int, ndmin=2)
        if buff.shape[1] == 2:
            self.data = buff
            self.count = np.full_like(self.data[:, 0], 1, dtype=int)
        elif buff.shape[1] == 3:
            self.data = buff[:, 0:1]
            self.count = buff[:, 2]
        else:
            raise Exception("Shape of arr_object is not as required.")

        self.length = self.data.shape[0]
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

    @staticmethod
    def find_intersect(arr1, arr2, ident=1):
        """
        Return logical indexes of intersecting ranges in two
        2d arrays with given identity regarding the first array.
        Asymmetrical!
        """
        mask1 = np.zeros(arr1.shape[0], dtype=bool)
        mask2 = np.zeros(arr2.shape[0], dtype=bool)
        i, k = 0, 0
        while i < arr1.shape[0] and k < arr2.shape[0]:
            if arr1[i, 0] >= arr2[k, 1]:
                k += 1
            elif arr1[i, 1] <= arr2[k, 0]:
                i += 1
            else:
                intersecting = min(arr1[i, 1], arr2[k, 1]) - max(arr1[i, 0], arr2[k, 0]) + 1
                if (intersecting / (arr1[i, 1] - arr1[i, 0] + 1)) >= ident:
                    mask1[i], mask2[k] = True, True
                k += 1
        return mask1, mask2

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

    def count_shared(self, other, ident=1):
        """
        Return share of shared TADs regarding the first range with given identity.
        Asymmetrical!
        """
        shared1, shared2 = map(sum, find_intersect(self.data, other.data, ident=ident))
        # TODO: consider the formulae.
        return shared1 / (self.length + other.length - shared1)

    # TODO: define JI and OC for boundaries with offset as in @agal

# TODO: test this.
def load_BED(filename):
    """
    Return dictionary of GenomicRanges with chromosomes 
    as keys from BED-like file. Can load 2-column file,
    3- and 6-column BED files.
    """
    buff = np.loadtxt(filename, dtype=object, ndmin=2)
    if buff.shape[1] not in (2, 3, 6):
        raise Exception("Given file is not BED-like.")
    elif buff.shape[1] == 2:
        return {"chr1": GenomicRanges(buff)}
    else:
        chrms = np.unique(buff[:, 0])
        indices = [np.searchsorted(buff[:, 0], chrm) for chrm in chrms]
        if buff.shape[1] == 3:
            return {chrm: GenomicRanges(arr) for chrm, arr in zip(chrms, np.vsplit(buff[:, 1:], indices))}
        if buff.shape[1] == 6:
            return {chrm: GenomicRanges(arr) for chrm, arr in zip(chrms, np.vsplit(buff[:, [1, 2, 4]], indices))}
