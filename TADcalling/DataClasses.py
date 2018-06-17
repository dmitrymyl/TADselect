"""
Classes for data handling like TAD segmentation
or genomic tracks.
"""

from os import stat as file_stat
from .utils import *
from .logger import TADcalling_logger
import numpy as np
import numpy.core.defchararray as npchar


class GenomicRanges(object):
    """
    Basic class for any genomic ranges.
    """

    def __init__(self, arr_object, data_type='simulation', **kwargs):
        """
        Initialize base genomic ranges object.
        :param arr_object (:obj:`collection`): any array-like object
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
            self.coverage = np.full_like(self.data[:, 0], 1, dtype=int)
        elif buff.shape[1] == 3:
            self.data = buff[:, 0:1]
            self.coverage = buff[:, 2]
        elif buff is None or buff.shape[1] == 0:  # No segments in a segmentation, TODO: @dmyl check
            self.data = np.zeros((1, 2))
            self.coverage = np.zeros(1)
        else:
            raise Exception("Inappropriate shape of arr_object: {}.".format(buff.shape))

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

    def __repr__(self):
        return '\n'.join(['\t'.join([str(self.data[i, 0]), str(self.data[i, 1]), str(self.coverage[i])]) for i in range(self.length)])

    @staticmethod
    def TAD_bins(arr):
        """
        Returns TADs as as objects from their coordinates.
        """
        if arr.shape[0]:
            _vector_str = np.vectorize(str)
            return npchar.add(_vector_str(arr[:, 0]), npchar.add(",", _vector_str(arr[:, 1])))
        return arr

    @staticmethod
    def TAD_boundaries(arr):
        """
        Returns TAD unique boundaries.
        """
        if arr.shape[0]:
            return np.unique(np.append(arr[:, 0], arr[:, 1]))
        return arr

    @staticmethod
    def jaccard_index(arr1, arr2):
        """
        Calculate Jaccard index between two 1-dimensional
        arrays containing genomic ranges.
        """
        if arr1.shape[0] * arr2.shape[0]:
            intersection = np.isin(arr1, arr2)
            return sum(intersection) / (arr1.shape[0] + arr2.shape[0] - sum(intersection))
        return 0

    @staticmethod
    def overlap_coef(arr1, arr2):
        """
        Calculate Overlap coefficient between two 1-dimensional
        arrays containing genomic ranges.
        """
        if arr1.shape[0] * arr2.shape[0]:
            intersection = np.isin(arr1, arr2)
            return sum(intersection) / min(arr1.shape[0], arr2.shape[0])
        return 0

    @staticmethod
    def TPR(arr1, arr2):
        """
        Calculate TPR of arr2 in arr1.
        """
        if arr1.shape[0] * arr2.shape[0]:
            return sum(np.isin(arr1, arr2)) / arr2.shape[0]
        return 0

    @staticmethod
    def FDR(arr1, arr2):
        """
        Calculate FDR in arr1 regarding arr2.
        """
        if arr1.shape[0] * arr2.shape[0]:
            return sum(~np.isin(arr1, arr2)) / arr1.shape[0]
        return 0

    @staticmethod
    def find_intersect(arr1, arr2, ident=1):
        """
        Return intersecting ranges in two
        2d arrays with given identity regarding the first array.
        Asymmetrical!
        """
        list_intersecting = list()
        i, k = 0, 0
        while i < arr1.shape[0] and k < arr2.shape[0]:
            if arr1[i, 0] >= arr2[k, 1]:
                k += 1
            elif arr1[i, 1] <= arr2[k, 0]:
                i += 1
            else:
                intersection = min(arr1[i, 1], arr2[k, 1]) - max(arr1[i, 0], arr2[k, 0]) + 1
                if (intersection / (arr1[i, 1] - arr1[i, 0] + 1)) >= ident:
                    list_intersecting.append([arr1[i, :], arr2[k, :]])
                if arr1[i, 1] < arr2[k, 1]:
                    i += 1
                else:
                    k += 1
        return np.array(list_intersecting)

    @staticmethod
    def make_offset(arr1, arr2, offset=1):
        """
        Tries to fit 2-dim arr1 to arr2 by shifting
        borders on value not greater then offset.
        Returns fitted arr1 and original arr2.
        """
        if not arr1.shape[0] * arr2.shape[0]:
            return arr1, arr2

        v1 = arr1.copy()
        v2 = arr2.copy()

        def cutzeros(i, offset):
            return i if abs(i) <= offset else 0

        cutzerosvec = np.vectorize(cutzeros)

        # First, find identical end borders and try to fit starts.
        mask_ends_1 = np.isin(v1[:, 1], v2[:, 1])
        mask_ends_2 = np.array(sum(v2[:, 1] == end for end in v1[:, 1][mask_ends_1])).astype(dtype=bool)
        if mask_ends_2.any():
            dists1 = v2[mask_ends_2] - v1[mask_ends_1]
            v1[mask_ends_1] += cutzerosvec(dists1, offset)

        # Second, find identical start borders and try to fit ends.
        mask_starts_1 = np.isin(v1[:, 0], v2[:, 0])
        mask_starts_2 = np.array(sum(v2[:, 0] == end for end in v1[:, 0][mask_starts_1])).astype(dtype=bool)
        if mask_starts_2.any():
            dists2 = v2[mask_starts_2] - v1[mask_starts_1]
            v1[mask_starts_1] += cutzerosvec(dists2, offset)

        # Finally, try to fit both starts and ends.
        start_dist = np.array([min(v2[:, 0] - i, key=abs) for i in v1[:, 0]], dtype=int)
        end_dist = np.array([min(v2[:, 1] - i, key=abs) for i in v1[:, 1]], dtype=int)
        mask_start = abs(start_dist) <= offset
        mask_end = abs(end_dist) <= offset
        if mask_start.any():
            v1[:, 0][mask_start] += start_dist[mask_start]
        if mask_end.any():
            v1[:, 1][mask_end] += end_dist[mask_end]
        return v1, v2

    # Redefine with two dictionaries of methods? Where to place dictionaries?
    def count_coef(self, other, coef="JI TADs", offset=0):
        """
        Calculate coefficient between two genomic ranges.
        :param type: JI TADs, JI boundaries, OC TADs, OC boundaries, TPR, FDR.

        In JI and OC other might be any genomic range, in FDR and TPR other is
        true segmentation obtained from simulation.
        """
        if coef == 'JI TADs':
            return GenomicRanges.jaccard_index(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'JI boundaries':
            return GenomicRanges.jaccard_index(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'OC TADs':
            return GenomicRanges.overlap_coef(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'OC boundaries':
            return GenomicRanges.overlap_coef(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'TPR TADs':
            return GenomicRanges.TPR(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'FDR TADs':
            return GenomicRanges.FDR(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'TPR boundaries':
            return GenomicRanges.TPR(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'FDR boundaries':
            return GenomicRanges.FDR(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        else:
            raise Exception('Coefficient not understood: {}'.format(coef))

    def count_shared(self, other, ident=1):
        """
        Return share of shared TADs regarding the first range with given identity.
        Asymmetrical!
        """
        if not self.data.shape[0] * other.data.shape[0]:
            return 0
        intersected = GenomicRanges.find_intersect(self.data, other.data, ident=ident)
        amount_shared = intersected.shape[0]
        shared_1 = np.unique(GenomicRanges.TAD_bins(intersected[:, 0, :])).shape[0]
        # TODO: consider the formula.
        return amount_shared / (self.length + other.length - shared_1)
    # TODO: tests
    # TODO: bin-boundariwise mode
    def find_closest(self, other, mode='boundariwise'):
        """
        Find closest feature in other for each feature
        in self. Two modes are available:
        binwise - distances for bins;
        boundariwise - distances from boundaries in self to
        boundaries in other.
        Return arrays of indexes.
        """
        if not self.data.shape[0] * other.data.shape[0]:
            return None
        v1 = np.copy(self.data)
        v2 = np.copy(other.data)
        if mode == 'boundariwise':
            ind_end = [np.unravel_index(*[func(np.abs(v2 - i)) for func in (np.argmin, np.shape)])  for i in v1[:, 1]]
            ind_start = [np.unravel_index(*[func(np.abs(v2 - i)) for func in (np.argmin, np.shape)])  for i in v1[:, 0]]
            return np.array([ind_start, ind_end], dtype=int)

        elif mode == 'binwise':
            left_closest = [k if k >= 0 else 0 for k in [sum(v2[:, 1] <= i[0]) - 1 for i in v1]]
            right_closest = [k if k >= 0 else v2.shape[0] - 1 for k in [sum(v2[:, 0] >= i[1]) - 1 for i in v1]]
            indexes = np.zeros(v1.shape, dtype=int)
            for coord1, coords2 in enumerate(zip(left_closest, right_closest)):
                left_coord, right_coord = coords2
                diff = right_coord - left_coord
                if diff <= 1:
                    if v1[coord1, 0] == v2[left_coord, 0]:
                        indexes[coord1, :] = left_coord, 2
                    elif v1[coord1, 1] == v2[right_coord, 1]:
                        indexes[coord1, :] = right_coord, 2
                    elif v1[coord1, 0] < v2[left_coord, 1]:
                        indexes[coord1, :] = left_coord, 2
                    elif v1[coord1, 1] > v2[right_coord, 0]:
                        indexes[coord1, :] = right_coord, 2
                    elif v1[coord1, 0] - v2[left_coord, 1] < v1[coord1, 1] - v2[right_coord, 0]:
                        indexes[coord1, :] = left_coord, 1
                    else:
                        indexes[coord1, :] = right_coord, 0
                else:
                    indexes[coord1, :] = left_coord + 1, 2
            return indexes
        elif mode == 'bin-boundariwise':
            raise Exception('Mode not implemented: %s' % mode)
        else:
            raise Exception("The mode isn't understood: {}".format(mode))

    # TODO: tests
    def dist_closest(self, other, mode='boundariwise'):
        """
        Find distances to closest features in other for each
        feature in self.
        """
        indexes = self.find_closest(other, mode=mode)
        if indexes is None:
            return None
        if mode == 'boundariwise':
            ind_start, ind_end = indexes
            dist_start = np.array([other.data[i[0], i[1]] for i in ind_start]) - self.data[:, 0]
            dist_end = np.array([other.data[i[0], i[1]] for i in ind_end]) - self.data[:, 1]
            return np.vstack((dist_start, dist_end)).T
        elif mode == 'binwise':
            indexes = self.find_closest(other, mode=mode)
            distances = np.zeros(indexes.shape[0], dtype=int)
            mask_zeros = indexes[:, 1] < 2
            closest = np.array([other.data[i[0], i[1]] for i in indexes[mask_zeros]], dtype=int)
            boundaries = np.array([self.data[i, 0] if indexes[i, 1] == 1 else self.data[i, 1] for i in range(self.data.shape[0])], dtype=int)
            distances[mask_zeros] = boundaries[mask_zeros] - closest
            return distances
        elif mode == 'bin-boundariwise':
            raise Exception('Mode not implemented: %s' % mode)
        else:
            raise Exception("The mode isn't understood: {}".format(mode))

# TODO: test this.
def load_BED(filename):
    """
    Return dictionary of GenomicRanges with chromosomes 
    as keys from BED-like file. Can load 2-column file,
    3- and 6-column BED files.
    """
    buff = np.loadtxt(filename, dtype=object, ndmin=2)
    if buff is None:
        return {"chr1": GenomicRanges(buff)}
    elif buff.shape[1] not in (2, 3, 6):
        raise Exception("Given file is not BED-like.")
    elif buff.shape[1] in (0, 2):
        return {"chr1": GenomicRanges(buff)}
    else:
        chrms = np.unique(buff[:, 0])
        indices = [np.searchsorted(buff[:, 0], chrm) for chrm in chrms]
        if buff.shape[1] == 3:
            return {chrm: GenomicRanges(arr) for chrm, arr in zip(chrms, np.vsplit(buff[:, 1:], indices[1:]))}
        if buff.shape[1] == 6:
            return {chrm: GenomicRanges(arr) for chrm, arr in zip(chrms, np.vsplit(buff[:, [1, 2, 4]], indices[1:]))}
