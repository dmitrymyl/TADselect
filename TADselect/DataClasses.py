"""
Classes for data handling like TAD segmentation
or genomic tracks.
"""

from os import stat as file_stat
from .utils import *
from .logger import TADselect_logger
import numpy as np
import numpy.core.defchararray as npchar
import scipy.stats


class GenomicRanges(object):
    """
    Basic class for any genomic ranges.
    """

    def __init__(self, arr_object, data_type='simulation', scale=1, **kwargs):
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
            buff = buff[buff[:, 0].argsort(), :]
            if scale != 1:
                self.data = buff / scale
            else:
                self.data = buff
            self.coverage = np.full_like(self.data[:, 0], 1, dtype=int)
        elif buff.shape[1] == 3:
            buff = buff[buff[:, 0].argsort(), :]
            if scale != 1:
                self.data = buff[:, 0:2] / scale
            else:
                self.data = buff[:, 0:2]
            self.coverage = buff[:, 2]
        elif buff is None or buff.shape[1] == 0:  # No segments in a segmentation, TODO: @dmyl check
            self.data = np.zeros((1, 2))
            self.coverage = np.zeros(1)
        else:
            raise Exception("Inappropriate shape of arr_object: {}.".format(buff.shape))

        self.length = self.data.shape[0]
        self.data_type = data_type
        self.sizes = self.data[:, 1] - self.data[:, 0]
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
        Returns TADs as objects from their coordinates.
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

        v1 = np.array(arr1.copy(), dtype=float)
        v2 = np.array(arr2.copy(), dtype=float)

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
        start_dist = np.array([min(v2[:, 0] - i, key=abs) for i in v1[:, 0]], dtype=float)
        end_dist = np.array([min(v2[:, 1] - i, key=abs) for i in v1[:, 1]], dtype=float)
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
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return GenomicRanges.jaccard_index(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'JI boundaries':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return GenomicRanges.jaccard_index(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'OC TADs':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return GenomicRanges.overlap_coef(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'OC boundaries':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return GenomicRanges.overlap_coef(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'TPR TADs':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return GenomicRanges.TPR(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'FDR TADs':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 1
            return GenomicRanges.FDR(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'PPV TADs':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return 1 - GenomicRanges.FDR(*map(GenomicRanges.TAD_bins, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'TPR boundaries':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return GenomicRanges.TPR(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'FDR boundaries':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 1
            return GenomicRanges.FDR(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

        elif coef == 'PPV boundaries':
            if (self.length == 1 and self.data[0, 0] + self.data[0, 1] == 0) or (other.length == 1 and other.data[0, 0] + other.data[0, 1] == 0):
                return 0
            return 1 - GenomicRanges.FDR(*map(GenomicRanges.TAD_boundaries, GenomicRanges.make_offset(self.data, other.data, offset=offset)))

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
    def find_closest(self, other, mode='boundariwise'):
        """
        Find closest feature in other for each feature
        in self. Return array of indexes with additional information.
        *Boundariwise mode*
        Search closest boundaries in other for start and end boundaries
        in self. Return indexes in manner [index_start, index_end],
        where index_start is for start boundaries in self and so.
        index_* is composed as [[row1, col1], [row2, col2], ...],
        where colN can be only 0 (start of feature in other)
        or 1 (end of feature in other).
        *Bin-boundariwise mode*
        Treat features as boundariwise mode except boundary in self
        overlaps feature in other. Then the distance is zero. The output
        is the same as in boundariwise mode but colN can be 2 indicating
        overlap of boundary and feature.
        *Binwise mode*
        Find closest bins in other for each bin in self. The output is
        [[row1, col1], [row2, col2], ...], where colN can be zero (means
        feature in self is leftmost regarding feature in other),
        one (feature in self is rightmost regarding feature in other)
        or two (features overlap).
        """
        if not self.data.shape[0] * other.data.shape[0]:
            return None
        v1 = np.copy(self.data)
        v2 = np.copy(other.data)
        if mode in ('boundariwise', 'bin-boundariwise'):
            # find indices of closest boundaries in v2 regarding v1 end boundaries
            index_end = [np.unravel_index(*[func(np.abs(v2.copy() - i)) for func in (np.argmin, np.shape)])  for i in v1[:, 1]]
            # find indices of closest boundaries in v2 regarding v1 start boundaries
            index_start = [np.unravel_index(*[func(np.abs(v2.copy() - i)) for func in (np.argmin, np.shape)])  for i in v1[:, 0]]
            if mode == 'bin-boundariwise':
                index_end = [(feature[0], 2) if v2[feature[0], 0] <= v1[i, 1] <= v2[feature[0], 1] else feature for i, feature in enumerate(index_end)]
                index_start = [(feature[0], 2) if v2[feature[0], 0] <= v1[i, 0] <= v2[feature[0], 1] else feature for i, feature in enumerate(index_start)]
            return np.array([index_start, index_end], dtype=int)  # return both start and end indices

        elif mode == 'binwise':
            # find indices of the closest feature in v2 that precedes feature in v1
            left_closest = [k if k >= 0 else 0 for k in [sum(v2[:, 1] <= i[0]) - 1 for i in v1]]
            # find indices of the closest feature in v2 that follows feature in v1
            right_closest = [k if v2.shape[0] > k >= 0 else v2.shape[0] - 1 for k in [v2.shape[0] - sum(v2[:, 0] >= i[1]) for i in v1]]
            indexes = np.zeros(v1.shape, dtype=int)
            for coord1, coords2 in enumerate(zip(left_closest, right_closest)):
                left_coord, right_coord = coords2  # closest left and right feature in v2 regarding v1
                diff = right_coord - left_coord  # difference in indices of two closest features in v2 regarding v1
                intersecting = lambda arr1, arr2: (arr2[0] <= arr1[0] <= arr2[1]
                                                   or arr2[0] <= arr1[1] <= arr2[1]
                                                   or arr1[0] <= arr2[0] <= arr1[1]
                                                   or arr1[0] <= arr2[1] <= arr1[1])
                if diff == 1:  # 1 means left_coord is followed by right_coord, 0 means it is the same feature in v2
                    if intersecting(v1[coord1], v2[left_coord]):
                        indexes[coord1, :] = left_coord, 2
                    elif intersecting(v1[coord1], v2[right_coord]):
                        indexes[coord1, :] = right_coord, 2
                    elif v1[coord1, 0] - v2[left_coord, 1] <= v2[right_coord, 0] - v1[coord1, 1]:
                        # dist to left v2 feature is equal to or less than dist to right v2 feature
                        indexes[coord1, :] = left_coord, 1  # the closest boundary in v2 to v1 feature
                    else:
                        # dist to left v2 feature is greater than dist to right v2 feature
                        indexes[coord1, :] = right_coord, 0  # the closest boundary in v2 to v1 feature
                elif diff == 0:  # 0 means left and right are the same in v2 i.e. v1 feature and v2 feature are either both first or both last
                    if intersecting(v1[coord1], v2[left_coord]):
                        indexes[coord1, :] = left_coord, 2
                    elif v1[coord1, 0] - v2[left_coord, 1] > 0:
                        indexes[coord1, :] = left_coord, 1
                    else:
                        indexes[coord1, :] = right_coord, 0
                else:
                    # there are at least one feature in v2 between left and right closest v2 features
                    indexes[coord1, :] = left_coord + 1, 2
            return indexes  # returns closest bins in v2 with code: 1 -- left, 0 -- right, 2 -- overlap 
        else:
            raise Exception("The mode isn't understood: {}".format(mode))

    def dist_closest(self, other, mode='boundariwise'):
        """
        For each feature in self find distances to closest feature
        in other.
        *Boundariwise mode*
        Count distances from start and end boundaries of self
        to any closest boundaries of other. Returns 2-dim array
        [[dist_from_start1, dist_from_end1], [dist_from_start2, dist_from_end2], ...]
        *Binwise mode*
        Count distances between bins. In case some bin in self
        overlaps bin in other, ttakes he distance between them as zero.
        Returns 1-dim array [dist_from_bin1, dist_from_bin2, ...]
        *Bin-boundariwise mode*
        Count distances from start and end boundaries of self
        to any closest boundary of other. In case boundary in self
        overlaps bin in other, takes the distance as zero.
        Returns 2-dim array as in boundariwise mode.
        """
        indexes = self.find_closest(other, mode=mode)
        if indexes is None:
            return None
        v1 = np.copy(self.data)
        v2 = np.copy(other.data)
        if mode in ('boundariwise', 'bin-boundariwise'):
            ind_start, ind_end = indexes
            mask_start, mask_end = ind_start[:, 1] < 2, ind_end[:, 1] < 2
            dist_start, dist_end = np.zeros(ind_start.shape[0], dtype=int), np.zeros(ind_start.shape[0], dtype=int)
            dist_start[mask_start] = np.array([v2[i[0], i[1]] for i in ind_start[mask_start]]) - v1[:, 0][mask_start]
            dist_end[mask_end] = np.array([v2[i[0], i[1]] for i in ind_end[mask_end]]) - v1[:, 1][mask_end]
            return np.vstack((dist_start, dist_end)).T
        elif mode == 'binwise':
            distances = np.zeros(indexes.shape[0], dtype=int)
            mask_zeros = indexes[:, 1] < 2
            closest = np.array([v2[i[0], i[1]] for i in indexes[mask_zeros]], dtype=int)
            boundaries = np.array([v1[i, 0] if indexes[i, 1] == 1 else v1[i, 1] for i in range(v1.shape[0])], dtype=int)
            distances[mask_zeros] = closest - boundaries[mask_zeros]
            return distances
        else:
            raise Exception("The mode isn't understood: {}".format(mode))

    def dot_in_triangle(self, dot):
        """
        Checks whether the dot (array-like of length 2)
        is located in triangle upper or lower than some
        range in self (e.g. in TAD). If so. returns a row
        index of found range else returns None.
        """
        dist_x = dot[0] - self.data[:, 0]
        mask = dist_x > 0
        bin_index = sum(mask) - 1
        bin_x = self.data[bin_index, 0]
        bin_y = self.data[bin_index, 1]
        if bin_x <= dot[0] <= bin_y and bin_x <= dot[1] <= bin_y:
            return bin_index
        else:
            return None

    def save(self, filename, filetype='BED', chrm='chr1'):
        """
        Saves data and counts in a file type specified by type:
            * BED -- makes a 6-col BED file;
            * TADs -- makes 2-col txt file.
        """
        if filetype == 'BED':
            chrms = np.array([chrm for _ in range(self.length)]).reshape(self.length, 1)
            dots = np.array(['.' for _ in range(self.length)]).reshape(self.length, 1)
            buff = np.append(chrms, self.data, axis=1)
            buff = np.append(buff, dots, axis=1)
            buff = np.append(buff, dots, axis=1)
            buff = np.append(buff, self.coverage, axis=1)
            buff = np.append(buff, dots, axis=1)
            np.savetxt(filename, buff, fmt='%d', delimiter='\t')
        elif filetype == 'TADs':
            np.savetxt(filename, self.data, fmt='%d', delimiter='\t')


def load_BED(filename, scale=1, chrm=None):
    """
    Return dictionary of GenomicRanges with chromosomes
    as keys from BED-like file. Can load:
    2-column file (start, end),
    3-column BED file (chr, start, end),
    6-column BED file (chr, start, end, name, score, strand).
    """
    buff = np.loadtxt(filename, dtype=object, ndmin=2)
    if buff is None:
        return {"chr1": GenomicRanges(buff)}
    elif buff.shape[1] not in (2, 3, 6) and buff.shape[1] < 6:
        raise Exception("Given file is not BED-like.")
    elif buff.shape[1] in (0, 2):
        if chrm is None:
            chrm = 'chr1'
        return {chrm: GenomicRanges(buff, scale=scale)}
    else:
        chrms = np.unique(buff[:, 0])
        indices = [np.searchsorted(buff[:, 0], chrm) for chrm in chrms]
        if buff.shape[1] == 3:
            return {chrm: GenomicRanges(arr, scale=scale) for chrm, arr in zip(chrms, np.vsplit(buff[:, 1:], indices[1:]))}
        if buff.shape[1] >= 6:
            return {chrm: GenomicRanges(arr, scale=scale) for chrm, arr in zip(chrms, np.vsplit(buff[:, [1, 2, 4]], indices[1:]))}