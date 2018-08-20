
# Universal imports
from .utils import *
from .logger import TADcalling_logger
from .DataClasses import GenomicRanges, load_BED
from copy import deepcopy
from functools import partial
import numpy as np
import pandas as pd
import cooler

# Class-specific imports
import tadtool.tad
import lavaburst
from .templates import hicseg_template

ACCEPTED_FORMATS = ['cool', 'txt', 'txt.gz', 'sparse', 'mr_sparse', 'hic', 'h5']


def lazyProcessing(func):

    def _wrapper(self, *args, **kwargs):

        push = kwargs.get('push', False)

        if push:
            mtx, modification, mod_number = func(self, *args, **kwargs)

            if self._nmods == 0:
                self_copy = InteractionMatrix(mtx,
                                              input_type='mtx',
                                              transformations=self._transformations + [modification])
                self_copy._nmods += 1
                return self_copy

            else:
                self._mtx = mtx
                self._transformations.append(modification)
                self._nmods += 1
                return self

        else:
            if len(self._operations_list) == 0:
                self_copy = InteractionMatrix(self._mtx,
                                              input_type='mtx',
                                              transformations=self._transformations)
                kwargs.update({'push': True})
                self_copy._operations_list.append((lazyProcessing(func), args, kwargs))
                return self_copy

            else:
                kwargs.update({'push': True})
                self._operations_list.append((lazyProcessing(func), args, kwargs))
                return self

    return _wrapper


class InteractionMatrix(object):

    def __init__(self, input_mtx, input_type='mtx', read=True, transformations='raw'):
        """

        :param input: input dataset with raw reads counts
        :param input_type: 'mtx'
        """

        if input_type == 'mtx':
            self._mtx = input_mtx.copy()
            self._prepared_formats = [input_type]
            self._transformations = [transformations]
            self._nmods = 0
            self._operations_list = []

        else:
            TADcalling_logger.warning("Input type {} currently not implemented. Skipping reading.".format(input_type))

    def _read_mtx(self, input_mtx, file_format):
        pass

    def _write_mtx(self, output):
        pass

    def convert_without_reading(self, input_filename, output_filename,
                                input_format=None,
                                output_format=None,
                                **kwargs):
        """
        Convert files/matrix without reading them into InteractionMatrix._mtx
        :param input_format:
        :param output_format:
        :param input:
        :param output:
        :return:
        """

        if type(input_filename) == np.ndarray:
            input_format = 'mtx'
            mtx = input_filename

        if input_format is None:
            input_format = input_filename.split('.')
        if output_format is None:
            output_format = output_filename.split('.')

        TADcalling_logger.info("Converting %s -> %s: from %s to %s",
                               input_format, output_format, input_filename, output_filename)

        chromosome = kwargs.get('chr', 'chr2L')
        resolution = kwargs.get('res', 20000)
        remove_intermediary_files = kwargs.get('remove_intermediary_files', True)
        balance = kwargs.get('balance', False)

        if input_format == 'cool' and 'txt' in output_format:
            output_prefix = '.'.join(input_filename.split('.')[:-1])
            output_filename = output_prefix + '.{}.txt'.format(chromosome)

            c = cooler.Cooler(input_filename)
            mtx = c.matrix(balance=balance, as_pixels=False).fetch(chromosome, chromosome)

            np.savetxt(output_filename, mtx, delimiter='\t')

            if 'gz' in output_format:
                command = 'gzip {}'.format(output_filename)
                run_command(command)
                output_filename += '.gz'

            return output_filename

        elif input_format == 'cool' and output_format == 'sparse':

            output_prefix = '.'.join(input_filename.split('.')[:-1])
            output_filename = output_prefix + '.{}.sparse.txt'.format(chromosome)

            c = cooler.Cooler(input_filename)
            mtx_df = c.matrix(balance=balance, as_pixels=True, join=True,
                              ignore_index=False).fetch(chromosome, chromosome)
            if balance:
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
            # TODO @agal implement this option
            # https://github.com/hms-dbmi/higlass/issues/100#issuecomment-302183312
            TADcalling_logger.error('Option currently not implemented.')

        elif input_format == 'cool' and output_format == 'h5':
            output_prefix = '.'.join(input_filename.split('.')[:-1])
            output_filename = output_prefix + '.h5'
            command = "hicExport --inFile {} --outFileName {} --inputFormat cool --outputFormat h5" \
                .format(input_filename, output_filename)
            run_command(command)
            return output_filename

        elif input_format == 'cool' and output_format == 'hic':

            binary_path = kwargs.get('binary_path', 'java')
            juicer_path = kwargs.get('juicer_path', './juicer_tools.1.8.9_jcuda.0.8.jar')
            genome = kwargs.get('genome', 'dm3')

            output_prefix = '.'.join(input_filename.split('.')[:-1])
            outfile_hic = "{}.{}.hic".format(output_prefix, chromosome)

            outfile_txt = outfile_hic + '.txt'
            outfile_tmp = outfile_hic + '.tmp'

            with open(outfile_tmp, 'w'):
                pass

            outfile_tmp = self.convert_without_reading(input_filename=input_filename,
                                                       output_filename=outfile_tmp,
                                                       input_format='cool',
                                                       output_format='sparse')

            command1 = "awk '{{print 0, $1, $2, 0, 0, $4, $5, 1, $7}}' {} > {}".format(outfile_tmp, outfile_txt)
            command2 = "gzip -f {}".format(outfile_txt)
            command3 = "{} -Xmx2g -jar {} pre -r {} -c {} {}.gz {} {}".format(binary_path,
                                                                              juicer_path,
                                                                              resolution,
                                                                              chromosome,
                                                                              outfile_txt,
                                                                              outfile_hic,
                                                                              genome)
            run_command(command1)
            run_command(command2)
            run_command(command3)

            if remove_intermediary_files:
                os.remove(outfile_txt + '.gz')

        elif input_format == 'mtx':

            if 'txt' in output_format:
                np.savetxt(output_filename, mtx, delimiter='\t')

                if 'gz' in output_format:
                    command = 'gzip {}'.format(output_filename)
                    run_command(command)
                    output_filename += '.gz'

            elif output_format == 'cool':
                TADcalling_logger.error('Option currently not implemented.')

        return output_filename

    def as_array(self):

        self.compute()

        return self._mtx

    def compute(self):

        if len(self._operations_list) > 0:
            new_self = self
            for func, args, kwargs in self._operations_list:
                new_self = func(new_self, *args, **kwargs)

            self._mtx = new_self._mtx.copy()
            self._transformations = new_self._transformations
            self._nmods = 0
            self._operations_list = []

        return self

    @lazyProcessing
    def log_transform(self, base=10, **kwargs):
        """
        Log transformation of input matrix.

        :param base: log base passed to math.log
        :return: transformed matrix
        """

        TADcalling_logger.info("Log-transform of input mtx, base: %d", base)
        mtx = np.log(self._mtx) / np.log(base)

        return mtx, 'log({})'.format(base), self._nmods + 1

    @lazyProcessing
    def subtract_min(self, **kwargs):

        TADcalling_logger.info("Subtracting minimum from mtx")

        min_value = np.nanmin(self._mtx)
        mtx = self._mtx - min_value

        return mtx, 'minSubtracted', self._nmods + 1

    @lazyProcessing
    def filter_extreme(self, min_percentile=1, max_percentile=99, **kwargs):
        """
        Percentile filtering of matrix.

        :param min_percentile:
        :param max_percentile:
        :return:
        """

        mtx = self._mtx.copy()

        mn = np.nanpercentile(mtx[mtx > 0], min_percentile)
        mx = np.nanpercentile(mtx[mtx > 0], max_percentile)

        TADcalling_logger.info("Percentile filtering of input mtx, min percentile: %f (%.3f), max percentile: %f (%.3f)",
                               min_percentile, mn, max_percentile, mx)

        mtx[mtx <= mn] = mn
        mtx[mtx >= mx] = mx

        return mtx, 'filterExtreme({},{})'.format(min_percentile, max_percentile), self._nmods + 1

    @lazyProcessing
    def remove_diagonal(self, ndiag=1, value=0, **kwargs):
        """
        Diagonals removal

        :param ndiag: number of diagonals to remove
        :param value: value to fill diagonal with
        :param kwargs:
        :return:
        """

        TADcalling_logger.info("Removing %d first diagonals", ndiag)

        mtx = self._mtx.copy()
        length = len(mtx)

        for i in range(ndiag):
            np.fill_diagonal(mtx[i:, :length - i], value)
            np.fill_diagonal(mtx[:length - i, i:], value)

        return mtx, 'removeDiagonal({})'.format(ndiag), self._nmods + 1

    @lazyProcessing
    def convert_to_int(self, **kwargs):
        """
        TODO @agal check validity and possible errors due to precision
        Converts float matrix to integer after setting the smallest value to 1

        :return:
        """
        TADcalling_logger.info("Converting matrix to integer")

        mtx = self._mtx.copy()

        min_value = np.nanmin(mtx[mtx > 0])
        mtx = mtx * np.power(10, -np.log10(min_value))
        mtx = mtx.astype(int)

        return mtx, 'convert2int', self._nmods + 1

    @lazyProcessing
    def fill_nans(self, value=0, **kwargs):
        """
        Fill nans with specified values.

        :param value: 0
        :param kwargs:
        :return:
        """

        TADcalling_logger.info("NaNs filling with %.2f", value)

        mtx = self._mtx.copy()
        mtx[np.isnan(mtx)] = value

        return mtx, 'fillNans', self._nmods + 1

    @lazyProcessing
    def add_pseudocount(self, value='default', **kwargs):
        """
        Add pseudocounts to the matrix

        :param value: 'default' or any float number.
            'float' means the counts will be the min values larger than 0 in the matrix
        :return:
        """

        mtx = self._mtx.copy()

        if value == 'default':
            value = np.nanmin(mtx[mtx > 0])

        mtx += value

        return mtx, 'addPseudocount', self._nmods + 1

    @lazyProcessing
    def fill_bins(self, bins='zeros', value=0, name='', **kwargs):
        """
        Fill bins (rows & cols) of matrix with specified values

        :param bins: array of bool or numpy array index (bone dimension)
        :param value:
        :param name: string name to put into description, optional
        :param kwargs:
        :return:
        """

        mtx = self._mtx.copy()

        if type(bins) == str:
            name = bins + name
            if bins == 'zeros':
                idx = np.sum(mtx, axis=1) == 0
            else:
                TADcalling_logger.error("Fill bins mode {} not implemented yet".format(bins))
        elif type(bins) == np.ndarray:
            idx = bins
        else:
            TADcalling_logger.error("bins type not recognised: {}".format(type(bins)))

        mtx[idx, :] = value
        mtx[:, idx] = value

        return mtx, 'fillBins({},{})'.format(name, value), self._nmods + 1

    @lazyProcessing
    def fill_mask(self, mask='non-positive', value=0, name='', **kwargs):
        """
        Fill submatrix of mtx (defined by 2D mask) with specified values

        :param mask: array of bool or numpy array index (two dimensions)
        :param value:
        :param name: string name to put into description, optional
        :param kwargs:
        :return:
        """

        mtx = self._mtx.copy()

        if type(mask) == str:
            name = mask + name
            if mask == 'non-positive':
                mask = mtx <= 0
            else:
                TADcalling_logger.error("Fill mask mode {} not implemented yet".format(mask))
        elif type(mask) == np.ndarray:
            pass
        else:
            TADcalling_logger.error("mask type not recognised: {}".format(type(mask)))

        mtx[mask] = value

        return mtx, 'fillMask({})'.format(name), self._nmods + 1

    @lazyProcessing
    def multiply_by(self, value=1, **kwargs):
        """
        Multiply all the values in matrix by constant value

        :param value:
        :param kwargs:
        :return:
        """

        mtx = self._mtx.copy()

        mtx = mtx * value

        return mtx, 'multiplyBy({})'.format(value), self._nmods + 1

    @lazyProcessing
    def add_value(self, value=0, **kwargs):
        """
        Add value to all the elements in the matrix

        :param value:
        :param kwargs:
        :return:
        """

        mtx = self._mtx.copy()

        mtx = mtx + value

        return mtx, 'addValue({})'.format(value), self._nmods + 1

    @lazyProcessing
    def delete_bins(self, bins, name='', **kwargs):
        """
        Remove bins from matrix

        :param bins: indexes of bins to remove or bool numpy array
        :param name: optional name for transformation description
        :param kwargs:
        :return:
        """

        mtx = self._mtx.copy()

        if bins.dtype == np.dtype('bool'):
            idx = np.where(bins)[0]
        else:
            idx = bins

        mtx = np.delete(np.delete(mtx, idx, 1), idx, 0)

        return mtx, 'deleteBins({})'.format(name), self._nmods + 1

    @staticmethod
    def restore_bins(v, bins_remove):
        """
        TODO @agal check on real example

        Restoring coordinates of bins (v) after bins removal.

        :param v:
        :param bins_remove:
        :return:
        """

        v_return = v.copy()

        if bins_remove.dtype == np.dtype('bool'):
            idx = np.where(bins_remove)[0]
        else:
            idx = bins_remove

        for i in bins_remove:
            v_return[v_return >= i] += 1

        return v_return

    def get_expected_matrix(self):  # @agal what is it?

        length = len(self._mtx)
        expected_vect = list(map(lambda x: self._mtx.diagonal(x).sum(), np.arange(length)))

        mtx_ret = np.zeros((length, length))

        for i in range(length):
            value = expected_vect[i]
            np.fill_diagonal(mtx_ret[i:, :length - i], value)
            np.fill_diagonal(mtx_ret[:length - i, i:], value)

        return mtx_ret

    @lazyProcessing
    def observed_over_expected(self, **kwargs):
        """
        Return observed over expected matrix
        :param kwargs:
        :return:
        """

        obs = self._mtx.copy()
        exp = self.get_expected_matrix()

        mtx = obs / exp

        return mtx, 'obsExp', self._nmods + 1


#TODO @agal  add modifications: 'subsample', 'add_noise', 'ic', 'vc'
