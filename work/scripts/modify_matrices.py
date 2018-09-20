from sys import argv
import TADcalling


cool_file = argv[1]
mtx = TADcalling.InteractionMatrix(argv[1], input_type='cool', balance=False, ch='Sim')
file_prefix = ".".join(cool_file.split(".")[:-1])
mtx = mtx.fill_nans(0)
removed = mtx.remove_diagonal(1, 0)
removed._write_mtx(file_prefix + '.nodiag.cool', 1, None, 'Sim', 1 * removed._mtx.shape[0])
filtered = mtx.filter_extreme()
filtered._write_mtx(file_prefix + '.filtered.cool', 1, None, 'Sim', 1 * filtered._mtx.shape[0])
log_transformed = mtx.filter_extreme().log_transform(2).subtract_min().convert_to_int()
log_transformed._write_mtx(file_prefix + '.log.cool', 1, None, 'Sim', 1 * log_transformed._mtx.shape[0])
