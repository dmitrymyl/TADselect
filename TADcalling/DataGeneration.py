from os import stat as file_stat
from .utils import *
from .logger import TADcalling_logger
from .InteractionMatrix import *
from .DataClasses import GenomicRanges
import numpy as np
import scipy.stats


def generate_TADs(mtx_size, min_tadsize, max_tadsize, min_intertadsize, max_intertadsize):
    """
    Return GenomicRanges instance of simulated TAD segmentation.
    The TAD and interTAD sizes are sample one by one from uniform
    distribution with paramenters specific to TAD (min_tadsize, max_tadsize)
    and to interTAD (min_intertadsize, max_intertadsize) regions until
    the sum of sizes hasn't overcome mtx_size and fit generated sizes to mtx_size.
    """
    sizes = list()
    while sum(sizes) < mtx_size:
        sizes.append(scipy.stats.randint.rvs(min_intertadsize, max_intertadsize + 1))
        sizes.append(scipy.stats.randint.rvs(min_tadsize, max_tadsize + 1))
    while sum(sizes) >= mtx_size:
        sizes.pop()
    sizes.append(mtx_size - sum(sizes) - 1)
    sizes = list(filter(lambda x: x != 0, sizes))
    range_buffer = list()
    mtx_index = 0
    size_index = 0
    while mtx_index <= mtx_size and size_index < len(sizes):
        if sizes[size_index] <= max_intertadsize:
            mtx_index += sizes[size_index]
        else:
            range_buffer.append([mtx_index, mtx_index + sizes[size_index]])
            mtx_index += sizes[size_index]
        size_index += 1
    return GenomicRanges(range_buffer)


def generate_matrix(mtx_size, tads, kdiag, kt, kd, krandom, knoise, c, max_tadsize, **kwargs):
    """
    Returns hic_matrix of mtx_size * mtx_size size, simulated from given TAD
    segmentation in tads as GenomicRanges instance.
    """
    interactions = kwargs.get('interactions', int(mtx_size ** 2 / 100000 * 2))
    pseudo = kwargs.get('pseudo', 1)
    noise = kwargs.get('noise', 0.0)
    dispersion = kwargs.get('dispersion', 0.01)
    mu_mtx = np.zeros(shape=(mtx_size, mtx_size))
    # add contacts based on pixel localisation
    for x in range(mtx_size):
        mu_mtx[x, x] += kdiag
    for i in range(tads.length):
        bgn = tads.data[i, 0]
        end = tads.data[i, 1]
        for x in range(bgn, end):
            for y in range(bgn, x):
                mu_mtx[x, y] += kt * (x - y + pseudo) ** c

    print('contact assignment is done')

    # add noise
    for _ in range(int(mtx_size * noise)):
        random_x = np.random.choice(range(4253))
        dots = np.array(range(random_x))
        prob_dots = krandom * (random_x - dots + pseudo) ** c
        prob_dots /= sum(prob_dots)
        random_y = np.random.choice(range(random_x), p=prob_dots)
        mu_mtx[random_x, random_y] += knoise

    print('noise added')

    # add interactions
    for _ in range(interactions):
        interaction_x = np.random.choice(range(4253))
        if interaction_x != 0:
            interaction_y = scipy.stats.randint.rvs(max(0, int(interaction_x - max_tadsize * 4 / 3)), interaction_x)
        else:
            interaction_y = 0
        mu_mtx[interaction_x, interaction_y] += kd * (interaction_x - interaction_y + pseudo) ** c

    print('interactions added')

    # transform mus into raw counts
    hic_mtx = np.zeros(shape=(mtx_size, mtx_size), dtype=int)
    non_zero_mask = mu_mtx != 0
    for indices in np.argwhere(non_zero_mask):
        x = indices[0]
        y = indices[1]
        mu = mu_mtx[x, y]
        r = 1 / dispersion
        prob = r / (r + mu)
        hic_mtx[x, y] = scipy.stats.nbinom.rvs(r, prob)

    print('matrix is ready')

    return hic_mtx + hic_mtx.T - np.diag(hic_mtx.diagonal())


def simulate_HiC(mtx_size, min_tadsize, max_tadsize, min_intertadsize,
                 max_intertadsize, tads_filename, mtx_filename, **kwargs):
    """
    A hub function for generate_TADs() and generate_matrix() that allows
    simultaneous simulation of both TADs and Hi-C matrix with writing them
    to files.
    """
    kdiag = kwargs.get('kdiag', 35)
    kt = kwargs.get('kt', 28)
    kd = kwargs.get('kd', 2 * kt)
    krandom = kwargs.get('krandom', 1)
    knoise = kwargs.get('knoise', 2)
    c = kwargs.get('c', -0.69)
    interactions = kwargs.get('interactions', int(mtx_size ** 2 / 100000 * 2))
    pseudo = kwargs.get('pseudo', 1)
    noise = kwargs.get('noise', 0)
    dispersion = kwargs.get('dispersion', 0.01)

    segmentation = generate_TADs(mtx_size, min_tadsize, max_tadsize, min_intertadsize, max_intertadsize)
    segmentation.save(tads_filename, filetype='TADs')

    hic_mtx = generate_matrix(mtx_size, segmentation, kdiag, kt, kd, krandom,
                              knoise, c, max_tadsize, pseudo=pseudo, noise=noise,
                              interactions=interactions, dispersion=dispersion)
    mtxObj = InteractionMatrix(hic_mtx)
    mtxObj._write_mtx(mtx_filename, 1, None, 'Sim', 1 * mtxObj._mtx.shape[0])
    # np.savetxt(mtx_filename, hic_mtx, fmt="%d", delimiter='\t')