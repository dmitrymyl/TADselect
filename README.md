# TADselect
Here we introduce a library for testing various TAD callers. The tool is capable of selecting the best parameter for certain TAD callers, plotting necessary data and benchmarking.

## Installation
### Ordinary mode:
```bash
git clone https://github.com/dmitrymyl/TADselect.git
cd TADselect
pip install ./
```
### Developer mode for pip:
```bash
pip install -e ./
```
### Conda developer mode:
```
cd TADselect
conda develop ./
```
## Currently implemented TAD callers
* Armatus standalone
* Lavaburst package:
    * Armatus
    * Modularity
* TADtools package:
    * Insulation score
    * Directionality index
* HiCExplorer
* HiCseg
* Arrowhead
* TADtree
* TADbit

## Strategies for parameter selection
Several TAD callers require selection of master parameter (so called 'gamma'), which affects resulting segmentation. We propose a number of strategies for picking out the most adequate parameter.

### Converegence between technical replica
Two Hi-C matrices derived from the same chromosome of the same cell line but from different replica are supposed to have identical TAD segmentations. The optimisation of parameter is based on maximisation of Jaccard index (JI) between segmentations taken from two technical replicates with the same master parameter.

### Colocalisation with genome features
TAD borders were shown to colocalise with CTCF peaks in Human and histone mark transitions in Drosophila. The module in specific regime is able to approximate TAD segmentation to given genomic track by minimising average distance between TAD borders and closest genomic features.

### Recovery of simulated segmentation
It is substantial to test TAD callers on simulated data. The master parameter is selected by maximisation of TPR and PPV of given TAD segmentation.

### Background function
Even if listed above strategies perform well, the callers might reproduce inadequate segmentations with few huge or many small TADs. To control this effect the background function was introduced. It takes the assumption that the most informative segmentation for each given Hi-C matrix tells mean size of TADs between 2 and 1000 bins regardless its resolution.

## Input data

## Description of available classes

### GenomicRanges
The class to store genomic ranges with coverage. The input is available from numpy 2D-array or from .bed file via load_BED() function. The class implements bedtools-like methods: finding closest features in other track for given track, counting distances to them; finding intersecting ranges, counting shared ranges with given offset; calculating JI, OC, TPR, FDR, PPV for ranges themselves and their boundaries.

### CallerClasses
Main subset of classes containing python interface to implemented TAD callers. Typical CallerClasses class contain two methods: call() and \_call_single(). The first method takes range of parameter values for TAD calling, the second perfoms a single TAD calling event. The obtained segmentation is returned as GenomicRanges instance with coverage 1. Segmentations are stored in dictionaries with parameter values as keys for each label key. A caller class also contain \_benchmark_list and \_benchmark_df collections for ctime, utime, walltime and RAM memory spent on each \_call_single() event.

### InteractionMatrix
This class is an interface to a Hi-C matrix. The main used format is .cool; .txt, .txt.gz, numpy matrix are also available as well as cross-convertation. The class also implements several common transformations for Hi-C matrices.

### Experiment
The class is a central hub for all other classes that performs parameter selection based on listed above strategies with iterative approach. The method converges when the step of newly produced parameter range is lower than given number of given number of iterations has been surpassed.

## Examples and explanations
See relative jupyter notebook for usage examples and technical details in [examples folder](https://github.com/dmitrymyl/TADselect/tree/master/examples)
