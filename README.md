### TADcalling

Here we introduce a library for testing various TAD callers.

TODO list: 
- Add setup.py and installation as pip -e ./
- Grid parameters search -- automatic call:
```python 
lc.load_segmentation( lc.call(0.9) )
```
- Add cool to h5 conversion (HiCExplorer), txt to cool (from simulated maps to cool), cool to hic (juicer tools arrowhead). 
- Add more Callers (Next: Armatus Cpp @dmyl, HiCExplorer @agal, TADtool @dmyl)

#### Installation

Ordinary mode:

```bash
git clone https://github.com/agalitsyna/TADcalling.git
cd TADcalling
pip install ./
```

Developer mode for pip:
```bash
pip install -e ./
```
