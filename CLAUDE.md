# CLAUDE.md - AI Assistant Guide for ShowBees Repository

This document provides comprehensive guidance for AI assistants working with the ShowBees codebase, a beehive sound analysis framework called AuDaCE (Audio Dataset Controlled Environment).

---

## Project Overview

**ShowBees** is a research project focused on beehive sound analysis for detecting beehive health status and stress levels. It presents the AuDaCE framework, which simplifies the management of machine learning datasets based on audio.

### Core Purpose
- Interpret sounds produced by bees to detect health status and stress levels
- Provide a framework for audio dataset management
- Support machine learning experiments with audio data
- Enable reproducible research through checksums and manifest files

### Technology Stack
- **Language**: Python 3.7+
- **Environment**: Jupyter Notebook/JupyterLab (preferably via Anaconda)
- **Key Libraries**:
  - `tensorflow` (1.14.0 in original experiments)
  - `librosa` (0.7.2) - audio processing
  - `numba` (0.48.0) - performance (specific version required due to librosa bug)
  - `soundfile` - audio file handling
  - `pyprojroot` (0.3.0+) - project root finding
  - `checksumdir` - dataset validation
  - `ipywidgets` - notebook UI components
  - `pandas`, `numpy`, `scipy`, `sklearn` - data science stack

---

## Repository Structure

```
ShowBees/
├── .kilroy                    # Project root marker (DO NOT REMOVE)
├── README.md                  # Main documentation
├── audace/                    # Shared Python modules (framework code)
│   ├── __init__.py
│   ├── audiodataset.py       # Core AudioDataset class
│   ├── jupytools.py          # mooltipath, predestination, iprint utilities
│   ├── dblib.py              # SQLite database operations
│   ├── featurizers.py        # Feature extraction
│   ├── providers.py          # Data providers
│   ├── splitters.py          # Train/test splitting
│   ├── transformers.py       # Data transformations
│   ├── augmenters.py         # Data augmentation
│   ├── plotters.py           # Visualization utilities
│   ├── callbacks.py          # Training callbacks
│   ├── metrics.py            # Custom metrics
│   └── WIP/                  # Work in progress (experimental code)
│
├── datasets/                  # Dataset manifests and generated data
│   ├── README.md
│   ├── *.mnf                 # Manifest files (dataset definitions)
│   └── <DATASET_NAME>/       # Generated dataset directories (gitignored)
│       ├── <DATASET_NAME>.db # SQLite database
│       └── samples/          # Audio chunk files (.wav)
│
├── experiments/               # Experiment notebooks and outputs
│   ├── EXP00 - EXPLORATION/
│   ├── EXP01 - BEE-NOBEE CLASSIFIERS/
│   └── EXP02 - QUEEN-NOQUEEN CLASSIFIERS/
│       ├── README.md
│       ├── *.ipynb           # Experiment notebooks
│       └── output/           # Generated outputs (gitignored)
│
├── tutorials/                 # Learning materials
│   ├── kilroy_was_here.py    # Import helper for notebooks
│   ├── 01 - Basics.ipynb
│   ├── 02 - Datasets (Creation).ipynb
│   └── 03 - Datasets (Usage).ipynb
│
├── userlib/                   # User-specific code (not in git)
├── docs/                      # Documentation (Sphinx)
└── snippets/                  # Scratch/exploratory code (gitignored)
```

---

## Key Conventions and Patterns

### 1. The .kilroy File
**CRITICAL**: The `.kilroy` file marks the project root. It contains ASCII art and must NEVER be removed or modified.

**Purpose**:
- Enables `pyprojroot.find_root(pyprojroot.has_file(".kilroy"))` to work
- Allows `mooltipath()` to build absolute paths from project root
- Makes code work regardless of notebook location in directory tree

### 2. Import Pattern for Notebooks
Every notebook that needs to use `audace` modules must start with:

```python
import kilroy_was_here  # Adds project root to sys.path
from audace.jupytools import mooltipath
from audace.audiodataset import AudioDataset
# ... other audace imports
```

The `kilroy_was_here.py` file is typically located in the same directory as the notebook.

### 3. Path Management with mooltipath()
**ALWAYS** use `mooltipath()` for file paths instead of relative paths:

```python
from audace.jupytools import mooltipath

# Good - works from anywhere
dataset_path = mooltipath("datasets", "TUTO")
output_path = mooltipath("experiments", "EXP01", "output")

# Bad - breaks if notebook location changes
dataset_path = "../../datasets/TUTO"
```

**Key features**:
- Returns absolute Path object from project root
- OS-agnostic (handles / and \\ automatically)
- Always use '/' in path strings, even on Windows

### 4. Code Organization Philosophy

**Core Principle**: Minimize code duplication across experiments.

**Rules**:
1. **NO duplicated .py files across experiments** - Use `audace/` for shared code
2. **Notebook code should be**:
   - Parameter setting and configuration
   - High-level orchestration of function calls
   - Results visualization
   - Sequential, no complex logic
3. **Python files in experiment directories** should be avoided unless:
   - Code is truly experiment-specific
   - Example: `proxycodelib.py` (helper for specific experiment)
4. **Make code generic and reusable** - If it could be used by another experiment, put it in `audace/`

### 5. Dataset Management

**Manifest Files (.mnf)**:
```
<sample_rate>     # e.g., 22050
<duration>        # chunk duration in seconds, e.g., 60
<overlap>         # chunk overlap in seconds, e.g., 0
<md5_checksum>    # for validation, or empty for first build
<source_file_1>
<source_file_2>
...
```

**Dataset Creation Workflow**:
1. Datasets are NOT stored in git (too large)
2. Manifest files (.mnf) define dataset parameters
3. Datasets are built from reference datasets (downloaded separately)
4. MD5 checksums ensure reproducibility
5. Each dataset gets a SQLite database + audio chunks directory

**Key Dataset Concepts**:
- **Chunks**: Audio files split into fixed-duration segments with optional overlap
- **Labels**: Binary or categorical labels (e.g., queen/no-queen)
- **Features**: Extracted features (e.g., MFCCs)
- **Attributes**: Additional metadata

### 6. Naming Conventions

**Python Code Style**:
- **Public methods**: `lowerCamelCase` (non-PEP8 by design - author's preference)
- **Private methods**: `_snake_case` (PEP8 compliant)
- **Note**: This is explicitly documented in `audiodataset.py:38-42`

```python
# Public API - lowerCamelCase
dataset.addLabel("queen")
dataset.setFeature("mfcc20", featurizer_func)
dataset.countSamples()

# Private/internal - snake_case
dataset._get_config_from_db()
dataset._slice_one_file()
```

**Experiment Naming**:
- Format: `EXPnn - DESCRIPTION`
- Example: `EXP02 - QUEEN-NOQUEEN CLASSIFIERS`

**Notebook Naming**:
- Format: `nn.nn - DESCRIPTION.ipynb`
- Example: `02.10.23 - SVM MFCC20 (4-Fold, downsampled majority, standard SVC).ipynb`

### 7. Reproducibility Features

**Random Seed Control** - Use `predestination()`:
```python
from audace.jupytools import predestination

# Sets seeds for Python, NumPy, and TensorFlow
predestination(seed_value=23081965)
```

**Progress Logging** - Use `iprint()` instead of `print()`:
```python
from audace.jupytools import iprint

# Includes timestamp, CPU%, memory%, process memory
iprint("Starting feature extraction...")
# Output: [2025-11-21/10:30:45.123|25.0%|45.2%|1.23GB] Starting feature extraction...
```

**Checksum Validation**:
- Every dataset directory can be validated with MD5 checksum
- Manifest files contain expected checksum
- `checksumdir` library computes directory hash

---

## Working with the Codebase

### AudioDataset Class - Core API

**Initialization**:
```python
# Create new dataset from manifest
ds = AudioDataset("TUTO", source_path="/path/to/reference/audio", nprocs=4)

# Load existing dataset
ds = AudioDataset("TUTO")
```

**Dataset Information**:
```python
ds.info()              # Print dataset summary
ds.getNbFiles()        # Number of source files
ds.countSamples()      # Total audio chunks
ds.dumpDataFrame()     # Get all data as pandas DataFrame
```

**Labels** (binary/categorical targets):
```python
ds.addLabel("queen")                    # Add label column
ds.setLabel("queen", labelizer_func)    # Populate with function
ds.getLabel("queen")                    # Retrieve label values
ds.listLabelsValues("queen")            # Unique values
ds.dropLabel("queen")                   # Remove label
```

**Features** (e.g., MFCCs):
```python
ds.addFeature("mfcc20")
ds.setFeature("mfcc20", featurizer_func)
ds.getFeature("mfcc20")
ds.dropFeature("mfcc20")
```

**Attributes** (metadata):
```python
ds.addAttribute("hive_id")
ds.setAttribute("hive_id", attributor_func)
ds.getAttribute("hive_id")
```

**Querying**:
```python
# Raw SQL query
ds.query("SELECT * FROM samples WHERE file_id = 1", as_dict=True)

# DataFrame query
df = ds.queryDataFrame("SELECT name, queen, mfcc20 FROM samples")
```

### Database Schema

Each dataset has a SQLite database with tables:
- `config`: Dataset metadata (sample_rate, duration, overlap, etc.)
- `samples`: One row per audio chunk (name, file_id, start_t, end_t, + dynamic columns)
- `filenames`: Source audio file names
- `dictionary`: Schema metadata for labels/features/attributes

---

## Development Workflows

### Starting a New Experiment

1. **Create experiment directory**:
   ```bash
   mkdir "experiments/EXP03 - MY EXPERIMENT"
   cd "experiments/EXP03 - MY EXPERIMENT"
   ```

2. **Copy `kilroy_was_here.py`** from tutorials or another experiment

3. **Create notebook** with standard imports:
   ```python
   import kilroy_was_here
   from audace.jupytools import mooltipath, predestination, iprint
   from audace.audiodataset import AudioDataset

   predestination()  # Set random seeds for reproducibility
   ```

4. **Load or create dataset**:
   ```python
   ds = AudioDataset("MAIN1000")  # Load existing
   # or
   ds = AudioDataset("NEWDS", source_path="/path/to/audio", nprocs=4)
   ```

5. **Follow experiment pattern**:
   - Initialize
   - Load data
   - Extract features
   - Train model
   - Evaluate
   - Visualize results

### Adding Shared Code

**When to add to `audace/`**:
- Function/class used by multiple experiments
- General-purpose audio processing
- Reusable ML utilities
- Database operations

**Steps**:
1. Create new file in `audace/` or add to existing module
2. Follow naming convention (lowerCamelCase for public API)
3. Add docstrings
4. Import in notebooks: `from audace.mymodule import myFunction`

### Dataset Creation from Reference Data

1. **Create manifest file** `datasets/MYDATA.mnf`:
   ```
   22050
   60
   0

   source_audio_1.wav
   source_audio_2.wav
   ```

2. **Build dataset**:
   ```python
   ds = AudioDataset("MYDATA",
                     source_path="/path/to/reference/audio",
                     nprocs=4)  # Use multiprocessing
   ```

3. **Compute checksum**:
   ```python
   from checksumdir import dirhash
   md5 = dirhash(ds.samples_path, "md5")
   print(md5)  # Add this to manifest file line 4
   ```

4. **Update manifest** with MD5 checksum for reproducibility

---

## Important Files to Know

### Core Framework Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `audace/audiodataset.py` | Main dataset class | AudioDataset, add/set/get/drop methods |
| `audace/jupytools.py` | Notebook utilities | mooltipath(), predestination(), iprint() |
| `audace/dblib.py` | Database operations | create(), add_thing(), set_thing() |
| `audace/featurizers.py` | Feature extraction | MFCC extraction, etc. |
| `audace/providers.py` | Data providers | Batch generation for training |
| `audace/splitters.py` | Train/test splitting | Stratified splits, k-fold |
| `audace/plotters.py` | Visualization | Confusion matrix, feature plots |

### Configuration Files

| File | Purpose |
|------|---------|
| `.kilroy` | Project root marker (NEVER MODIFY) |
| `.gitignore` | Excludes datasets/, tmp/, output/, snippets/ |
| `datasets/*.mnf` | Dataset manifests (reproducibility) |

---

## Common Patterns and Idioms

### 1. Featurizer Functions
Functions that compute features for each audio sample:

```python
def compute_mfcc20(row):
    """Featurizer: extracts 20 MFCC coefficients"""
    import librosa
    chunk_path = ds.samples_path / (row['name'] + '.wav')
    audio, sr = librosa.load(str(chunk_path), sr=ds.sample_rate)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
    return mfcc.mean(axis=1)  # Return mean across time

# Apply to dataset
ds.addFeature("mfcc20")
ds.setFeature("mfcc20", compute_mfcc20)
```

### 2. Labelizer Functions
Functions that assign labels based on filename patterns:

```python
def label_queen(row):
    """Labelizer: 1 if filename contains 'QueenBee', else 0"""
    filename = ds.filenames[row['file_id'] - 1][0]
    return 1.0 if 'QueenBee' in filename else 0.0

# Apply to dataset
ds.addLabel("queen")
ds.setLabel("queen", label_queen)
```

### 3. Train/Test Workflow
```python
from audace.splitters import stratifiedSplit

# Get data
df = ds.queryDataFrame("SELECT queen, mfcc20 FROM samples")
X = np.vstack(df['mfcc20'].values)
y = df['queen'].values

# Split
X_train, X_test, y_train, y_test = stratifiedSplit(X, y, test_size=0.3)

# Train model
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
iprint(f"Test accuracy: {score:.4f}")
```

---

## Git Workflow

### Gitignored Items
The following are automatically excluded from version control:
- `datasets/*/` - Dataset directories (too large, reproducible via manifests)
- `tmp/` - Temporary files anywhere
- `snippets/` - Exploratory code anywhere
- `output/` - Generated outputs anywhere
- `.ipynb_checkpoints/` - Jupyter checkpoints
- `__pycache__/` - Python bytecode

### What SHOULD be in Git
- All `.py` files in `audace/`
- All notebooks (`.ipynb`) in experiments and tutorials
- Manifest files (`datasets/*.mnf`)
- Documentation files
- The `.kilroy` marker file

### Branching
- Use feature branches for development
- Branch format: `claude/add-claude-documentation-<session-id>`
- Always push with: `git push -u origin <branch-name>`

---

## Troubleshooting Guide

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'audace'`
- **Cause**: Missing `import kilroy_was_here` at notebook start
- **Fix**: Add as first import in notebook

**Issue**: `ModuleNotFoundError: No module named 'numba.decorators'`
- **Cause**: Incompatible numba version
- **Fix**: `python -m pip install numba==0.48 --user`

**Issue**: Paths break when moving notebook
- **Cause**: Using relative paths instead of `mooltipath()`
- **Fix**: Replace relative paths with `mooltipath("dir", "file")`

**Issue**: Can't reproduce experiment results
- **Cause**: Different random seed or dataset
- **Fix**:
  - Call `predestination()` at notebook start
  - Verify dataset MD5 checksum matches manifest

**Issue**: Dataset build fails
- **Cause**: Source files not found or manifest incorrect
- **Fix**: Verify source_path exists and manifest lists correct files

---

## Performance Tips

### Multiprocessing for Dataset Creation
```python
# Use all CPU cores
import multiprocessing
nprocs = multiprocessing.cpu_count()
ds = AudioDataset("MYDATA", source_path="...", nprocs=nprocs)
```

### Memory Management
```python
# Monitor memory with iprint
iprint("Starting heavy computation...")
# [timestamp|CPU%|RAM%|ProcessGB] Starting heavy computation...

# Use garbage collection for large datasets
import gc
del large_variable
gc.collect()
```

### Progress Tracking
```python
from tqdm.auto import tqdm

# Use tqdm for loops
for i in tqdm(range(1000), desc="Processing"):
    # work...
```

---

## Testing and Validation

### Dataset Validation
```python
from checksumdir import dirhash

# Verify dataset matches manifest
expected_md5 = "d02ebf42437ed11fa55c3d35cc5502ec"
computed_md5 = dirhash(ds.samples_path, "md5")
assert computed_md5 == expected_md5, "Dataset checksum mismatch!"
```

### Reproducibility Checklist
- [ ] `predestination()` called at notebook start
- [ ] Dataset MD5 matches manifest
- [ ] All dependencies at correct versions
- [ ] Random seed documented
- [ ] Environment info captured (use `watermark` extension)

---

## AI Assistant Best Practices

### When Modifying Code

1. **Respect the coding style**:
   - Use lowerCamelCase for public methods (even though non-PEP8)
   - Use snake_case for private methods
   - Author prefers this style - "Deal with it" (see audiodataset.py:42)

2. **Maintain reproducibility**:
   - Don't break MD5 checksums
   - Document any changes to random seed behavior
   - Preserve `predestination()` calls

3. **Keep experiments isolated**:
   - Don't modify shared code unless necessary
   - If changing `audace/`, verify impact on all experiments
   - Consider backward compatibility

4. **Path handling**:
   - ALWAYS use `mooltipath()` for paths
   - Never hardcode absolute paths
   - Use `os.fspath()` if library requires strings not Path objects

### When Adding Features

1. **Choose the right location**:
   - Generic → `audace/`
   - Experiment-specific → experiment directory
   - Exploratory → `snippets/` (gitignored)

2. **Follow the pattern**:
   - Study similar existing code
   - Use same import style
   - Match logging style (`iprint()` not `print()`)

3. **Document properly**:
   - Add docstrings to functions
   - Update README if significant
   - Add examples in docstring

### When Debugging

1. **Use provided utilities**:
   ```python
   iprint("Debug info")  # Includes timestamp, CPU, memory
   ds.info()            # Dataset summary
   ds.query("SELECT * FROM samples LIMIT 5", as_dict=True)
   ```

2. **Check the database**:
   ```python
   # Inspect schema
   ds.query("SELECT * FROM dictionary", as_dict=True)

   # Count samples
   ds.countSamples()
   ds.countSamples("queen = 1")
   ```

3. **Verify file structure**:
   ```python
   # Check dataset exists
   assert ds.db_path.is_file()
   assert ds.samples_path.is_dir()

   # Count audio files
   import glob
   wav_files = glob.glob(str(ds.samples_path / "*.wav"))
   iprint(f"Found {len(wav_files)} audio chunks")
   ```

---

## External Resources

### Reference Datasets
- Primary source: https://zenodo.org/record/1321278
- Download and extract to local directory
- Use path when creating AudioDataset

### Documentation
- Sphinx docs generated (see `docs/` directory)
- Napoleon extension for Google-style docstrings
- Generate with: `sphinx-build docs docs/_build`

### Tools Used in Development
- Sonic Visualiser: https://www.sonicvisualiser.org/
- Vamp Plugins: https://www.vamp-plugins.org/
- TensorFlow Embedding Projector: https://projector.tensorflow.org/
- Watermark (environment capture): https://github.com/rasbt/watermark

---

## Quick Reference

### Essential Imports
```python
import kilroy_was_here
from audace.jupytools import mooltipath, predestination, iprint
from audace.audiodataset import AudioDataset
```

### Dataset Lifecycle
```python
# Create
ds = AudioDataset("NAME", source_path="/path", nprocs=4)

# Add labels/features
ds.addLabel("target")
ds.setLabel("target", labelizer_func)
ds.addFeature("mfcc")
ds.setFeature("mfcc", featurizer_func)

# Query
df = ds.queryDataFrame("SELECT * FROM samples")

# Export
ds.exportTSV(sql, output_dir, ["target"], ["mfcc"])
```

### Path Construction
```python
# Always use mooltipath
dataset_dir = mooltipath("datasets", "TUTO")
output_file = mooltipath("experiments", "EXP01", "output", "results.csv")
```

---

## Summary

ShowBees/AuDaCE is a well-structured research codebase with strong emphasis on:
- **Reproducibility** (checksums, manifests, random seeds)
- **Code reuse** (shared `audace/` framework)
- **Path independence** (`.kilroy` + `mooltipath()`)
- **Dataset management** (SQLite + audio chunks)
- **Notebook-based workflows** (Jupyter integration)

When working with this codebase, prioritize maintaining these principles while respecting the author's coding style preferences.

---

*This guide was auto-generated for AI assistants working with the ShowBees repository. Last updated: 2025-11-21*
