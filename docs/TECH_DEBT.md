# Technical Debt — Track and revisit

Items here are known imperfect designs that work for now but should be
revisited when their cost-of-inaction grows.

---

## 1. Label-source switch is a feature flag, not a Plugin / Strategy Pattern

**Recorded**: 2026-04-21
**Severity**: Medium (works but limits extensibility)
**Estimated fix cost**: 2-3h refactoring + tests

### Current state

Label source selection is implemented as scattered `if/else` branches and
ad-hoc `--scheme` arguments across three files:

| File | Mechanism |
|---|---|
| `src/training/train_v3.py` STEP 4 | `if args.label_fusion: ... else: ...` |
| `scripts/build_fire_labels.py` | `if args.scheme == 'cwfis': build_cwfis() else: build_nbac_nfdb()` |
| `src/evaluation/benchmark_baselines.py` | `--fire_label_npy` accepts a pre-built file |

Adding a third source (e.g. MODIS MCD64A1, provincial perimeters) requires
modifying all three files. There is no shared interface or registry.

### Why it's not a problem yet

We currently have only two sources (CWFIS / NBAC+NFDB). The branching
overhead is tolerable.

### When to fix

Refactor to a true Strategy / Plugin pattern when **any** of these
happen:

1. Adding a third label source (e.g. MCD64A1 burned area, provincial
   2025 perimeters)
2. Multiple downstream files need their own label-source switch
3. Need to unit-test each label-source implementation in isolation

### Proposed refactor

```python
# src/data_ops/labels/base.py
class LabelSource(ABC):
    @abstractmethod
    def build_fire_stack(self, dates, profile, args) -> np.ndarray: ...
    @abstractmethod
    def cache_suffix(self) -> str: ...    # for cache key
    @abstractmethod
    def provenance(self) -> dict: ...     # for sidecar JSON

# src/data_ops/labels/cwfis.py
class CWFISLabelSource(LabelSource):
    def build_fire_stack(self, dates, profile, args):
        df = load_hotspot_data(args.hotspot_csv)
        return rasterize_hotspots_batch(df, dates, profile)
    def cache_suffix(self): return "_cwfis"
    def provenance(self): return {"source": "CWFIS hotspot CSV", ...}

# src/data_ops/labels/nbac_nfdb.py
class NBACNFDBLabelSource(LabelSource): ...

# src/data_ops/labels/__init__.py — registry
LABEL_SOURCES = {
    "cwfis":      CWFISLabelSource,
    "nbac_nfdb":  NBACNFDBLabelSource,
}

# In train_v3.py / build_fire_labels.py / benchmark_baselines.py:
src = LABEL_SOURCES[args.label_source]()
fire_stack = src.build_fire_stack(dates, profile, args)
cache_key = f"...r{r}{src.cache_suffix()}_{...}.npy"
```

Single CLI flag `--label_source {cwfis,nbac_nfdb,...}` replaces:
- `--label_fusion` (boolean)
- `--scheme` (string in build_fire_labels.py)
- ad-hoc `--fire_label_npy` (still useful but as override)

### Migration path (when triggered)

1. Create `src/data_ops/labels/` package with abstract base + 2 concrete classes.
2. Migrate train_v3.py STEP 4 to use registry; keep `--label_fusion` as
   deprecated alias for backward compat.
3. Migrate build_fire_labels.py and benchmark_baselines.py.
4. Add unit tests (`tests/test_label_sources.py`).
5. Delete deprecated aliases after one or two confirmed-working runs.
