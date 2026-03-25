# Optimal Spatial Reference Objects Selection

This project implements a stepwise spatial reference object selection workflow for a target polygon. It takes one target object and a set of candidate OSM polygon and line features, then filters and ranks the candidates to generate an optimal reference object set for spatial localization or spatial description tasks.

The current core implementation is in:

- `OptimalRefObjSelect.py`

## What this code does

Given:

- one target polygon Shapefile,
- one polygon reference dataset,
- one line reference dataset,

the algorithm performs the following operations:

1. **Containment check**: if a reference polygon almost contains the target polygon, keep only that object and terminate early.
2. **Extent retrieval**: expand the target's minimum rotated rectangle and keep only references intersecting the expanded extent.
3. **Salience computation and ranking**: compute multiple salience components and a final salience score.
4. **Direction uniqueness filtering**: keep directionally representative objects while avoiding redundant references.
5. **Intermediate result export**: save polygon and line outputs for each major step.
6. **Runtime statistics export**: save a time cost report to Excel.

## Main workflow

The main external entry point is:

```python
run_reference_selection(target_shp_path, osm_data_polygons, osm_data_lines, results_dir)
```

### Inputs

- `target_shp_path`: path to a target polygon Shapefile.
- `osm_data_polygons`: a `GeoDataFrame` of polygon reference objects.
- `osm_data_lines`: a `GeoDataFrame` of line reference objects.
- `results_dir`: directory for intermediate and final outputs.

### Output

Returns a `GeoDataFrame` containing the final selected reference objects.

The returned result may include:

- salience-related fields such as `S_topo`, `S_dist`, `S_size`, `S_area`, `S_sim`, `S_prop`, `S_name`, `S_name_sa`, `S_name_su`, `salience`
- direction fields such as `dir_all` and `dir_u`

## Algorithm structure

### 1. `ContainmentChecker`

Purpose:

- Performs a fuzzy topological containment test.
- If a candidate polygon overlaps the target polygon by at least a threshold ratio, the algorithm keeps only that object.

Key behavior:

- Threshold is controlled by `FUZZY_CONTAINMENT_THRESHOLD = 0.90`.
- Only polygon-like geometries are considered for containment.

### 2. `ExtentRetriever`

Purpose:

- Builds an expanded retrieval extent from the target polygon's minimum rotated rectangle.
- Filters reference objects by spatial intersection with the expanded rectangle.

Key behavior:

- Computes the target MRR.
- Uses a predefined area-level table (`AREA_LIMITS`) to determine an expansion increment.
- Scales the rectangle around its centroid.
- Saves the expanded rectangle as a Shapefile when `save_dir` is provided.

### 3. `SalienceRanker`

Purpose:

- Computes salience scores for each reference object.
- Produces a ranked candidate set.

Implemented salience components:

- **Topology salience**
- **Distance salience**
- **Area/size salience**
- **Attribute salience**
- **Name salience**
- **Similarity salience**

Final salience score:

```text
salience = 0.2 * S_topo
         + 0.2 * S_dist
         + 0.2 * S_prop
         + 0.2 * S_name
         + 0.2 * S_size
```

Notes:

- `S_size` is defined as the mean of area/size salience and similarity salience.
- Name salience uses public-awareness and uniqueness subcomponents.
- The code includes optional support for `pytrends` when computing part of the name salience.

### 4. `DirectionUniquenessChecker`

Purpose:

- Removes references that are too internal to the target object.
- Assigns direction labels to candidates.
- Keeps only directionally representative objects.

Key behavior:

- Builds 8 conical direction patches around the target centroid:
  - `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`
- Computes the proportion of each object falling into each directional patch.
- Filters by `rho_threshold`.
- Removes objects whose inside-target ratio exceeds `inside_ratio_threshold`.
- Supports:
  - first-round direction uniqueness within polygon and line subsets separately,
  - second-round direction uniqueness across the merged line + polygon result.

Important methods:

- `process(...)`: perform direction uniqueness filtering.
- `assign_directions(...)`: assign direction labels without final directional competition.
- `merge_line_polygon_refs(...)`: merge line and polygon results and run a second-round global directional competition.

## Expected data requirements

### Geometry types

The workflow expects:

- target object: `Polygon` or `MultiPolygon`
- references: `Polygon`, `MultiPolygon`, `LineString`, or `MultiLineString`

### CRS requirements

This code performs:

- distance calculations,
- area calculations,
- length calculations,
- geometric intersections.

So the input data should use a **projected CRS** rather than a geographic CRS.

The example in the script uses projected data such as `EPSG:3395`.

### Attribute fields

Some salience computations expect these fields in the reference datasets:

- `majorclass`: used for polygon attribute salience
- `fclass`: used for line attribute salience
- `name`: used for name salience

If any of these fields are missing, the corresponding salience component may become inaccurate or fall back to zero.

## File outputs

The code writes several intermediate outputs into `results_dir`.

### Intermediate step outputs

For each main step, polygon and line results are saved separately when available:

- `1_refs_polygons_containment_check.shp`
- `1_refs_lines_containment_check.shp`
- `3_refs_polygons_extent_retrieval.shp`
- `3_refs_lines_extent_retrieval.shp`
- `4_refs_polygons_salience_ranking.shp`
- `4_refs_lines_salience_ranking.shp`
- `5_refs_polygons_direction_uniqueness.shp`
- `5_refs_lines_direction_uniqueness.shp`

### Additional outputs

Depending on the workflow and parameters, the following may also be written:

- expanded MRR extent Shapefile
- direction patch Shapefiles
- `time_cost_report.xlsx`

## Installation

Recommended Python version:

- Python 3.10+

Install core dependencies:

```bash
pip install numpy pandas geopandas shapely openpyxl fiona pyproj
```

Optional dependency for name salience:

```bash
pip install pytrends
```

Depending on your environment, you may also need compatible GDAL/Fiona/GeoPandas binaries.

## Basic usage

### Option 1: Run the script directly

At the bottom of the file there is a `__main__` block. Update these paths first:

```python
TARGET_SHP_PATH = r'path_to_target.shp'
OSM_DATA_POLYGONS_PATH = r'path_to_polygon_refs.shp'
OSM_DATA_LINES_PATH = r'path_to_line_refs.shp'
RESULT_DIR = r'path_to_output_folder'
```

Then run:

```bash
python v17_StartWithSalienceRanker_en.py
```

### Option 2: Import and call from another script

```python
import geopandas as gpd
from v17_StartWithSalienceRanker_en import run_reference_selection

TARGET_SHP_PATH = r"path_to_target.shp"
POLYGON_PATH = r"path_to_polygon_refs.shp"
LINE_PATH = r"path_to_line_refs.shp"
RESULT_DIR = r"path_to_results"

osm_polygons = gpd.read_file(POLYGON_PATH)
osm_lines = gpd.read_file(LINE_PATH)

final_refs = run_reference_selection(
    TARGET_SHP_PATH,
    osm_polygons,
    osm_lines,
    RESULT_DIR,
)

print(final_refs[["salience", "dir_u"]].head())
```

## Workflow notes

### Early termination behavior

The workflow can terminate early in at least two cases:

1. A fuzzy containing object is found in Step 1.
2. The reference set becomes empty after a filtering step.

### Direction filtering behavior

Direction processing includes a topological pre-filter that removes objects mostly inside the target object.

This is controlled by:

- `inside_ratio_threshold`

If this threshold is too high, strongly internal objects may remain.
If it is too low, too many objects may be removed.

### Shapefile field-name limitations

Because intermediate results are saved as ESRI Shapefiles, be aware that:

- field names may be truncated,
- data types may be simplified,
- long text fields may be constrained.

If you need richer output tables, consider also exporting to GeoPackage or Parquet in a future revision.

## Known limitations

- The workflow currently assumes one target object per target Shapefile.
- Attribute salience depends on the specific semantic classes in your data.
- Name salience quality depends on the availability and quality of the `name` field.
- `pytrends`-based public-awareness scoring may be unstable depending on network, locale, rate limits, or library availability.
- Some thresholds are hard-coded and may need tuning for different study areas.
- The current example paths in `__main__` are local Windows paths and must be changed before use.

## Suggested project layout

A simple project layout could look like this:

```text
project_root/
├─ OptimalRefObjSelect.py
├─ README.md
├─ data/
   ├─ Target/
   ├─ OSM/
   └─ results/
```

## Next improvements

Possible future improvements:

- add a configuration file for thresholds and weights,
- support batch processing for multiple target objects,
- export final results in GeoPackage format,
- separate algorithm code from experiment scripts,
- add unit tests for each step,
- add logging instead of plain `print()` statements.

## Citation / description suggestion

If you use this code in a report or thesis, you can describe it as:

> A stepwise spatial reference object selection framework that integrates containment filtering, extent retrieval, multi-factor salience ranking, and direction uniqueness constraints for selecting representative polygon and line reference objects around a target polygon.

