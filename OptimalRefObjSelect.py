import os
import time
import inspect
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Union, Tuple
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, LineString, box, Point
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ContainmentChecker:
    """
    Step 1: Topological containment check
    Logic: fuzzy topological containment based on intersection area
    """
    # Fuzzy containment threshold: ratio of intersection area to target object area
    FUZZY_CONTAINMENT_THRESHOLD = 0.90 # 90% threshold

    def __init__(self, target_obj: Polygon):
        self.target_obj = target_obj

    def process(self, refs: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, bool]:
        """
        If ∃ Contain(O_r, O_t), O_r ∈ R then R <- O_r
        Return R and a boolean indicating whether a fuzzy containing object was found.
        """
        print("-> 1. Checking fuzzy containment...") # Step numbering updated for the new workflow

        if self.target_obj.geom_type not in ['Polygon', 'MultiPolygon']:
            print("   The target object is not a polygon type. Skipping containment check.")
            return refs, False

        target_area = self.target_obj.area
        if target_area <= 0:
            print("   The target object area is invalid. Skipping containment check.")
            return refs, False

        for index, row in refs.iterrows():
            geom = row.geometry
            # Containment is only meaningful when both O_r (reference object) and O_t (target object) are polygons
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Check intersection
                if geom.intersects(self.target_obj):
                    intersection = geom.intersection(self.target_obj)

                    # Ensure the intersection is polygonal and has positive area
                    if intersection.geom_type in ['Polygon', 'MultiPolygon'] and intersection.area > 0:
                        # Compute the ratio of intersection area to target object area
                        overlap_ratio = intersection.area / target_area

                        # Fuzzy containment test: if the overlap ratio exceeds the threshold
                        if overlap_ratio >= self.FUZZY_CONTAINMENT_THRESHOLD:
                            print(f"   Fuzzy containing object found: {index}, overlap ratio={overlap_ratio:.4f}. R keeps only this object.")
                            # Keep only this object and return True
                            refs = gpd.GeoDataFrame([row], crs=refs.crs).to_crs(refs.crs)
                            return refs, True

        print("   No fuzzy containing object found.")
        return refs, False


class ExtentRetriever:
    """
    Step 3: Extent retrieval
    Logic:
    1. Compute the minimum rotated rectangle (MRR) of target object Ot.
    2. Find the nearest standard area level larger than Ot as the "increment area".
    3. Retrieval area = MRR area + increment area.
    4. Expand proportionally around the original MRR center to avoid shifting.
    """

    # MLSG grid scales (China)
    AREA_LIMITS = [
        36346609670794.6, 9086652417698.65, 2271663104424.66, 567915776106.17,
        141978944026.54, 35494736006.64, 8873684001.66, 2218421000.41,
        554605250.1, 138651312.53, 34662828.13, 8665707.03,
        2166426.76, 541606.69, 135401.67, 33850.42,
        8462.6, 2115.65, 528.91, 132.23,
        33.06, 8.26, 2.07, 0.52,
        0.13, 0.03, 0.01, 0.002
    ]

    # # MLSG grid scales (California)
    # AREA_LIMITS = [
    #     1481082000000.00, 370270500000.00, 92567620000.00, 23141910000.00,
    #     5785477000.00, 1446369000.00, 361592300.00, 90398070.00,
    #     22599520.00, 5649879.00, 1412470.00, 353117.50,
    #     88279.37, 22069.84, 5517.46, 1379.37,
    #     344.84, 86.21, 21.55, 5.39,
    #     1.35, 0.34, 0.08, 0.02,
    #     0.01
    # ]

    def __init__(self, target_obj: Polygon, save_dir: str = None): 
        self.target_obj = target_obj
        # Directory for saving intermediate results
        self.save_dir = save_dir

    def _get_mrr_attributes(self, mrr_polygon: Polygon) -> tuple:
       
        coords = list(mrr_polygon.exterior.coords)
        v0, v1, v2, v3 = np.array(coords[:4])

        len_v0_v1 = np.linalg.norm(v1 - v0)
        len_v1_v2 = np.linalg.norm(v2 - v1)

        l = max(len_v0_v1, len_v1_v2)
        w = min(len_v0_v1, len_v1_v2)

        if l == len_v0_v1:
            long_edge_vector = v1 - v0
        else:
            long_edge_vector = v2 - v1

        angle_rad = np.arctan2(long_edge_vector[1], long_edge_vector[0])
        angle_deg = np.degrees(angle_rad)

        if angle_deg < 0:
            angle_deg += 180
        if angle_deg >= 180:
            angle_deg -= 180

        # Get the center point of the MRR
        center_x, center_y = mrr_polygon.centroid.coords[0]

        return center_x, center_y, l, w, angle_deg

    def _create_scaled_rotated_rectangle(self, center_x: float, center_y: float, l_new: float, w_new: float,
                                         angle_deg: float) -> Polygon:
        
        half_l, half_w = l_new / 2, w_new / 2

        rect = Polygon([
            (-half_l, -half_w),
            (half_l, -half_w),
            (half_l, half_w),
            (-half_l, half_w)
        ])

        rotated_rect = rotate(rect, angle_deg, origin=(0, 0), use_radians=False)
        new_rect = translate(rotated_rect, xoff=center_x, yoff=center_y)

        return new_rect

    def process(self, refs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print("-> 3. Performing extent retrieval (area-level incremental expansion)...")

        # 1. Compute the MRR of O_t
        mrr_polygon = self.target_obj.minimum_rotated_rectangle

        # 2. Compute the original attributes
        center_x, center_y, l_mbr, w_mbr, angle_deg = self._get_mrr_attributes(mrr_polygon)

        if l_mbr <= 0 or w_mbr <= 0:
            raise ValueError("MRR length or width is invalid; expansion cannot be performed.")

        # 3. Match an area level as the increment
        area_original = self.target_obj.area
        area_mbr = mrr_polygon.area

        # Find the smallest area level larger than target_area as the increment
        larger_areas = [a for a in self.AREA_LIMITS if a > area_original]
        if not larger_areas:
            area_expansion = self.AREA_LIMITS[0] # Use the largest level if the target exceeds the maximum grade
        else:
            area_expansion = min(larger_areas)
            
        area_new_target = area_mbr + area_expansion

        # 4. Compute the scale factor and expand
        k = np.sqrt(area_new_target / area_mbr)
        l_new = l_mbr * k
        w_new = w_mbr * k

        # 5. Generate the expanded geometry
        expanded_mrr = self._create_scaled_rotated_rectangle(center_x, center_y, l_new, w_new, angle_deg)

        # 6. Save intermediate results
        if self.save_dir is not None:
            try:
                mrr_gdf = gpd.GeoDataFrame(
                    {"id": [1], "type": ["expanded_mrr"]},
                    geometry=[expanded_mrr],
                    crs=refs.crs
                )
                mrr_shp_path = os.path.join(self.save_dir, "3_extent_expanded_mrr_extent_retrieval.shp")
                mrr_gdf.to_file(mrr_shp_path, driver="ESRI Shapefile", encoding="utf-8")
                print(f"   Expanded MRR saved to: {mrr_shp_path}")
            except Exception as e:
                print(f"   Warning: failed to save expanded MRR: {e}")
        
        # 7. Perform spatial intersection query
        r_filtered = refs[refs.geometry.intersects(expanded_mrr)].copy()
        print(f"   - Target area: {area_original:.2f}, matched increment: {area_expansion:.2f}")
        print(f"   - {len(r_filtered)} objects remain in R after extent retrieval.")
        return r_filtered


class SalienceRanker:
    """
    Step 4: Salience computation and ranking
    """

    def __init__(self, target_obj: Polygon):  # Added projected_crs support
        self.target_obj = target_obj
        self._fame_cache = {} # Cache for Google Trends scores

    # --- 1. Distance salience calculation ---
    def _calculate_distance_salience(self, refs: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute distance salience

        -Parameters
        refs: reference object set

        -Returns
        salience_dist: distance salience [0,1]
        """
        print("Compute distance salience...")

        if refs.empty:
            return np.array([])

        # 1. Compute the shortest distance from the target object to each reference object
        distances = refs.geometry.distance(self.target_obj).values

        d = distances

        # 2. Find the maximum and minimum values
        max_d = d.max()
        min_d = d.min()

        if max_d == min_d:
            salience_dist = np.ones_like(distances)  # Avoid division by zero
        else:
            salience_dist = (max_d - distances) / (max_d - min_d)

        return salience_dist

    # --- 2. Area/length salience calculation ---
    def _calculate_area_size_salience(self, refs: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute area/length salience: area for polygons and length for lines.

        -Parameters
        refs: reference object set

        -Returns
        salience_size: area (polygon) and length (line) salience [0,1]
        """
        print("Computing area/size salience...")

        if refs.empty:
            return np.array([])

        salience_size = np.zeros(len(refs), dtype=float)

        # Use geometry information from R_calc (already projected)
        polygon_indices = refs.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        line_indices = refs.geometry.geom_type.isin(['LineString', 'MultiLineString'])

        # --- A. Polygon objects ---
        r_poly = refs[polygon_indices]
        if not r_poly.empty:
            # Compute area in the projected CRS; the unit is typically square meters
            areas = r_poly.geometry.area.values

            max_a = areas.max()
            min_a = areas.min()

            if max_a == min_a:
                salience_poly = np.ones_like(areas)
            else:
                salience_poly = (areas - min_a) / (max_a - min_a)

            # Put the result back to the corresponding positions in the original GeoDataFrame
            original_indices = refs.index[polygon_indices]
            salience_size[refs.index.get_indexer(original_indices)] = salience_poly

        # --- B. Line objects ---
        r_line = refs[line_indices]
        if not r_line.empty:
            # Compute length in the projected CRS; the unit is typically meters
            lengths = r_line.geometry.length.values

            max_l = lengths.max()
            min_l = lengths.min()

            if max_l == min_l:
                salience_line = np.ones_like(lengths)
            else:
                salience_line = (lengths - min_l) / (max_l - min_l)

            # Put the result back to the corresponding positions in the original GeoDataFrame
            original_indices = refs.index[line_indices]
            salience_size[refs.index.get_indexer(original_indices)] = salience_line

        return salience_size

    # --- 3. Attribute salience calculation --- 
    def _calculate_attribute_salience(self, refs: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute attribute salience according to the graded assignment table in the paper.
        Assignment is based on the feature geometry type (Polygon/Line) and the corresponding attribute. [Polygons use majorclass; lines use fclass.]

        -Parameters
        refs: reference object set

        -Returns
        salience_pro: attribute salience [0,1]
        """
        print("Computing attribute salience (based on graded assignment table S_pro)...")

        required_cols = {'fclass', 'majorclass'}
        missing_cols = required_cols - set(refs.columns)

        if refs.empty:
            return np.zeros(len(refs))
    
        if not refs.geometry.geom_type.isin(['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']).all():
            print("   Warning: reference objects contain geometries other than polygons or lines; attribute salience may be inaccurate.")

        # Warning: if key columns are missing, salience for all objects will be set to 0
        if missing_cols:
            print(f"   Warning: reference object GeoDataFrame is missing fields: {missing_cols}; attribute salience may be inaccurate or set to 0.")

        # --- 1. Define graded salience mapping rules ---
        POLYGON_SALIENCE_MAP = {
            0.8: {'居住用地', '公共服务用地', '交通用地'},
            0.5: {'农业用地', '水体', '商业用地', '水资源用地', '工业用地'},
            0.2: {'特殊用地', '未分类','未利用土地'}
        }

        LINE_SALIENCE_MAP = {
            0.8: {'primary_road', 'railway', 'river'},
            0.5: {'secondary_road', 'subway', 'canal'},
            0.2: {'nonmotor_road,', 'else_road', 'tram', 'drain', 'stream'}
        }

        # --- 2. Initialize the salience array ---
        salience_pro = np.zeros(len(refs), dtype=float)

        # --- 3. Iterate through the reference object set and assign scores ---
        pos_index = 0  # Positional index
        for original_index, row in refs.iterrows():
            geom_type = row.geometry.geom_type
            current_score = 0.0  # Default salience is 0.0

            # A. Process polygon features
            if geom_type in ['Polygon', 'MultiPolygon']:
                if 'majorclass' in row:
                    # Convert majorclass to lowercase string for matching and handle NaN
                    attr_val = str(row['majorclass']).lower().replace(' ', '_')
                    
                    for score, keywords in POLYGON_SALIENCE_MAP.items():
                        # Check whether the attribute value contains/matches any keyword
                        if attr_val in keywords or any(k in attr_val for k in keywords):
                            current_score = score
                            break

            # B. Process line features
            elif geom_type in ['LineString', 'MultiLineString']:
                if 'fclass' in row:
                    # Convert fclass to lowercase string for matching and handle NaN
                    attr_val = str(row['fclass']).lower().replace(' ', '_')
                    
                    for score, keywords in LINE_SALIENCE_MAP.items():
                        if attr_val in keywords or any(k in attr_val for k in keywords):
                            current_score = score
                            break

            # C. Assign value
            salience_pro[pos_index] = current_score  # Assign using pos_index
            pos_index += 1  # Increment positional index

        return salience_pro

    # --- 4. Name salience calculation ---
    def _calculate_name_salience(self, refs: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute name salience
        S_name = (Sa + Su) / 2

        -Parameters
        refs: reference object set (must include the 'name' field)

        -Returns
        Name salience [0,1]
        """
        print("Compute name salience...")

        # === 1. Compute S_a (Public Awareness Salience) ===
        #    Use the Google Trends API to obtain 0-100 scores and normalize them to [0, 1]

        # --- Google Trends configuration ---
        HL = 'zh-CN' # Chinese
        # HL = 'en-US' # English
        TZ = 360
        TIMEFRAME = 'all'  # Use 'all' or a custom timeframe
        CATEGORY = 0
        GEO = 'CN'
        GPROP = ''
        SLEEP_TIME_SECONDS = 1  # Pause after each query (seconds)
        PROXIES = {}  # Proxy settings

        # --- urllib3 v2 compatibility patch ---
        try:
            import urllib3.util.retry as _retry_mod
            _OrigRetry = getattr(_retry_mod, "Retry", None)

            if _OrigRetry is not None:
                sig = inspect.signature(_OrigRetry.__init__)
                has_allowed = "allowed_methods" in sig.parameters
                has_method_whitelist = "method_whitelist" in sig.parameters

                if has_allowed and not has_method_whitelist:
                    class _RetryV2Compat(_OrigRetry):  # type: ignore
                        def __init__(self, *args, **kwargs):
                            if "method_whitelist" in kwargs and "allowed_methods" not in kwargs:
                                kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
                            super().__init__(*args, **kwargs)

                    setattr(_retry_mod, "Retry", _RetryV2Compat)
        except Exception:
            pass

        # Import pytrends
        try:
            from pytrends.request import TrendReq
        except ImportError:
            print("    Warning: pytrends is not installed; all Sa salience values will be set to 0.")
            Sa_scores = np.zeros(len(refs))
            pass 

        def build_pytrends():
            """Initialize TrendReq."""
            kwargs = dict(
                hl=HL,
                tz=TZ,
                retries=3,
                backoff_factor=0.5,
                proxies=PROXIES
            )

            sig = inspect.signature(TrendReq.__init__)
            if "timeout" in sig.parameters:
                kwargs["timeout"] = (5, 30)
            else:
                kwargs["requests_args"] = {"timeout": (5, 30)}

            return TrendReq(**kwargs)

        def download_interest_over_time(keyword):
            """Get interest_over_time data for a keyword."""
            if not keyword:
                return pd.DataFrame()

            try:
                pytrends = build_pytrends()
                pytrends.build_payload(
                    kw_list=[keyword],
                    cat=CATEGORY,
                    timeframe=TIMEFRAME,
                    geo=GEO,
                    gprop=GPROP
                )
                data = pytrends.interest_over_time()
                if 'isPartial' in data.columns:
                    data = data.drop(columns=['isPartial'])
                return data
            except Exception:
                return pd.DataFrame()

        def get_google_trend_mean(keyword: str):
            """Given a keyword, return the mean popularity over time (0-100)."""
            if not keyword:
                return None

            df = download_interest_over_time(keyword)

            if df.empty:
                return None

            mean_value = df.mean().values[0]
            return float(mean_value)

        # Main loop to compute Sa (0-100)
        Sa_unnormalized_scores = []

        for index, row in refs.iterrows():
            name = str(row['name']).strip() if row['name'] is not None else ""

            if not name:
                Sa_unnormalized_scores.append(0.0)
                continue

            # Check cache (using self._fame_cache)
            if name in self._fame_cache:
                score = self._fame_cache[name]
                Sa_unnormalized_scores.append(score if score is not None else 0.0)
                continue

            # Call the Google Trends function
            score = get_google_trend_mean(name)
            time.sleep(SLEEP_TIME_SECONDS)  # Respect API rate limits

            # Update cache
            self._fame_cache[name] = score

            Sa_unnormalized_scores.append(score if score is not None else 0.0)
        

        # Sa normalization: Min-Max normalize to [0, 1]
        Sa_array = np.array(Sa_unnormalized_scores)
        max_sa = Sa_array.max()
        min_sa = Sa_array.min()

        if max_sa == min_sa:
            # Safeguard: if all objects in the current batch have identical popularity (e.g., all 0 or all the same value)
            # Set to 1.0 when positive popularity exists; keep 0 when all are 0
            Sa_scores = np.where(Sa_array > 0, 1.0, 0.0)
        else:
            # Apply Min-Max normalization to scale the original range to [0, 1]
            Sa_scores = (Sa_array - min_sa) / (max_sa - min_sa)


        # === 2. Compute S_u (Toponym Uniqueness Salience) ===
        
        # Count occurrences of each place name (N)
        name_counts = refs['name'].value_counts()
        
        # First, convert name_counts to Su values (1/N)
        name_su_map = (1 / name_counts).to_dict()
        
        # Then map Su values to each row
        Su_scores = refs['name'].map(name_su_map).fillna(0.0).values
        Su_scores = np.array(Su_scores)

        # === 3. Combine to compute S_name ===
        
        # Check whether the lengths of Sa and Su arrays match
        if len(Sa_scores) != len(Su_scores):
             print("    Warning: Sa and Su arrays have different lengths; returning Sa or Su.")
             return Sa_scores # Assume Sa is more reliable

        # Final computation of S_name
        S_name = (Sa_scores + Su_scores) / 2.0
        
        print("    - Name salience (S_name) computation completed.")
        return S_name, Sa_scores, Su_scores

    # --- 5. Topology salience calculation ---
    def _calculate_topology_salience(self, refs: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute topology salience (S_topo).

        Polygon objects (Polygon / MultiPolygon):
            Use the ratio of intersection area between the target object and the reference polygon to the target area:
                overlap_ratio = Area(O_r ∩ O_t) / Area(O_t)
            Following the logic that more overlap means higher salience:
                - If they do not intersect: S_topo_poly = 0
                - If they intersect: S_topo_poly = overlap_ratio

        Line objects (LineString / MultiLineString):
            - For each line vertex, compute:
                * the distance array to the target polygon centroid
                * the distance array to the target polygon boundary
            - Compute Var (variance) and MAD (mean absolute deviation) for the two distance arrays above
            - Perform min-max normalization on Var / MAD across all lines
            - Combine topology salience using MAD metrics:
                S_topo_line = (cent_mad_norm + bound_mad_norm) / 2

        Returns:
            salience_topo: np.ndarray, topology salience of each reference object in [0, 1]
        """
        print("Computing topology salience (S_topo)...")

        # def normalize_to_0_to_1(values):
        #     """
        #     Min-Max normalize to [0, 1].
        #     values: can be list / np.ndarray / pd.Series
        #     """
        #     arr = np.asarray(values, dtype=float)
        #     if arr.size == 0:
        #         return arr

        #     min_v = arr.min()
        #     max_v = arr.max()
        #     if max_v == min_v:
        #         # All values are equal; normalize uniformly to 0
        #         return np.zeros_like(arr, dtype=float)
        #     return (arr - min_v) / (max_v - min_v)

        def normalize_to_1_to_0(values):
            """
            Min-Max normalize to [1, 0] (reversed: smaller values are closer to 1).
            """
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return arr

            min_v = arr.min()
            max_v = arr.max()
            if max_v == min_v:
                # All values are equal; normalize uniformly to 1
                return np.ones_like(arr, dtype=float)
            return (max_v - arr) / (max_v - min_v)

        def calculate_metrics_for_distance(distances_array):
            """
            Compute the variance (Var) and mean absolute deviation (MAD) of a distance array.
            Return (variance, mad)
            """
            arr = np.asarray(distances_array, dtype=float)
            if arr.size <= 1:
                # Too short to produce meaningful statistics; return 0 directly
                return 0.0, 0.0

            # Sample variance
            variance = np.var(arr, ddof=1)
            # Mean distance
            mean_val = np.mean(arr)
            # Mean of absolute residuals, i.e., mean absolute deviation
            mad = np.mean(np.abs(arr - mean_val))
            
            return variance, mad

        # ------------------------
        # Main logic starts here
        # ------------------------
        # if refs.empty:
        #     print("   R is empty; set S_topo to [].")
        #     return np.array([])

        salience_topo = np.zeros(len(refs), dtype=float)

        target = self.target_obj
        # if target is None or target.is_empty:
        #     print("   The target object is empty; S_topo is all zero.")
        #     return salience_topo

        # ========================================================
        # 1. Polygon objects: S_topo_poly = overlap_ratio (no intersection = 0)
        # ========================================================
        polygon_mask = refs.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        polygon_indices = np.where(polygon_mask)[0]

        if len(polygon_indices) > 0:
            target_area = getattr(target, "area", 0.0)

            if target_area <= 0:
                print("   Target object area <= 0; polygon topology salience is all zero.")
            else:
                for idx in polygon_indices:
                    geom = refs.geometry.iloc[idx]

                    # If they do not intersect, salience = 0
                    if not geom.intersects(target):
                        salience_topo[idx] = 0.0
                        continue

                    try:
                        inter = geom.intersection(target)
                        inter_area = getattr(inter, "area", 0.0)
                    except Exception:
                        inter_area = 0.0

                    if inter_area <= 0:
                        # Treat as non-intersecting
                        salience_topo[idx] = 0.0
                    else:
                        overlap_ratio = inter_area / target_area
                        # Safely clip numerical values to [0, 1]
                        overlap_ratio = max(0.0, min(1.0, overlap_ratio)) # Can be commented out
                        salience_topo[idx] = overlap_ratio

        # ========================================================
        # 2. Line objects: topology salience based on line-vertex distance statistics
        # ========================================================
        line_mask = refs.geometry.geom_type.isin(['LineString', 'MultiLineString'])
        line_indices = np.where(line_mask)[0]

        if len(line_indices) > 0:
            # Target polygon centroid and boundary
            centroid = target.centroid
            Cx, Cy = centroid.x, centroid.y  # Compute the centroid coordinates of the polygon
            boundary = target.boundary  # Get the boundary of the polygon geometry

            n_lines = len(line_indices)
            cent_var_arr = np.zeros(n_lines, dtype=float)
            cent_mad_arr = np.zeros(n_lines, dtype=float)
            bound_var_arr = np.zeros(n_lines, dtype=float)
            bound_mad_arr = np.zeros(n_lines, dtype=float)

            # 2.1 Compute distance arrays for each line and derive Var / MAD
            for pos, idx in enumerate(line_indices):
                geom = refs.geometry.iloc[idx]

                # Collect all vertex coordinates
                coords = []
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                elif geom.geom_type == 'MultiLineString':
                    for part in geom.geoms:
                        coords.extend(list(part.coords))

                if not coords:
                    # No vertices; metrics are 0
                    cent_var, cent_mad = 0.0, 0.0
                    bound_var, bound_mad = 0.0, 0.0
                else:
                    centroid_dists = []
                    boundary_dists = []

                    for (x, y) in coords:
                        dx = x - Cx
                        dy = y - Cy
                        dist_centroid = (dx * dx + dy * dy) ** 0.5
                        centroid_dists.append(dist_centroid)

                        p = Point(x, y)
                        dist_boundary = p.distance(boundary)
                        boundary_dists.append(dist_boundary)

                    cent_var, cent_mad = calculate_metrics_for_distance(centroid_dists)
                    bound_var, bound_mad = calculate_metrics_for_distance(boundary_dists)

                cent_var_arr[pos] = cent_var
                cent_mad_arr[pos] = cent_mad
                bound_var_arr[pos] = bound_var
                bound_mad_arr[pos] = bound_mad

            # 2.2 Normalize metrics across all lines
            # cent_var_norm = normalize_to_0_to_1(cent_var_arr)
            cent_mad_norm = normalize_to_1_to_0(cent_mad_arr)
            # bound_var_norm = normalize_to_0_to_1(bound_var_arr)
            bound_mad_norm = normalize_to_1_to_0(bound_mad_arr)

            # 2.3 Combine into line topology salience
            # Use the average of normalized MAD values as the final S_topo_line
            S_topo_line = (cent_mad_norm + bound_mad_norm) / 2.0

            # Write the result back to the global salience_topo array
            for pos, idx in enumerate(line_indices):
                salience_topo[idx] = S_topo_line[pos]

        return salience_topo

    # --- 6. Similarity salience calculation ---
    def _calculate_similarity_salience(self, refs: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute similarity salience between the target object and reference objects
        - If the reference object is a polygon, compute the area difference between the target and the reference;
        - If the reference object is a line, compute the difference between the target perimeter and the reference length;
        - Smaller differences yield higher salience scores.

        - Parameters
        refs: reference object set

        - Returns
        salience_sim: similarity salience [0, 1]
        """
        print("Computing similarity salience...")

        if refs.empty:
            return np.array([])

        # Get the area and perimeter of the target object
        target_area = self.target_obj.area
        target_perimeter = self.target_obj.length  # Polygon perimeter

        # Salience array for reference objects
        salience_sim = np.zeros(len(refs), dtype=float)

        # Used to store all computed difference values
        area_diffs = []
        length_diffs = []

        # First, compute all difference values
        for idx, row in refs.iterrows():
            geom = row.geometry

            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Polygon object: compute area difference
                ref_area = geom.area
                area_diff = abs(target_area - ref_area)
                area_diffs.append(area_diff)
                # Use positional index to avoid out-of-bounds issues
                area_idx = refs.index.get_loc(idx)  # Get positional index
                salience_sim[area_idx] = area_diff  # Store area difference

            elif geom.geom_type in ['LineString', 'MultiLineString']:
                # Line object: compute perimeter/length difference
                ref_length = geom.length
                length_diff = abs(target_perimeter - ref_length)
                length_diffs.append(length_diff)
                # Use positional index to avoid out-of-bounds issues
                length_idx = refs.index.get_loc(idx)  # Get positional index
                salience_sim[length_idx] = length_diff  # Store length difference

        # Compute the maximum and minimum difference values for polygon objects
        if len(area_diffs) > 0:
            max_area_diff = max(area_diffs)
            min_area_diff = min(area_diffs)
        else:
            max_area_diff = min_area_diff = 0

        # Compute the maximum and minimum difference values for line objects
        if len(length_diffs) > 0:
            max_length_diff = max(length_diffs)
            min_length_diff = min(length_diffs)
        else:
            max_length_diff = min_length_diff = 0

        # Normalize difference values for polygon objects
        for i in range(len(refs)):
            if refs.geometry.iloc[i].geom_type in ['Polygon', 'MultiPolygon']:
                # Polygon object: normalize
                if max_area_diff == min_area_diff:
                    salience_sim[i] = 1.0  # Identical area differences; set salience to the maximum
                else:
                    salience_sim[i] = (max_area_diff - salience_sim[i]) / (max_area_diff - min_area_diff + 1e-5)

        # Normalize difference values for line objects
        for i in range(len(refs)):
            if refs.geometry.iloc[i].geom_type in ['LineString', 'MultiLineString']:
                # Line object: normalize
                if max_length_diff == min_length_diff:
                    salience_sim[i] = 1.0  # Identical perimeter differences; set salience to the maximum
                else:
                    salience_sim[i] = (max_length_diff - salience_sim[i]) / (max_length_diff - min_length_diff + 1e-5)

        return salience_sim


    def process(self, refs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print("-> 4. Computing salience and ranking...")

        if refs.empty:
            print("   R is empty. Skipping salience computation.")
            return refs

        # --- 1. Compute all salience components ---
        score_t = self._calculate_topology_salience(refs)  # Topology salience
        score_d = self._calculate_distance_salience(refs)  # Distance salience
        score_area = self._calculate_area_size_salience(refs)  # Area salience
        score_p = self._calculate_attribute_salience(refs)  # Attribute salience
        score_n, score_n_sa, score_n_su = self._calculate_name_salience(refs) # Name salience
        score_sim = self._calculate_similarity_salience(refs)  # Similarity salience

        # --- 2. Redefine size salience S_size ---
        # The new size salience is the mean of the current area score and similarity score
        score_size_combined = (score_area + score_sim) / 2.0

        # Write each salience component into the attribute table (for shapefile export)
        refs['S_topo'] = score_t.astype(float)
        refs['S_dist'] = score_d.astype(float)
        refs['S_size'] = score_size_combined.astype(float) # Store the averaged size salience
        refs['S_area'] = score_area.astype(float) # Keep the original size score
        refs['S_sim'] = score_sim.astype(float)  # Newly added similarity salience
        refs['S_prop'] = score_p.astype(float) 
        refs['S_name'] = score_n.astype(float)
        refs['S_name_sa'] = score_n_sa.astype(float) # Name salience subcomponent
        refs['S_name_su'] = score_n_su.astype(float) # Uniqueness subcomponent

        w_t, w_d, w_s, w_p, w_n, = 0.2, 0.2, 0.2, 0.2, 0.2  # Includes the similarity salience weight

        if len(score_t) != len(refs):
            print("Error: salience component results do not match the number of GeoDataFrame rows.")
            refs['salience'] = 0.0
            return refs.sort_values(by='salience', ascending=False).reset_index(drop=True)

        refs['salience'] = (w_t * score_t + w_d * score_d + w_p * score_p + w_n * score_n + w_s * score_size_combined)  # Compute total salience

        r_ranked = refs.sort_values(by='salience', ascending=False).reset_index(drop=True)

        return r_ranked



class DirectionUniquenessChecker:
    """
    Step 5: Direction uniqueness
    """

    def __init__(self, target_obj: Polygon,
                 rho_threshold=0.1,
                 inside_ratio_threshold=0.9,
                 save_dir=None,
                 ):
        self.target_obj = target_obj
        self.rho_threshold = rho_threshold
        self.inside_ratio_threshold = inside_ratio_threshold
        self.save_dir = save_dir

    # ==========================================================
    # Direction patch saving
    # ==========================================================
    def _save_direction_patches(self, direction_patches, patch_type, crs):
        if self.save_dir is None or (isinstance(self.save_dir, str) and not self.save_dir.strip()):
            return
        try:
            patches_gdf = gpd.GeoDataFrame(
                [{"direction": k, "geometry": v} for k, v in direction_patches.items()],
                crs=crs
            )
            out_path = os.path.join(self.save_dir, f"5_direction_patches_{patch_type}.shp")
            patches_gdf.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
            print(f"   [Debug] Direction patches ({patch_type}) saved: {out_path}")
        except Exception as e:
            print(f"   [Warning] Failed to save direction patches ({patch_type}): {e}")

    # ==========================================================
    # Conical direction patches
    # ==========================================================
    def _construct_conical_direction_patches(self, base_object, distance=10000000):
        cx, cy = base_object.centroid.x, base_object.centroid.y
        directions = [
            ("N", 90), ("NE", 45), ("E", 0),
            ("SE", 315), ("S", 270), ("SW", 225),
            ("W", 180), ("NW", 135)
        ]
        tiles = {}
        for d, ang in directions:
            x1 = cx + distance * np.cos(np.radians(ang - 22.5))
            y1 = cy + distance * np.sin(np.radians(ang - 22.5))
            x2 = cx + distance * np.cos(np.radians(ang + 22.5))
            y2 = cy + distance * np.sin(np.radians(ang + 22.5))
            tiles[d] = Polygon([(cx, cy), (x1, y1), (x2, y2)])
        return tiles

    # ==========================================================
    # Proportion calculation (kept as is)
    # ==========================================================
    def _calculate_proportion_matrix(self, ref_object, direction_tiles):
        ref_type = ref_object.geom_type

        if ref_type in ["Polygon", "MultiPolygon"]:
            total = ref_object.area
        elif ref_type in ["LineString", "MultiLineString"]:
            total = ref_object.length
        else:
            return {}

        if total == 0:
            return {}

        result = {}
        for d, tile in direction_tiles.items():
            if not tile.intersects(ref_object):
                continue
            inter = ref_object.intersection(tile)
            val = inter.area if ref_type.startswith("Poly") else inter.length
            result[d] = val / total
        return result

    # ==========================================================
    # Topological filtering: remove objects whose inside-target ratio exceeds the threshold
    # ==========================================================
    def _remove_objects_inside_target(self, refs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove objects whose inside-target ratio exceeds the threshold.
        Polygon: inside_area / geom.area
        Line: inside_length / geom.length
        """
        if refs is None or refs.empty:
            return refs

        # Validate threshold type first
        try:
            threshold = float(self.inside_ratio_threshold)
        except Exception as e:
            raise ValueError(
                f"inside_ratio_threshold must be numeric, but the current value is: {self.inside_ratio_threshold!r}"
            ) from e

        keep_rows = []
        debug_rows = []

        for idx, row in refs.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            geom_type = geom.geom_type

            if geom_type not in ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]:
                keep_rows.append(idx)
                debug_rows.append((idx, geom_type, None, "KEEP_UNSUPPORTED_TYPE"))
                continue

            try:
                inter = geom.intersection(self.target_obj)
            except Exception as e:
                print(f"   [Topological filtering warning] idx={idx} intersection failed: {e}")
                keep_rows.append(idx)
                debug_rows.append((idx, geom_type, None, "KEEP_INTERSECTION_ERROR"))
                continue

            if geom_type in ["Polygon", "MultiPolygon"]:
                total_val = geom.area
                inside_val = 0.0 if inter.is_empty else inter.area
            else:
                total_val = geom.length
                inside_val = 0.0 if inter.is_empty else inter.length

            if total_val <= 0:
                debug_rows.append((idx, geom_type, None, "DROP_ZERO_TOTAL"))
                continue

            inside_ratio = inside_val / total_val

            if inside_ratio >= threshold:
                debug_rows.append((idx, geom_type, inside_ratio, "DROP"))
            else:
                keep_rows.append(idx)
                debug_rows.append((idx, geom_type, inside_ratio, "KEEP"))

        refs_filtered = refs.loc[keep_rows].copy()

        removed_count = len(refs) - len(refs_filtered)
        print(f"   [Topological filtering] threshold={threshold}, removed {removed_count} objects, kept {len(refs_filtered)} objects.")

        # Print only the first few rows for debugging
        for item in debug_rows[:20]:
            idx, geom_type, inside_ratio, status = item
            print(f"      idx={idx}, type={geom_type}, inside_ratio={inside_ratio}, status={status}")

        return refs_filtered

    # ==========================================================
    # Main workflow
    # ==========================================================
    def process(self, refs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print("-> 5. Checking direction uniqueness (lines/polygons separately)...")

        if refs is None or refs.empty or "salience" not in refs.columns:
            return refs

        # First perform topological filtering: remove objects whose inside-target ratio exceeds the threshold
        refs = self._remove_objects_inside_target(refs)

        if refs is None or refs.empty:
            print("   [Note] No objects remain for direction-uniqueness checking after topological filtering.")
            return refs

        try:
            max_sal_idx = refs["salience"].astype(float).idxmax()
        except Exception:
            max_sal_idx = refs["salience"].idxmax()

        crs = getattr(refs, "crs", "EPSG:3395")

        # Build direction patches
        tile = self._construct_conical_direction_patches(self.target_obj)
        self._save_direction_patches(tile, "tile", crs)

        directions = [
            "NW", "N", "NE", "W", "E", "SW", "S", "SE",
        ]

        best = {d: {"salience": -np.inf, "index": None} for d in directions}

        for idx, row in refs.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            try:
                sal = float(row["salience"])
            except Exception:
                continue

            ratio = self._calculate_proportion_matrix(geom, tile)

            for d, r in ratio.items():
                if d not in best:
                    continue
                if r <= self.rho_threshold:
                    continue
                if sal > best[d]["salience"]:
                    best[d] = {"salience": sal, "index": idx}

        from collections import defaultdict

        idx_to_dirs = defaultdict(list)
        for d, v in best.items():
            if v["index"] is not None:
                idx_to_dirs[v["index"]].append(d)

        selected = []
        seen = set()
        for d in directions:
            idx = best[d]["index"]
            if idx is not None and idx not in seen:
                seen.add(idx)
                selected.append(idx)

        if max_sal_idx not in selected:
            selected.insert(0, max_sal_idx)

        r_final = refs.loc[selected].copy()
        r_final["dir_u"] = [
            ",".join(idx_to_dirs.get(i, ["GLOBAL_MAX"] if i == max_sal_idx else []))
            for i in r_final.index
        ]

        print(f"   Finally kept {len(r_final)} objects.")
        return r_final

    # ==========================================================
    # Second-round direction uniqueness (lines + polygons together)
    # ==========================================================
    @staticmethod
    def merge_line_polygon_refs(
        lines_gdf: gpd.GeoDataFrame,
        polys_gdf: gpd.GeoDataFrame
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Second-round direction uniqueness (lines + polygons together):

        Description:
        - This function performs another round of competition at the combined line + polygon level so that only one object is retained for each direction.
        - The result after the second-round global competition is written back to the final dir_u field.

        Inputs:
        - lines_gdf, polys_gdf:
          results that have already been processed by DirectionUniquenessChecker.process,
          must contain the following fields:
            - salience
            - dir_u (first-round result, e.g., N,NE,SW or GLOBAL_MAX)

        Output:
        - The result filtered again by direction uniqueness at the combined line + polygon level (with dir_u updated)
        - Lines and polygons are still returned separately (structure unchanged)
        """

        # ------------------------------------------------------
        # 1) Basic checks
        # ------------------------------------------------------
        has_lines = lines_gdf is not None and not lines_gdf.empty
        has_polys = polys_gdf is not None and not polys_gdf.empty

        if not (has_lines and has_polys):
            return lines_gdf, polys_gdf

        # ------------------------------------------------------
        # 2) Merge (preserve type and original index for splitting back later)
        # ------------------------------------------------------
        lines_tmp = lines_gdf.copy()
        lines_tmp["__geom_type"] = "line"
        lines_tmp["__orig_index"] = lines_tmp.index

        polys_tmp = polys_gdf.copy()
        polys_tmp["__geom_type"] = "polygon"
        polys_tmp["__orig_index"] = polys_tmp.index

        refs_all = pd.concat([lines_tmp, polys_tmp], axis=0)

        if "salience" not in refs_all.columns or "dir_u" not in refs_all.columns:
            return lines_gdf, polys_gdf

        # ------------------------------------------------------
        # 3) Globally most salient object (force keep, but it does not occupy a direction by default)
        # ------------------------------------------------------
        try:
            max_sal_idx_all = refs_all["salience"].astype(float).idxmax()
        except Exception:
            max_sal_idx_all = refs_all["salience"].idxmax()

        # ------------------------------------------------------
        # 4) Parse first-round dir_u to obtain "object -> candidate directions"
        # ------------------------------------------------------
        from collections import defaultdict

        directions = [
            "NW", "N", "NE", "W", "E", "SW", "S", "SE",
        ]

        idx_to_dirs = defaultdict(list)
        for idx, row in refs_all.iterrows():
            dir_u_val = row.get("dir_u", "")
            if not isinstance(dir_u_val, str) or not dir_u_val:
                continue
            if dir_u_val == "GLOBAL_MAX":
                continue

            parts = [p.strip() for p in dir_u_val.split(",") if p.strip()]
            for d in parts:
                if d in directions:
                    idx_to_dirs[idx].append(d)

        # ------------------------------------------------------
        # 5) Build "direction -> candidate object list"
        # ------------------------------------------------------
        direction_to_indices = {d: [] for d in directions}
        for idx, dirs in idx_to_dirs.items():
            for d in dirs:
                direction_to_indices[d].append(idx)

        # ------------------------------------------------------
        # 6) For each direction, choose the object with the highest salience across the combined line+polygon set
        # ------------------------------------------------------
        win_dirs = defaultdict(list)
        selected_indices_ordered = []
        selected_indices_set = set()

        for d in directions:
            cand_indices = direction_to_indices[d]
            if not cand_indices:
                continue

            try:
                best_idx = max(cand_indices, key=lambda i: float(refs_all.at[i, "salience"]))
            except Exception:
                best_idx = max(cand_indices, key=lambda i: refs_all.at[i, "salience"])

            win_dirs[best_idx].append(d)

            if best_idx not in selected_indices_set:
                selected_indices_set.add(best_idx)
                selected_indices_ordered.append(best_idx)

        # ------------------------------------------------------
        # 7) Force-add the globally most salient object (placed first)
        # ------------------------------------------------------
        if max_sal_idx_all in selected_indices_set:
            if max_sal_idx_all in selected_indices_ordered:
                selected_indices_ordered.remove(max_sal_idx_all)
            selected_indices_ordered.insert(0, max_sal_idx_all)
        else:
            selected_indices_ordered.insert(0, max_sal_idx_all)
            selected_indices_set.add(max_sal_idx_all)

        # ------------------------------------------------------
        # 8) Filter and write back the second-round dir_u
        # ------------------------------------------------------
        refs_final_all = refs_all.loc[selected_indices_ordered].copy()

        def _make_dir_u(idx) -> str:
            dirs_won = win_dirs.get(idx, [])
            if idx == max_sal_idx_all and not dirs_won:
                return "GLOBAL_MAX"
            return ",".join([d for d in directions if d in dirs_won])

        refs_final_all["dir_u"] = [_make_dir_u(i) for i in refs_final_all.index]

        # ------------------------------------------------------
        # 9) Split back into lines / polygons and write the updated dir_u back to the final output table
        # ------------------------------------------------------
        lines_mask = refs_final_all["__geom_type"] == "line"
        polys_mask = refs_final_all["__geom_type"] == "polygon"

        line_diru_map = dict(
            zip(
                refs_final_all.loc[lines_mask, "__orig_index"],
                refs_final_all.loc[lines_mask, "dir_u"]
            )
        )
        poly_diru_map = dict(
            zip(
                refs_final_all.loc[polys_mask, "__orig_index"],
                refs_final_all.loc[polys_mask, "dir_u"]
            )
        )

        selected_line_idx = refs_final_all.loc[lines_mask, "__orig_index"].tolist()
        selected_poly_idx = refs_final_all.loc[polys_mask, "__orig_index"].tolist()

        lines_final = lines_gdf.loc[selected_line_idx].copy() if has_lines else lines_gdf
        polys_final = polys_gdf.loc[selected_poly_idx].copy() if has_polys else polys_gdf

        if lines_final is not None and not lines_final.empty:
            lines_final["dir_u"] = lines_final.index.map(lambda i: line_diru_map.get(i, ""))
        if polys_final is not None and not polys_final.empty:
            polys_final["dir_u"] = polys_final.index.map(lambda i: poly_diru_map.get(i, ""))

        return lines_final, polys_final

    def assign_directions(self, refs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assign direction labels to all input reference objects.
        No filtering is performed; only directional assignment is computed.

        Note:
        - Topological filtering is also performed first:
          - If the ratio of inside-target area/length exceeds the threshold, no direction will be assigned
        """
        if refs is None or refs.empty:
            return refs

        # Perform topological filtering first
        refs = self._remove_objects_inside_target(refs)

        if refs is None or refs.empty:
            return refs

        tile = self._construct_conical_direction_patches(self.target_obj)

        def get_row_dirs(geom):
            if geom is None or geom.is_empty:
                return ""

            ratio = self._calculate_proportion_matrix(geom, tile)

            matched_dirs = []
            for d, r in ratio.items():
                if r > self.rho_threshold:
                    matched_dirs.append(d)

            return ",".join(matched_dirs)

        refs = refs.copy()
        refs["dir_all"] = refs.geometry.apply(get_row_dirs)
        return refs

# ====================================================================
# === II. Data loading and main workflow functions (single-object processing) ===
# ====================================================================

def load_target_object(target_shp_path):
    """
    Load the target object Shapefile and extract the first polygon geometry as O_t.
    As required, assume the target object Shapefile contains only one object.

    Parameters:
    - target_shp_path: target polygon feature to be localized

    Returns:
    - first_geometry: geometric information of the target polygon object
    - target_gdf_info: attribute information of the target polygon object
    """
    try:
        # Read target polygon vector data
        target_gdf = gpd.read_file(target_shp_path)

        # The target object is known to be unique, so process the first one directly
        # Use .iloc[0] to extract the first row
        first_row = target_gdf.iloc[0]
        first_geometry = first_row.geometry

        # Ensure that a GeoDataFrame containing only the target object information is returned as info
        target_gdf_info = gpd.GeoDataFrame([first_row], crs=target_gdf.crs)

        if first_geometry.geom_type == 'MultiPolygon':
            return first_geometry.convex_hull, target_gdf_info
        return first_geometry, target_gdf_info

    except Exception as e:
        print(f"Failed to load Shapefile: {e}")
        return None, None

# New helper function: prepare a GeoDataFrame for saving. This helps prevent errors when saving shapefiles starting from the salience-computation stage.
def _prepare_gdf_for_saving(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Helper function: prepare a GeoDataFrame for saving.
    Handle duplicate column names, data type compatibility, and Shapefile limitations.
    """
    if gdf is None or gdf.empty:
        return gdf

    # 1. Force removal of duplicate column names
    # Keep only the first occurrence of duplicate column names and remove later duplicates
    gdf_to_save = gdf.loc[:, ~gdf.columns.duplicated()].copy()

    # 2. Ensure the important field 'salience' has the correct type
    if 'salience' in gdf_to_save.columns:
        try:
            gdf_to_save['salience'] = gdf_to_save['salience'].astype(float)
        except Exception:
            pass

    # 3. Convert datetime types unsupported by Shapefile
    for col in gdf_to_save.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf_to_save[col]):
            gdf_to_save[col] = gdf_to_save[col].astype(str)

    # 4. Automatically handle field name length limits (Shapefile limit: 10 characters)
    # This step prevents new duplicate column-name conflicts caused by truncation
    new_columns = {}
    for col in gdf_to_save.columns:
        if col != 'geometry' and len(str(col)) > 10:
            # Shorten field names, e.g., salience_score -> salience_s
            new_name = str(col)[:10]
            new_columns[col] = new_name
    
    if new_columns:
        gdf_to_save = gdf_to_save.rename(columns=new_columns)
        # Deduplicate once again after renaming to avoid identical names after truncation (e.g., S_name_unique and S_name_uniqueness)
        gdf_to_save = gdf_to_save.loc[:, ~gdf_to_save.columns.duplicated()].copy()

    return gdf_to_save

def _save_intermediate_result(refs: gpd.GeoDataFrame, step_num: int, step_name: str, base_dir: str):
    """ 
    Internal helper function: save the intermediate GeoDataFrame to the specified step folder, separated into polygons and lines. (Saved within the same output folder)

    Parameters:
    - refs: reference object set
    - step_num: step number, used as the folder/index number
    - step_name: step name, appended to the file name
    - base_dir: parent output directory path
    """

    if refs.empty:
        print(f"   Warning: result R of step {step_name} is empty; no file was saved.")
        return

    # 2. Split by geometry type
    polygon_types = ['Polygon', 'MultiPolygon']
    line_types = ['LineString', 'MultiLineString']

    refs_polygons = refs[refs.geometry.geom_type.isin(polygon_types)].copy()
    refs_lines = refs[refs.geometry.geom_type.isin(line_types)].copy()

    # 3. Save polygon objects separately
    if not refs_polygons.empty:
        output_path_poly = os.path.join(base_dir, str(step_num) + "_refs_polygons_" + step_name + ".shp")
        refs_poly_to_save = _prepare_gdf_for_saving(refs_polygons)
        try:
            refs_poly_to_save.to_file(output_path_poly, driver='ESRI Shapefile', encoding='utf-8')
            print(f" => Intermediate result (polygon) saved to: {output_path_poly}")
        except Exception as e:
            print(f"Warning: failed to save the intermediate result (polygon) for step {step_name} ({e}).")

    # 4. Save line objects separately
    if not refs_lines.empty:
        output_path_line = os.path.join(base_dir, str(step_num) + "_refs_lines_" + step_name + ".shp")
        refs_line_to_save = _prepare_gdf_for_saving(refs_lines)
        try:
            refs_line_to_save.to_file(output_path_line, driver='ESRI Shapefile', encoding='utf-8')
            print(f" => Intermediate result (line) saved to: {output_path_line}")
        except Exception as e:
            print(f"Warning: failed to save the intermediate result (line) for step {step_name} ({e}).")

    if refs_polygons.empty and refs_lines.empty:
        print(f"Warning: result R of step {step_name} contains neither polygons nor lines; no file was saved.")

def run_reference_selection(target_shp_path, osm_data_polygons, osm_data_lines, results_dir):
    """
    External entry point of the reference object selection algorithm, responsible for data preparation and workflow orchestration.
    Processes both polygon and line reference datasets.
    Adds a process_dir parameter for saving intermediate results.

    Parameters:
    - target_shp_path: target polygon object to be localized
    - osm_data_polygons: OSM polygon vector data
    - osm_data_lines: OSM line vector data
    - area_expansion_ratio: area expansion ratio for extent retrieval
    - results_dir: result file output directory

    Returns:
    - references_set: output reference object set (updated step by step)
    """
    stats = [] 
    t_start_all = time.time()
    print("--- Algorithm workflow started (step-class mode) ---")

    # Load target object O_t
    t0 = time.time()
    O_t, target_gdf_info = load_target_object(target_shp_path)

    # If CRS matches or is undefined, concatenate directly
    references_set = pd.concat([osm_data_polygons, osm_data_lines], ignore_index=True)
    stats.append(["Data loading", time.time()-t0]) # Data loading time statistics
    print(f"OSM datasets merged, with a total of {len(references_set)} objects.")

    # --- Workflow orchestration (order adjusted) ---

    # 1. Step 1: Containment check
    t1 = time.time()
    containment_checker = ContainmentChecker(O_t)
    references_set, found_containment = containment_checker.process(references_set)
    _save_intermediate_result(references_set, 1, "containment_check", results_dir)  # *** Save trace output ***
    stats.append(["Step 1: containment check", time.time()-t1]) # Step 1 timing statistics

    # If a containing object is found, terminate the algorithm early
    if found_containment:
        save_stats_to_excel(stats, t_start_all, results_dir) # Save Excel statistics even if the process terminates early
        print("--- Algorithm finished: a fuzzy containing object was found ---")
        return references_set
    
    # If R is empty after containment check, terminate early
    if references_set.empty:
        print("--- Algorithm finished: R is empty after containment check ---")
        return references_set


    # 3. Step 3: Extent retrieval (line 9)
    t3 = time.time()
    extent_retriever = ExtentRetriever(O_t, save_dir=results_dir)
    references_set = extent_retriever.process(references_set)
    _save_intermediate_result(references_set, 3, "extent_retrieval", results_dir)  # *** Save trace output ***
    stats.append(["Step 3: extent retrieval", time.time()-t3]) # Step 3 timing statistics

    # If R is empty after extent retrieval, terminate early
    if references_set.empty:
        save_stats_to_excel(stats, t_start_all, results_dir) # Save Excel statistics even if the process terminates early
        print("--- Algorithm finished: R is empty after extent retrieval ---")
        return references_set

    # 4. Step 4: Salience computation and ranking (topological exclusion already included)
    t4 = time.time()
    salience_ranker = SalienceRanker(O_t)
    references_set = salience_ranker.process(references_set)

    # --- Pre-label directions in the Step 4 result for later analysis ---
    # Instantiate a temporary checker (or instantiate earlier)
    direction_tagger = DirectionUniquenessChecker(O_t, save_dir=results_dir)
    references_set = direction_tagger.assign_directions(references_set)

    _save_intermediate_result(references_set, 4, "salience_ranking", results_dir)  # *** Save trace output ***
    stats.append(["Step 4: salience computation", time.time()-t4]) # Step 4 timing statistics

    # 5. Step 5: Direction uniqueness
    # First split by geometry type, then perform one round of direction uniqueness separately, followed by a second round over combined lines + polygons

    t5 = time.time()
    polygon_mask = references_set.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    line_mask = references_set.geometry.geom_type.isin(['LineString', 'MultiLineString'])

    refs_polygons = references_set[polygon_mask].copy()
    refs_lines = references_set[line_mask].copy()

    # (1) Perform the first round of direction uniqueness separately for lines / polygons
    checker_poly = DirectionUniquenessChecker(O_t, save_dir=results_dir)
    checker_line = DirectionUniquenessChecker(O_t, save_dir=results_dir)

    if not refs_polygons.empty:
        polys_step5 = checker_poly.process(refs_polygons)
    else:
        polys_step5 = refs_polygons

    if not refs_lines.empty:
        lines_step5 = checker_line.process(refs_lines)
    else:
        lines_step5 = refs_lines

    # (2) Combined line + polygon direction uniqueness: keep only the most salient object for each direction
    lines_final, polys_final = DirectionUniquenessChecker.merge_line_polygon_refs(lines_step5, polys_step5)

    # (3) Merge back into one overall R for saving intermediate results (the final output can still be split into lines / polygons)
    refs_crs = references_set.crs
    parts = []
    if polys_final is not None and not polys_final.empty:
        parts.append(polys_final)
    if lines_final is not None and not lines_final.empty:
        parts.append(lines_final)

    if parts:
        references_set = gpd.GeoDataFrame(pd.concat(parts, axis=0), crs=refs_crs)
    else:
        # If both are empty, return the original empty structure
        references_set = gpd.GeoDataFrame(columns=references_set.columns, crs=refs_crs)

    _save_intermediate_result(references_set, 5, "direction_uniqueness", results_dir)  # *** Save trace output ***

    stats.append(["Step 5: direction uniqueness", time.time()-t5]) # Step 5 timing statistics
    print("--- Algorithm finished: the optimal reference object set has been generated ---")
    # --- Save Excel output at the end ---
    save_stats_to_excel(stats, t_start_all, results_dir)
    return references_set

def save_stats_to_excel(stats_list, start_time, output_dir):
    """Export function"""
    df = pd.DataFrame(stats_list, columns=["Step name", "Time cost (s)"])
    # Compute and append total time
    total_time = time.time() - start_time
    df.loc[len(df)] = ["★ Total runtime", total_time]
    
    excel_path = os.path.join(output_dir, "time_cost_report.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\n[OK] Time statistics exported to: {excel_path}")


if __name__ == '__main__':

    print("\n--- Algorithm framework test run ---")

    # Target polygon object Shapefile path
    TARGET_SHP_PATH = r'OptimalRefObjSelect\data\Target\Camp1.shp'
    # OSM polygon base geographic information Shapefile path
    OSM_DATA_POLYGONS_PATH = r'OptimalRefObjSelect\data\OSM\California\Polygon_California_mkt.shp'
    # OSM line base geographic information Shapefile path
    OSM_DATA_LINES_PATH = r'OptimalRefObjSelect\data\OSM\California\Line_California_mkt.shp'
    # Final result output directory
    RESULT_DIR = r"OptimalRefObjSelect\data\results"

    try:
        print(f"Loading OSM polygon reference data: {OSM_DATA_POLYGONS_PATH}")
        OSM_POLYGONS = gpd.read_file(OSM_DATA_POLYGONS_PATH)

        print(f"Loading OSM line reference data: {OSM_DATA_LINES_PATH}")
        OSM_LINES = gpd.read_file(OSM_DATA_LINES_PATH)

        print(f"\n========================================================")
        print(f"--- Processing target object ---")  # Only one target object

        optimal_references = run_reference_selection(
            TARGET_SHP_PATH,
            OSM_POLYGONS.copy(),
            OSM_LINES.copy(),
            RESULT_DIR
        )
       
    except Exception as e:
        # Catch and print all runtime exceptions
        import sys, traceback

        print(f"\n!!! Algorithm run failed. Please check the environment and data. Error: {type(e).__name__}: {e} !!!")
        traceback.print_exc(file=sys.stdout)  # Print detailed stack trace

