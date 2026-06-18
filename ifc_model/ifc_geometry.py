"""World-space bounding-box computation for an IFC model.

IFC vertices live in representation-local coordinate systems and are positioned
by a chain of ``IfcLocalPlacement`` transforms (and, for shared geometry, by
``IfcMappedItem`` transforms). A naive min/max over raw cartesian points mixes
local and world coordinates and reports wrong dimensions, so this module
resolves the placement/mapping transforms and projects geometry into world space
before accumulating the bounding box.

Coverage is pragmatic but grounded:
- explicit BREP / surface-model vertices are transformed directly (furniture,
  doors, mapped families);
- extruded-area solids contribute both their profile (base) and the profile
  swept by the extrusion depth, so swept walls/slabs/columns get real extents;
- rectangle/circle/arbitrary profiles are handled for the base outline.

The horizontal footprint (X/Y) is captured accurately; vertical extent (Z) is
captured from explicit vertices and extrusion tops. Everything is derived from
real geometry — nothing is invented.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Entities whose children we recurse into when gathering explicit vertices.
_RECURSE_TYPES = frozenset(
    {
        "IFCFACETEDBREP",
        "IFCCLOSEDSHELL",
        "IFCOPENSHELL",
        "IFCCONNECTEDFACESET",
        "IFCSHELLBASEDSURFACEMODEL",
        "IFCFACEBASEDSURFACEMODEL",
        "IFCFACE",
        "IFCFACEOUTERBOUND",
        "IFCFACEBOUND",
        "IFCPOLYLOOP",
        "IFCPOLYLINE",
        "IFCGEOMETRICCURVESET",
        "IFCGEOMETRICSET",
        "IFCgeometricCurveSet".upper(),
    }
)

EntityTable = Dict[int, Tuple[str, str]]


class _Geo:
    """Holds the entity table + a field-split cache and accumulates the bbox."""

    def __init__(self, entities: EntityTable, split_fn, refs_fn):
        self.e = entities
        self._split = split_fn
        self._refs = refs_fn
        self._fcache: Dict[int, List[str]] = {}
        self._place_cache: Dict[int, np.ndarray] = {}
        self.lo = np.array([np.inf, np.inf, np.inf])
        self.hi = np.array([-np.inf, -np.inf, -np.inf])

    def flds(self, eid: int) -> List[str]:
        if eid not in self._fcache:
            ent = self.e.get(eid)
            self._fcache[eid] = self._split(ent[1]) if ent else []
        return self._fcache[eid]

    # -- primitive readers -------------------------------------------------
    def point(self, eid: int) -> Optional[np.ndarray]:
        ent = self.e.get(eid)
        if not ent or not ent[0].startswith("IFCCARTESIANPOINT"):
            return None
        inner = ent[1].strip().lstrip("(").rstrip(")")
        try:
            vals = [float(x.strip().rstrip(".")) for x in inner.split(",") if x.strip()]
        except ValueError:
            return None
        if len(vals) == 2:
            vals.append(0.0)
        if len(vals) != 3:
            return None
        return np.array(vals)

    def direction(self, eid: int, default) -> np.ndarray:
        ent = self.e.get(eid)
        if not ent:
            return np.array(default, dtype=float)
        inner = ent[1].strip().lstrip("(").rstrip(")")
        try:
            vals = [float(x.strip().rstrip(".")) for x in inner.split(",") if x.strip()]
        except ValueError:
            return np.array(default, dtype=float)
        if len(vals) == 2:
            vals.append(0.0)
        if len(vals) != 3:
            return np.array(default, dtype=float)
        return np.array(vals)

    # -- transform builders ------------------------------------------------
    def axis2placement(self, eid: int) -> np.ndarray:
        """4x4 matrix from IfcAxis2Placement3D (or 2D)."""
        m = np.identity(4)
        ent = self.e.get(eid)
        if not ent:
            return m
        f = self.flds(eid)
        if not f:
            return m
        loc_ref = _ref(f[0])
        loc = self.point(loc_ref) if loc_ref else None
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        if "PLACEMENT3D" in ent[0]:
            if len(f) > 1 and _ref(f[1]):
                z = _norm(self.direction(_ref(f[1]), [0, 0, 1]))
            if len(f) > 2 and _ref(f[2]):
                x = self.direction(_ref(f[2]), [1, 0, 0])
        else:  # 2D placement: RefDirection is param 1
            if len(f) > 1 and _ref(f[1]):
                x = self.direction(_ref(f[1]), [1, 0, 0])
        x = x - np.dot(x, z) * z
        if np.linalg.norm(x) < 1e-9:
            x = np.array([1.0, 0.0, 0.0])
        x = _norm(x)
        y = np.cross(z, x)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        if loc is not None:
            m[:3, 3] = loc
        return m

    def transform_operator(self, eid: int) -> np.ndarray:
        """4x4 from IfcCartesianTransformationOperator3D (mapped-item target)."""
        m = np.identity(4)
        ent = self.e.get(eid)
        if not ent:
            return m
        f = self.flds(eid)
        x = self.direction(_ref(f[0]), [1, 0, 0]) if len(f) > 0 and _ref(f[0]) else np.array([1.0, 0, 0])
        y = self.direction(_ref(f[1]), [0, 1, 0]) if len(f) > 1 and _ref(f[1]) else np.array([0.0, 1, 0])
        origin = self.point(_ref(f[2])) if len(f) > 2 and _ref(f[2]) else None
        scale = 1.0
        if len(f) > 3:
            try:
                scale = float((f[3] or "1").strip().rstrip("."))
            except ValueError:
                scale = 1.0
        z = self.direction(_ref(f[4]), [0, 0, 1]) if len(f) > 4 and _ref(f[4]) else np.cross(x, y)
        x = _norm(x)
        z = _norm(z)
        y = _norm(np.cross(z, x))
        m[:3, 0] = x * scale
        m[:3, 1] = y * scale
        m[:3, 2] = z * scale
        if origin is not None:
            m[:3, 3] = origin
        return m

    def local_placement(self, eid: int) -> np.ndarray:
        """World 4x4 from an IfcLocalPlacement chain."""
        if eid in self._place_cache:
            return self._place_cache[eid]
        ent = self.e.get(eid)
        if not ent or "LOCALPLACEMENT" not in ent[0]:
            return np.identity(4)
        f = self.flds(eid)
        parent_ref = _ref(f[0]) if len(f) > 0 else None
        rel_ref = _ref(f[1]) if len(f) > 1 else None
        rel = self.axis2placement(rel_ref) if rel_ref else np.identity(4)
        world = self.local_placement(parent_ref) @ rel if parent_ref else rel
        self._place_cache[eid] = world
        return world

    # -- accumulation ------------------------------------------------------
    def _add(self, world_pt: np.ndarray) -> None:
        self.lo = np.minimum(self.lo, world_pt)
        self.hi = np.maximum(self.hi, world_pt)

    def collect(self, item_id: int, matrix: np.ndarray, depth: int = 0) -> None:
        if depth > 80:
            return
        ent = self.e.get(item_id)
        if not ent:
            return
        t = ent[0]

        if t.startswith("IFCCARTESIANPOINT"):
            p = self.point(item_id)
            if p is not None:
                self._add((matrix @ np.append(p, 1.0))[:3])
            return

        if t == "IFCMAPPEDITEM":
            f = self.flds(item_id)
            repmap = _ref(f[0]) if len(f) > 0 else None
            target = _ref(f[1]) if len(f) > 1 else None
            if not repmap:
                return
            rf = self.flds(repmap)
            origin = _ref(rf[0]) if len(rf) > 0 else None
            mapped_rep = _ref(rf[1]) if len(rf) > 1 else None
            m = matrix
            if target:
                m = m @ self.transform_operator(target)
            if origin:
                m = m @ self.axis2placement(origin)
            if mapped_rep:
                self.collect(mapped_rep, m, depth + 1)
            return

        if t == "IFCSHAPEREPRESENTATION":
            f = self.flds(item_id)
            items = self._refs(f[3]) if len(f) > 3 else []
            for it in items:
                self.collect(it, matrix, depth + 1)
            return

        if t == "IFCEXTRUDEDAREASOLID":
            self._collect_extruded(item_id, matrix, depth)
            return

        if t == "IFCRECTANGLEPROFILEDEF":
            self._collect_rectangle(item_id, matrix)
            return

        if t in _RECURSE_TYPES or t.endswith("PROFILEDEF") or t.startswith("IFCCOMPOSITECURVE") or t == "IFCTRIMMEDCURVE":
            for ref in self._refs(ent[1]):
                self.collect(ref, matrix, depth + 1)
            return

        # Unknown container: recurse references defensively (cheap; points only add).
        for ref in self._refs(ent[1]):
            child = self.e.get(ref)
            if child and (
                child[0].startswith("IFCCARTESIANPOINT")
                or child[0] in _RECURSE_TYPES
                or child[0].endswith("PROFILEDEF")
            ):
                self.collect(ref, matrix, depth + 1)

    def _collect_extruded(self, eid: int, matrix: np.ndarray, depth: int) -> None:
        f = self.flds(eid)
        if len(f) < 4:
            return
        swept = _ref(f[0])
        position = _ref(f[1])
        direction = self.direction(_ref(f[2]), [0, 0, 1]) if _ref(f[2]) else np.array([0.0, 0, 1])
        try:
            depth_val = float((f[3] or "0").strip().rstrip("."))
        except ValueError:
            depth_val = 0.0
        pos_m = matrix @ self.axis2placement(position) if position else matrix
        if swept:
            self.collect(swept, pos_m, depth + 1)
            # Swept (top) cap: translate profile along the extrusion vector.
            top = np.identity(4)
            top[:3, 3] = _norm(direction) * depth_val
            self.collect(swept, pos_m @ top, depth + 1)

    def _collect_rectangle(self, eid: int, matrix: np.ndarray) -> None:
        f = self.flds(eid)
        # IfcRectangleProfileDef(ProfileType, ProfileName, Position, XDim, YDim)
        if len(f) < 5:
            return
        pos = _ref(f[2])
        try:
            xd = float((f[3] or "0").strip().rstrip("."))
            yd = float((f[4] or "0").strip().rstrip("."))
        except ValueError:
            return
        pos_m = matrix @ self.axis2placement(pos) if pos else matrix
        hx, hy = xd / 2.0, yd / 2.0
        for cx, cy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            self._add((pos_m @ np.array([cx, cy, 0.0, 1.0]))[:3])

    # -- area / perimeter --------------------------------------------------
    def product_refs(self, eid: int) -> Tuple[Optional[int], Optional[int]]:
        """Return (placement_ref, product_definition_shape_ref) for an IfcProduct."""
        placement_ref = repr_ref = None
        for tok in self.flds(eid):
            r = _ref(tok)
            if not r:
                continue
            rt = self.e.get(r)
            if not rt:
                continue
            if "LOCALPLACEMENT" in rt[0] and placement_ref is None:
                placement_ref = r
            elif "PRODUCTDEFINITIONSHAPE" in rt[0] and repr_ref is None:
                repr_ref = r
        return placement_ref, repr_ref

    def iter_extruded_solids(self, rep_id: int, depth: int = 0):
        """Yield IfcExtrudedAreaSolid ids reachable from a representation item."""
        if depth > 60:
            return
        ent = self.e.get(rep_id)
        if not ent:
            return
        t = ent[0]
        if t == "IFCEXTRUDEDAREASOLID":
            yield rep_id
            return
        if t == "IFCMAPPEDITEM":
            f = self.flds(rep_id)
            repmap = _ref(f[0]) if f else None
            if repmap:
                rf = self.flds(repmap)
                mapped_rep = _ref(rf[1]) if len(rf) > 1 else None
                if mapped_rep:
                    yield from self.iter_extruded_solids(mapped_rep, depth + 1)
            return
        if t == "IFCSHAPEREPRESENTATION":
            f = self.flds(rep_id)
            for it in (self._refs(f[3]) if len(f) > 3 else []):
                yield from self.iter_extruded_solids(it, depth + 1)
            return
        for r in self._refs(ent[1]):
            child = self.e.get(r)
            if child and child[0] in ("IFCSHAPEREPRESENTATION", "IFCMAPPEDITEM", "IFCEXTRUDEDAREASOLID"):
                yield from self.iter_extruded_solids(r, depth + 1)

    def _points_2d(self, curve_id: int, out: List[np.ndarray], depth: int = 0) -> None:
        if depth > 60:
            return
        ent = self.e.get(curve_id)
        if not ent:
            return
        if ent[0].startswith("IFCCARTESIANPOINT"):
            p = self.point(curve_id)
            if p is not None:
                out.append(p[:2])
            return
        if ent[0] == "IFCPOLYLINE":
            for r in self._refs(ent[1]):
                self._points_2d(r, out, depth + 1)
            return
        # Composite curves / segments / trimmed curves: gather any cartesian points.
        for r in self._refs(ent[1]):
            child = self.e.get(r)
            if not child:
                continue
            if child[0].startswith("IFCCARTESIANPOINT") or "CURVE" in child[0] or child[0] == "IFCPOLYLINE":
                self._points_2d(r, out, depth + 1)

    def profile_area_perimeter(self, profile_id: int) -> Optional[Tuple[float, float]]:
        """Return (area, perimeter) of a profile in the model's length unit² / unit."""
        ent = self.e.get(profile_id)
        if not ent:
            return None
        t = ent[0]
        f = self.flds(profile_id)
        if t == "IFCRECTANGLEPROFILEDEF" and len(f) >= 5:
            try:
                xd = float((f[3] or "0").strip().rstrip("."))
                yd = float((f[4] or "0").strip().rstrip("."))
            except ValueError:
                return None
            return xd * yd, 2.0 * (xd + yd)
        if t == "IFCCIRCLEPROFILEDEF" and len(f) >= 4:
            try:
                r = float((f[3] or "0").strip().rstrip("."))
            except ValueError:
                return None
            return float(np.pi * r * r), float(2 * np.pi * r)
        if t == "IFCARBITRARYCLOSEDPROFILEDEF" and len(f) >= 3:
            outer = _ref(f[2])
            if not outer:
                return None
            pts: List[np.ndarray] = []
            self._points_2d(outer, pts)
            return _shoelace(pts)
        return None


def _ref(token: str) -> Optional[int]:
    token = (token or "").strip()
    if token.startswith("#"):
        try:
            return int(token[1:])
        except ValueError:
            return None
    return None


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _shoelace(points: List[np.ndarray]) -> Optional[Tuple[float, float]]:
    """Polygon area (absolute) and perimeter from ordered 2D boundary points."""
    pts = [p for p in points if p is not None and len(p) >= 2]
    if len(pts) < 3:
        return None
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    area2 = 0.0
    perim = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        area2 += x1 * y2 - x2 * y1
        perim += float(np.hypot(x2 - x1, y2 - y1))
    return abs(area2) / 2.0, perim


def compute_floor_metrics(
    entities: EntityTable,
    slab_ids: List[int],
    split_fn,
    refs_fn,
) -> List[Dict[str, float]]:
    """Per-slab floor-plate area and perimeter (in the model's length unit² / unit).

    For each slab the largest extruded profile is taken as the floor plate (slabs
    are extruded vertically, so the profile area equals the plan area, invariant
    under the rigid placement transform).
    """
    geo = _Geo(entities, split_fn, refs_fn)
    plates: List[Dict[str, float]] = []
    for eid in slab_ids:
        _, repr_ref = geo.product_refs(eid)
        if not repr_ref:
            continue
        rf = geo.flds(repr_ref)
        best: Optional[Tuple[float, float]] = None
        for rep in (refs_fn(rf[2]) if len(rf) > 2 else []):
            for solid in geo.iter_extruded_solids(rep):
                sf = geo.flds(solid)
                profile = _ref(sf[0]) if sf else None
                if not profile:
                    continue
                ap = geo.profile_area_perimeter(profile)
                if ap and (best is None or ap[0] > best[0]):
                    best = ap
        if best:
            plates.append({"eid": eid, "area": best[0], "perimeter": best[1]})
    return plates


def compute_world_bbox(
    entities: EntityTable,
    element_ids: List[int],
    split_fn,
    refs_fn,
) -> Optional[Dict[str, float]]:
    """Return world-space bounding box / dimensions over the given element ids.

    ``split_fn(params)`` splits a parameter string into top-level fields and
    ``refs_fn(token)`` returns the ``#`` references inside a token — both reused
    from the STEP parser so this module stays parser-agnostic.
    """
    geo = _Geo(entities, split_fn, refs_fn)
    for eid in element_ids:
        ent = entities.get(eid)
        if not ent:
            continue
        f = geo.flds(eid)
        # IfcProduct: ObjectPlacement and Representation positions vary by type;
        # locate them by scanning for the local placement + product-def-shape refs.
        placement_ref = None
        repr_ref = None
        for tok in f:
            r = _ref(tok)
            if not r:
                continue
            rt = entities.get(r)
            if not rt:
                continue
            if "LOCALPLACEMENT" in rt[0] and placement_ref is None:
                placement_ref = r
            elif "PRODUCTDEFINITIONSHAPE" in rt[0] and repr_ref is None:
                repr_ref = r
        world = geo.local_placement(placement_ref) if placement_ref else np.identity(4)
        if not repr_ref:
            continue
        rf = geo.flds(repr_ref)
        reps = refs_fn(rf[2]) if len(rf) > 2 else []
        for rep in reps:
            geo.collect(rep, world)

    if not np.all(np.isfinite(geo.lo)) or not np.all(np.isfinite(geo.hi)):
        return None
    dims = geo.hi - geo.lo
    return {
        "min": [round(float(v), 1) for v in geo.lo],
        "max": [round(float(v), 1) for v in geo.hi],
        "length_x": round(float(dims[0]), 1),
        "width_y": round(float(dims[1]), 1),
        "height_z": round(float(dims[2]), 1),
    }
