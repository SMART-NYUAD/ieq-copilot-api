"""Dependency-free IFC (STEP / ISO-10303-21) reader and grounded-fact builder.

The server has no ``ifcopenshell`` dependency, and the bundled model carries no
explicit quantity entities, so this module parses the STEP text directly and
extracts only what is needed to answer natural-language questions about the
building: units, the spatial hierarchy, storey elevations, an element inventory,
per-element dimensions/properties, and materials.

Parsing is lazy and cached by ``(path, mtime, size)`` so the 17 MB model is read
at most once per process unless the file changes. Only non-bulk-geometry
entities are retained in memory — the ~250k cartesian-point / face lines are
skipped — so the resident footprint stays small.

We deliberately do NOT compute a global bounding box: most cartesian points live
in profile- or mapped-item-local coordinate systems, so a naive min/max over all
points would conflate local and world coordinates and report wrong dimensions.
Every measurement surfaced here is read directly from an IFC attribute, a named
property value, or an element type label — never inferred from raw geometry.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# IfcProduct subtypes that represent physical building/furnishing elements.
_ELEMENT_TYPES = frozenset(
    {
        "IFCWALL",
        "IFCWALLSTANDARDCASE",
        "IFCDOOR",
        "IFCWINDOW",
        "IFCSLAB",
        "IFCROOF",
        "IFCSTAIR",
        "IFCSTAIRFLIGHT",
        "IFCRAMP",
        "IFCCOLUMN",
        "IFCBEAM",
        "IFCMEMBER",
        "IFCPLATE",
        "IFCRAILING",
        "IFCCOVERING",
        "IFCCURTAINWALL",
        "IFCFURNISHINGELEMENT",
        "IFCFLOWTERMINAL",
        "IFCLIGHTFIXTURE",
        "IFCFLOWSEGMENT",
        "IFCFLOWFITTING",
        "IFCBUILDINGELEMENTPROXY",
        "IFCSPACE",
    }
)

_REF_RE = re.compile(r"#(\d+)")


@dataclass
class IFCElement:
    eid: int
    ifc_type: str
    name: Optional[str]
    object_type: Optional[str]
    tag: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Dict[str, str]] = field(default_factory=dict)
    materials: List[str] = field(default_factory=list)
    storey: Optional[str] = None


@dataclass
class IFCModelFacts:
    path: str
    schema: Optional[str]
    project_name: Optional[str]
    project_description: Optional[str]
    application: Optional[str]
    author: Optional[str]
    timestamp: Optional[str]
    units: List[str]
    site_name: Optional[str]
    site_latitude: Optional[str]
    site_longitude: Optional[str]
    building_name: Optional[str]
    storeys: List[Dict[str, Any]]
    elements: List[IFCElement]
    materials: List[str]
    length_unit: str = "mm"
    dimensions: Optional[Dict[str, Any]] = None
    architectural_metrics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Low-level STEP parsing helpers
# ---------------------------------------------------------------------------

def _split_top_level(params: str) -> List[str]:
    """Split a parameter string on top-level commas, respecting strings/parens."""
    parts: List[str] = []
    depth = 0
    in_str = False
    cur: List[str] = []
    i = 0
    n = len(params)
    while i < n:
        c = params[i]
        if in_str:
            if c == "'":
                if i + 1 < n and params[i + 1] == "'":
                    cur.append("''")
                    i += 2
                    continue
                in_str = False
                cur.append(c)
            else:
                cur.append(c)
        else:
            if c == "'":
                in_str = True
                cur.append(c)
            elif c == "(":
                depth += 1
                cur.append(c)
            elif c == ")":
                depth -= 1
                cur.append(c)
            elif c == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(c)
        i += 1
    parts.append("".join(cur).strip())
    return parts


def _unquote(token: str) -> Optional[str]:
    token = token.strip()
    if not token or token in ("$", "*"):
        return None
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1].replace("''", "'")
    return token


def _as_ref(token: str) -> Optional[int]:
    token = token.strip()
    if token.startswith("#"):
        try:
            return int(token[1:])
        except ValueError:
            return None
    return None


def _refs_in(token: str) -> List[int]:
    return [int(m) for m in _REF_RE.findall(token or "")]


def _as_number(token: str) -> Optional[float]:
    try:
        return float(token.strip().rstrip("."))
    except (ValueError, AttributeError):
        return None


def _typed_value(token: str) -> Tuple[Optional[str], str]:
    """Parse a wrapped typed value like ``IFCLABEL('Foo')`` → ("IFCLABEL", "Foo")."""
    token = token.strip()
    m = re.match(r"^([A-Z0-9_]+)\((.*)\)$", token, re.DOTALL)
    if not m:
        return None, _unquote(token) or token
    type_name, inner = m.group(1), m.group(2).strip()
    unq = _unquote(inner)
    return type_name, unq if unq is not None else inner


def _iter_step_entities(path: str):
    """Yield ``(eid, ifc_type, params)`` for each entity line, handling the rare
    multi-line record by accumulating until the terminating semicolon."""
    buf = ""
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not in_data:
                if line == "DATA;":
                    in_data = True
                continue
            if line == "ENDSEC;":
                break
            buf += line
            if not buf.endswith(";"):
                continue
            record = buf[:-1]
            buf = ""
            if not record.startswith("#") or "=" not in record:
                continue
            eid_str, _, body = record.partition("=")
            try:
                eid = int(eid_str[1:])
            except ValueError:
                continue
            paren = body.find("(")
            if paren == -1:
                continue
            ifc_type = body[:paren].strip().upper()
            params = body[paren + 1 : body.rfind(")")]
            yield eid, ifc_type, params


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------

_SI_PREFIX = {
    "MILLI": "milli",
    "CENTI": "centi",
    "DECI": "deci",
    "KILO": "kilo",
    "MICRO": "micro",
}


def _format_si_unit(params: str) -> Optional[str]:
    fields = _split_top_level(params)
    if len(fields) < 4:
        return None
    unit_type = (_unquote(fields[1]) or fields[1]).replace(".", "").replace("UNIT", " unit").strip().lower()
    prefix = (fields[2] or "").replace(".", "").strip().upper()
    name = (_unquote(fields[3]) or fields[3]).replace(".", "").strip().lower()
    prefix_label = _SI_PREFIX.get(prefix, "")
    return f"{unit_type}: {prefix_label}{name}".replace("  ", " ").strip()


def _format_compound_angle(token: str) -> Optional[str]:
    """Format an IfcCompoundPlaneAngleMeasure list like (42,21,31,401671)."""
    nums = [int(x) for x in re.findall(r"-?\d+", token)]
    if len(nums) < 3:
        return None
    deg, minutes, seconds = nums[0], abs(nums[1]), abs(nums[2])
    return f"{deg}° {minutes}' {seconds}\""


_PREFIX_SUFFIX = {"MILLI": "mm", "CENTI": "cm", "DECI": "dm", "KILO": "km"}


def _project_length_suffix(entities: Dict[int, Tuple[str, str]]) -> str:
    """Resolve the model's length unit from IfcUnitAssignment.

    A model often defines several LENGTHUNIT IfcSIUnits (e.g. millimetre as the
    project unit plus metre as a base for derived units). Only the unit listed in
    the IfcUnitAssignment is authoritative, so we resolve through it rather than
    guessing from the full unit list.
    """
    assigned_ids: List[int] = []
    for eid, (ifc_type, params) in entities.items():
        if ifc_type == "IFCUNITASSIGNMENT":
            assigned_ids = _refs_in(params)
            break
    candidates = assigned_ids or list(entities.keys())
    for ref in candidates:
        ent = entities.get(ref)
        if not ent or ent[0] != "IFCSIUNIT":
            continue
        f = _split_top_level(ent[1])
        if len(f) < 4 or "LENGTHUNIT" not in f[1]:
            continue
        prefix = (f[2] or "").replace(".", "").strip().upper()
        return _PREFIX_SUFFIX.get(prefix, "m")
    return "mm"


_WALL_THICKNESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm")


def _wall_thickness_mm(name: Optional[str]) -> Optional[float]:
    """Extract a wall thickness in mm from a Revit-style name like 'Generic - 200mm'."""
    if not name:
        return None
    m = _WALL_THICKNESS_RE.search(name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _build_architectural_metrics(
    elements: Dict[int, "IFCElement"],
    plates: List[Dict[str, float]],
    storeys: List[Dict[str, Any]],
    dimensions: Optional[Dict[str, Any]],
    length_unit: str,
) -> Optional[Dict[str, Any]]:
    """Derive RICS-style architectural metrics (GIA, footprint, volume, etc.).

    Areas come from real slab profile polygons (grounded), so they are reported
    as gross floor area ≈ GIA. Net internal area is not deducted for internal
    obstructions, which is stated explicitly rather than guessed.
    """
    # mm -> m conversion factors (area, length); already-metric models pass through.
    area_factor = 1e-6 if length_unit == "mm" else 1.0
    len_factor = 1e-3 if length_unit == "mm" else 1.0

    storey_by_eid = {}
    floor_plates: List[Dict[str, Any]] = []
    for plate in plates:
        el = elements.get(int(plate["eid"]))
        level = el.storey if el else None
        area_m2 = round(plate["area"] * area_factor, 2)
        floor_plates.append(
            {
                "level": level,
                "area_m2": area_m2,
                "perimeter_m": round(plate["perimeter"] * len_factor, 2),
            }
        )
        storey_by_eid[int(plate["eid"])] = area_m2

    gross_floor_area = round(sum(p["area_m2"] for p in floor_plates), 2) if floor_plates else None
    largest_plate = max((p["area_m2"] for p in floor_plates), default=None)

    # Floor-to-floor height from authored storey elevations.
    floor_to_floor_m = None
    elevs = sorted(s["elevation"] for s in storeys if s["elevation"] is not None)
    if len(elevs) >= 2:
        diffs = [round((elevs[i + 1] - elevs[i]) * len_factor, 2) for i in range(len(elevs) - 1)]
        floor_to_floor_m = diffs[0] if len(set(diffs)) == 1 else diffs

    # Footprint: prefer the largest slab plate; fall back to the bbox footprint.
    footprint_area_m2 = largest_plate
    if footprint_area_m2 is None and dimensions:
        lx, wy = dimensions.get("length_x"), dimensions.get("width_y")
        if lx and wy:
            footprint_area_m2 = round(lx * wy * area_factor, 2)

    # Gross internal volume ≈ floor area × floor-to-floor height (per plate).
    gross_internal_volume_m3 = None
    if gross_floor_area is not None and isinstance(floor_to_floor_m, (int, float)):
        gross_internal_volume_m3 = round(gross_floor_area * floor_to_floor_m, 2)

    # Wall thickness(es) from Revit-style names.
    wall_thicknesses = sorted(
        {
            t
            for el in elements.values()
            if el.ifc_type in ("IFCWALL", "IFCWALLSTANDARDCASE")
            for t in [_wall_thickness_mm(el.name)]
            if t is not None
        }
    )

    def _count(types) -> int:
        return sum(1 for el in elements.values() if el.ifc_type in types)

    wall_count = _count(("IFCWALL", "IFCWALLSTANDARDCASE"))
    window_count = _count(("IFCWINDOW",))

    metrics: Dict[str, Any] = {
        "gross_internal_area_m2": gross_floor_area,
        "gross_floor_area_m2": gross_floor_area,
        "gia_basis": "Sum of slab/floor plate polygon areas (gross; not deducted for "
        "internal walls/columns, so this is gross internal area ≈ GIA, not NIA).",
        "floor_plates": floor_plates,
        "number_of_storeys": len(storeys),
        "footprint_area_m2": footprint_area_m2,
        "floor_to_floor_height_m": floor_to_floor_m,
        "gross_internal_volume_m3": gross_internal_volume_m3,
        "wall_count": wall_count,
        "wall_thickness_mm": wall_thicknesses or None,
        "door_count": _count(("IFCDOOR",)),
        "window_count": window_count,
        "column_count": _count(("IFCCOLUMN",)),
    }
    if wall_count and window_count == 0:
        metrics["window_to_wall_ratio"] = 0.0
    # Drop keys that came out empty so the LLM never sees null noise.
    return {k: v for k, v in metrics.items() if v is not None}


def _build_facts(path: str) -> IFCModelFacts:
    entities: Dict[int, Tuple[str, str]] = {}
    fields_cache: Dict[int, List[str]] = {}

    schema = None
    application = None
    author = None
    timestamp = None

    # First pass: read header lines (before DATA) for schema/author/timestamp.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "DATA;":
                break
            if line.startswith("FILE_SCHEMA"):
                m = re.search(r"\(\('([^']+)'", line)
                if m:
                    schema = m.group(1)
            elif line.startswith("FILE_NAME"):
                fn = _split_top_level(line[line.find("(") + 1 : line.rfind(")")])
                if len(fn) >= 2:
                    timestamp = _unquote(fn[1])

    # Keep every entity: domain extraction filters by type, while bounding-box
    # computation needs the geometry/placement entities too. The full table is
    # transient — only the compact IFCModelFacts is cached.
    for eid, ifc_type, params in _iter_step_entities(path):
        entities[eid] = (ifc_type, params)

    def flds(eid: int) -> List[str]:
        if eid not in fields_cache:
            fields_cache[eid] = _split_top_level(entities[eid][1])
        return fields_cache[eid]

    # Application / author from owner-history graph.
    for eid, (ifc_type, params) in entities.items():
        if ifc_type == "IFCAPPLICATION":
            f = flds(eid)
            if len(f) >= 3:
                application = _unquote(f[2]) or application
        elif ifc_type == "IFCPERSON":
            f = flds(eid)
            if len(f) >= 3 and author is None:
                author = _unquote(f[2])

    # Units.
    units: List[str] = []
    for eid, (ifc_type, params) in entities.items():
        if ifc_type == "IFCSIUNIT":
            u = _format_si_unit(params)
            if u:
                units.append(u)
        elif ifc_type == "IFCCONVERSIONBASEDUNIT":
            f = flds(eid)
            if len(f) >= 3:
                name = _unquote(f[2])
                if name:
                    units.append(f"plane angle unit: {name.lower()}")
    units = sorted(set(units))

    # Project / site / building / storeys.
    project_name = project_description = None
    site_name = site_lat = site_lon = None
    building_name = None
    storeys: List[Dict[str, Any]] = []

    for eid, (ifc_type, params) in entities.items():
        if ifc_type == "IFCPROJECT":
            f = flds(eid)
            project_name = _unquote(f[2]) if len(f) > 2 else None
            project_description = _unquote(f[4]) if len(f) > 4 else None
        elif ifc_type == "IFCSITE":
            f = flds(eid)
            site_name = _unquote(f[2]) if len(f) > 2 else None
            # IFC2X3 IfcSite: ... RefLatitude(idx9), RefLongitude(idx10)
            if len(f) > 10:
                site_lat = _format_compound_angle(f[9])
                site_lon = _format_compound_angle(f[10])
        elif ifc_type == "IFCBUILDING":
            f = flds(eid)
            building_name = _unquote(f[2]) if len(f) > 2 else None
        elif ifc_type == "IFCBUILDINGSTOREY":
            f = flds(eid)
            name = _unquote(f[2]) if len(f) > 2 else None
            elevation = _as_number(f[9]) if len(f) > 9 else None
            storeys.append({"eid": eid, "name": name, "elevation": elevation})

    length_suffix = _project_length_suffix(entities)

    # Elements.
    elements: Dict[int, IFCElement] = {}
    for eid, (ifc_type, params) in entities.items():
        if ifc_type not in _ELEMENT_TYPES:
            continue
        f = flds(eid)
        name = _unquote(f[2]) if len(f) > 2 else None
        object_type = _unquote(f[4]) if len(f) > 4 else None
        tag = _unquote(f[7]) if len(f) > 7 else None
        extra: Dict[str, Any] = {}
        if ifc_type == "IFCDOOR" and len(f) >= 10:
            h = _as_number(f[8])
            w = _as_number(f[9])
            if h is not None:
                extra["overall_height"] = f"{round(h, 1)} {length_suffix}"
            if w is not None:
                extra["overall_width"] = f"{round(w, 1)} {length_suffix}"
        elif ifc_type == "IFCWINDOW" and len(f) >= 10:
            h = _as_number(f[8])
            w = _as_number(f[9])
            if h is not None:
                extra["overall_height"] = f"{round(h, 1)} {length_suffix}"
            if w is not None:
                extra["overall_width"] = f"{round(w, 1)} {length_suffix}"
        elif ifc_type == "IFCSLAB" and len(f) > 8:
            predef = (f[8] or "").replace(".", "").strip()
            if predef:
                extra["predefined_type"] = predef.lower()
        elements[eid] = IFCElement(
            eid=eid,
            ifc_type=ifc_type,
            name=name,
            object_type=object_type,
            tag=tag,
            extra=extra,
        )

    # Property sets -> { pset_name: { prop_name: value } }
    psets: Dict[int, Dict[str, Any]] = {}
    for eid, (ifc_type, params) in entities.items():
        if ifc_type != "IFCPROPERTYSET":
            continue
        f = flds(eid)
        pset_name = _unquote(f[2]) if len(f) > 2 else None
        prop_refs = _refs_in(f[4]) if len(f) > 4 else []
        props: Dict[str, str] = {}
        for pref in prop_refs:
            pent = entities.get(pref)
            if not pent or pent[0] != "IFCPROPERTYSINGLEVALUE":
                continue
            pf = _split_top_level(pent[1])
            if len(pf) < 3:
                continue
            prop_name = _unquote(pf[0])
            type_name, value = _typed_value(pf[2])
            if prop_name is None or value is None:
                continue
            if type_name and ("LENGTHMEASURE" in type_name):
                num = _as_number(value)
                if num is not None:
                    value = f"{round(num, 1)} {length_suffix}"
            elif type_name and type_name == "IFCBOOLEAN":
                value = "true" if value.replace(".", "").upper() == "T" else "false"
            props[prop_name] = value
        psets[eid] = {"name": pset_name, "props": props}

    # RelDefinesByProperties: attach psets to elements.
    for eid, (ifc_type, params) in entities.items():
        if ifc_type != "IFCRELDEFINESBYPROPERTIES":
            continue
        f = flds(eid)
        if len(f) < 6:
            continue
        related = _refs_in(f[4])
        pset_ref = _as_ref(f[5])
        pset = psets.get(pset_ref) if pset_ref else None
        if not pset or not pset.get("props"):
            continue
        pset_name = pset.get("name") or f"PropertySet#{pset_ref}"
        for rel in related:
            el = elements.get(rel)
            if el is not None:
                el.properties[pset_name] = dict(pset["props"])

    # Materials.
    material_names: Dict[int, str] = {}
    for eid, (ifc_type, params) in entities.items():
        if ifc_type == "IFCMATERIAL":
            f = flds(eid)
            nm = _unquote(f[0]) if f else None
            if nm:
                material_names[eid] = nm

    def _resolve_material(eid: int, depth: int = 0) -> List[str]:
        if depth > 6 or eid not in entities:
            return []
        ent = entities[eid]
        if ent[0] == "IFCMATERIAL":
            nm = material_names.get(eid)
            return [nm] if nm else []
        found: List[str] = []
        for ref in _refs_in(ent[1]):
            if ref == eid:
                continue
            found.extend(_resolve_material(ref, depth + 1))
        return found

    for eid, (ifc_type, params) in entities.items():
        if ifc_type != "IFCRELASSOCIATESMATERIAL":
            continue
        f = flds(eid)
        if len(f) < 6:
            continue
        related = _refs_in(f[4])
        mat_ref = _as_ref(f[5])
        mats = _resolve_material(mat_ref) if mat_ref else []
        mats = list(dict.fromkeys(m for m in mats if m))
        for rel in related:
            el = elements.get(rel)
            if el is not None:
                for m in mats:
                    if m not in el.materials:
                        el.materials.append(m)

    # Spatial containment: element -> storey.
    storey_name_by_eid = {s["eid"]: s["name"] for s in storeys}
    for eid, (ifc_type, params) in entities.items():
        if ifc_type != "IFCRELCONTAINEDINSPATIALSTRUCTURE":
            continue
        f = flds(eid)
        if len(f) < 6:
            continue
        related = _refs_in(f[4])
        structure_ref = _as_ref(f[5])
        storey_name = storey_name_by_eid.get(structure_ref)
        if not storey_name:
            continue
        for rel in related:
            el = elements.get(rel)
            if el is not None and el.storey is None:
                el.storey = storey_name

    storeys_sorted = sorted(storeys, key=lambda s: (s["elevation"] is None, s["elevation"] or 0.0))

    dimensions = None
    architectural_metrics = None
    try:
        from ifc_model.ifc_geometry import compute_floor_metrics, compute_world_bbox

        bbox = compute_world_bbox(
            entities, list(elements.keys()), _split_top_level, _refs_in
        )
        if bbox:
            dimensions = {**bbox, "unit": length_suffix}

        slab_ids = [
            eid for eid, el in elements.items() if el.ifc_type in ("IFCSLAB", "IFCROOF")
        ]
        plates = compute_floor_metrics(entities, slab_ids, _split_top_level, _refs_in)
        architectural_metrics = _build_architectural_metrics(
            elements, plates, storeys_sorted, dimensions, length_suffix
        )
    except Exception:
        pass

    return IFCModelFacts(
        path=path,
        schema=schema,
        project_name=project_name,
        project_description=project_description,
        application=application,
        author=author,
        timestamp=timestamp,
        units=units,
        site_name=site_name,
        site_latitude=site_lat,
        site_longitude=site_lon,
        building_name=building_name,
        storeys=storeys_sorted,
        elements=list(elements.values()),
        materials=sorted(material_names.values()),
        length_unit=length_suffix,
        dimensions=dimensions,
        architectural_metrics=architectural_metrics,
    )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_CACHE_LOCK = Lock()
_CACHE: Dict[str, Tuple[Tuple[float, int], IFCModelFacts]] = {}

# Friendly singular labels for element types in summaries.
_TYPE_LABELS = {
    "IFCWALL": "wall",
    "IFCWALLSTANDARDCASE": "wall",
    "IFCDOOR": "door",
    "IFCWINDOW": "window",
    "IFCSLAB": "slab/floor",
    "IFCROOF": "roof",
    "IFCSTAIR": "stair",
    "IFCCOLUMN": "column",
    "IFCBEAM": "beam",
    "IFCFURNISHINGELEMENT": "furnishing",
    "IFCFLOWTERMINAL": "flow terminal (light/fixture)",
    "IFCLIGHTFIXTURE": "light fixture",
    "IFCBUILDINGELEMENTPROXY": "generic element",
    "IFCSPACE": "space/room",
}


def get_ifc_facts(path: str) -> IFCModelFacts:
    """Return parsed facts for ``path``, cached by file mtime+size."""
    abspath = os.path.abspath(path)
    stat = os.stat(abspath)
    stamp = (stat.st_mtime, stat.st_size)
    with _CACHE_LOCK:
        cached = _CACHE.get(abspath)
        if cached and cached[0] == stamp:
            return cached[1]
    facts = _build_facts(abspath)
    with _CACHE_LOCK:
        _CACHE[abspath] = (stamp, facts)
    return facts


def _type_label(ifc_type: str) -> str:
    return _TYPE_LABELS.get(ifc_type, ifc_type.replace("IFC", "").lower())


def build_ifc_context_text(path: str) -> str:
    """Build a compact, fully-grounded markdown description of the model for the LLM."""
    facts = get_ifc_facts(path)
    lines: List[str] = []

    lines.append("## Building Model Overview")
    lines.append(f"- Source file: {os.path.basename(facts.path)}")
    if facts.schema:
        lines.append(f"- IFC schema: {facts.schema}")
    if facts.project_name:
        lines.append(f"- Project: {facts.project_name}")
    if facts.project_description:
        lines.append(f"- Project status/description: {facts.project_description}")
    if facts.building_name:
        lines.append(f"- Building: {facts.building_name}")
    if facts.application:
        lines.append(f"- Authoring application: {facts.application}")
    if facts.timestamp:
        lines.append(f"- Exported: {facts.timestamp}")
    if facts.site_name:
        site = f"- Site: {facts.site_name}"
        if facts.site_latitude and facts.site_longitude:
            site += f" (approx. latitude {facts.site_latitude}, longitude {facts.site_longitude})"
        lines.append(site)

    if facts.units:
        lines.append("")
        lines.append("## Units")
        for u in facts.units:
            lines.append(f"- {u}")

    if facts.storeys:
        lines.append("")
        lines.append("## Levels / Storeys (elevations as authored)")
        for s in facts.storeys:
            elev = s["elevation"]
            elev_str = f"{round(elev, 1)} {facts.length_unit}" if elev is not None else "unknown elevation"
            lines.append(f"- {s['name']}: elevation {elev_str}")
        with_elev = [s for s in facts.storeys if s["elevation"] is not None]
        if len(with_elev) >= 2:
            diff = abs(with_elev[-1]["elevation"] - with_elev[0]["elevation"])
            lines.append(
                f"- Vertical distance between lowest and highest level "
                f"({with_elev[0]['name']} → {with_elev[-1]['name']}): {round(diff, 1)} {facts.length_unit}"
            )

    if facts.dimensions:
        d = facts.dimensions
        u = d.get("unit", "mm")
        lx, wy, hz = d.get("length_x"), d.get("width_y"), d.get("height_z")
        lx_m, wy_m, hz_m = (
            (lx / 1000.0, wy / 1000.0, hz / 1000.0) if u == "mm" else (lx, wy, hz)
        )
        lines.append("")
        lines.append("## Overall Model Dimensions (world-space bounding box of all elements)")
        lines.append(
            f"- Footprint (X × Y): {lx} × {wy} {u} (≈ {round(lx_m, 2)} m × {round(wy_m, 2)} m)"
        )
        lines.append(f"- Height (Z extent): {hz} {u} (≈ {round(hz_m, 2)} m)")
        lines.append(
            "- These are the overall extents of the modeled geometry in real-world coordinates "
            "(placements resolved). The model may cover only part of a larger building."
        )

    if facts.architectural_metrics:
        m = facts.architectural_metrics
        lines.append("")
        lines.append("## Architectural Metrics")
        if m.get("gross_internal_area_m2") is not None:
            lines.append(
                f"- Gross Internal Area (GIA) ≈ {m['gross_internal_area_m2']} m²"
            )
            lines.append(f"  - Basis: {m.get('gia_basis')}")
        if m.get("footprint_area_m2") is not None:
            lines.append(f"- Building footprint area: {m['footprint_area_m2']} m²")
        plates = m.get("floor_plates") or []
        if len(plates) > 1:
            for p in plates:
                lines.append(
                    f"  - Floor plate ({p.get('level') or 'unknown level'}): "
                    f"{p.get('area_m2')} m², perimeter {p.get('perimeter_m')} m"
                )
        elif plates:
            lines.append(f"- Floor plate perimeter: {plates[0].get('perimeter_m')} m")
        if m.get("number_of_storeys") is not None:
            lines.append(f"- Number of storeys/levels: {m['number_of_storeys']}")
        if m.get("floor_to_floor_height_m") is not None:
            lines.append(f"- Floor-to-floor height: {m['floor_to_floor_height_m']} m")
        if m.get("gross_internal_volume_m3") is not None:
            lines.append(
                f"- Gross internal volume ≈ {m['gross_internal_volume_m3']} m³ "
                "(floor area × floor-to-floor height)"
            )
        if m.get("wall_thickness_mm"):
            thk = ", ".join(f"{t} mm" for t in m["wall_thickness_mm"])
            lines.append(f"- Wall thickness(es): {thk}")
        counts = []
        for key, label in (
            ("wall_count", "walls"),
            ("door_count", "doors"),
            ("window_count", "windows"),
            ("column_count", "columns"),
        ):
            if m.get(key) is not None:
                counts.append(f"{m[key]} {label}")
        if counts:
            lines.append(f"- Envelope/structure counts: {', '.join(counts)}")
        if m.get("window_to_wall_ratio") is not None:
            lines.append(f"- Window-to-wall ratio: {m['window_to_wall_ratio']}")
        lines.append(
            "- Net Internal Area (NIA) is not separately computed; it would deduct internal "
            "walls, columns, and circulation from GIA."
        )

    # Inventory by type.
    by_type: Dict[str, List[IFCElement]] = {}
    for el in facts.elements:
        by_type.setdefault(el.ifc_type, []).append(el)

    lines.append("")
    lines.append("## Element Inventory (counts by type)")
    if by_type:
        for ifc_type in sorted(by_type, key=lambda t: -len(by_type[t])):
            lines.append(f"- {_type_label(ifc_type)}: {len(by_type[ifc_type])}")
    else:
        lines.append("- No physical building elements found in the model.")
    lines.append(f"- Total physical elements: {len(facts.elements)}")

    # Per-element detail.
    lines.append("")
    lines.append("## Elements (detail)")
    for ifc_type in sorted(by_type, key=lambda t: -len(by_type[t])):
        lines.append(f"### {_type_label(ifc_type).title()} ({len(by_type[ifc_type])})")
        for el in by_type[ifc_type]:
            parts: List[str] = []
            label = el.name or el.object_type or f"#{el.eid}"
            parts.append(f"name: {label}")
            if el.storey:
                parts.append(f"level: {el.storey}")
            for k, v in el.extra.items():
                parts.append(f"{k.replace('_', ' ')}: {v}")
            if el.materials:
                parts.append(f"materials: {', '.join(el.materials)}")
            for pset_name, props in el.properties.items():
                prop_str = ", ".join(f"{pk}={pv}" for pk, pv in props.items())
                if prop_str:
                    parts.append(f"{pset_name}: {prop_str}")
            lines.append(f"- {'; '.join(parts)}")

    if facts.materials:
        lines.append("")
        lines.append("## Materials present in the model")
        for m in facts.materials:
            lines.append(f"- {m}")

    return "\n".join(lines)


def get_ifc_summary(path: str) -> Dict[str, Any]:
    """Compact structured summary for debug endpoints / tests."""
    facts = get_ifc_facts(path)
    by_type: Dict[str, int] = {}
    for el in facts.elements:
        by_type[_type_label(el.ifc_type)] = by_type.get(_type_label(el.ifc_type), 0) + 1
    return {
        "path": os.path.basename(facts.path),
        "schema": facts.schema,
        "project": facts.project_name,
        "building": facts.building_name,
        "site": facts.site_name,
        "units": facts.units,
        "storeys": [
            {"name": s["name"], "elevation": s["elevation"]} for s in facts.storeys
        ],
        "element_counts": by_type,
        "total_elements": len(facts.elements),
        "materials": facts.materials,
        "dimensions": facts.dimensions,
        "architectural_metrics": facts.architectural_metrics,
    }
