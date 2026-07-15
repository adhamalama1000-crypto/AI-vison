"""
Electrical graph construction.

Turns a set of detected components, terminals and wires into a graph:

* **Nodes** — one per component (``kind="component"``) and one per terminal
  (``kind="terminal"``). Connection points that don't belong to a component are
  represented as free terminal nodes.
* **Edges** — one per wire, linking the node its start connects to with the node
  its end connects to (terminal preferred, else component). Each edge carries
  the wire's colour, length and endpoints.

The graph is JSON-serialisable and stored both denormalised (nodes/edges blobs
on ``reference_graph``) and normalised (``reference_connections``). It is the
canonical structure the comparison stage diffs.
"""

from __future__ import annotations

from typing import Any, Optional


def build_graph(components: list[dict], terminals: list[dict],
                wires: list[dict]) -> dict[str, Any]:
    """Build a serialisable electrical graph from element dicts.

    ``components`` items need at least ``ref_id`` and (``cx``,``cy``) or
    ``bbox``. ``terminals`` need ``ref_id``,``x``,``y``. ``wires`` need
    ``wire_uid`` and the ``from_*``/``to_*`` snap fields.
    """
    nodes: list[dict] = []
    node_index: dict[str, dict] = {}

    for c in components:
        ref = c.get("ref_id") or c.get("label")
        if ref is None:
            continue
        cx, cy = _center(c)
        node = {"id": ref, "kind": "component", "type": c.get("comp_type") or c.get("label"),
                "label": c.get("label"), "x": cx, "y": cy}
        nodes.append(node)
        node_index[ref] = node

    for t in terminals:
        ref = t.get("ref_id")
        if ref is None:
            continue
        node = {"id": ref, "kind": "terminal", "type": t.get("kind"),
                "label": t.get("label"), "x": t.get("x"), "y": t.get("y"),
                "component": t.get("component_ref")}
        nodes.append(node)
        node_index[ref] = node

    edges: list[dict] = []
    for wnode in wires:
        uid = wnode.get("wire_uid")
        frm = wnode.get("from_terminal") or wnode.get("from_component")
        to = wnode.get("to_terminal") or wnode.get("to_component")
        edges.append({
            "id": uid, "kind": "wire", "from": frm, "to": to,
            "from_terminal": wnode.get("from_terminal"),
            "to_terminal": wnode.get("to_terminal"),
            "from_component": wnode.get("from_component"),
            "to_component": wnode.get("to_component"),
            "color": wnode.get("color"), "length": wnode.get("length"),
            "start": wnode.get("start"), "end": wnode.get("end"),
        })
    return {
        "nodes": nodes, "edges": edges,
        "node_count": len(nodes), "edge_count": len(edges),
        "component_count": sum(1 for n in nodes if n["kind"] == "component"),
        "terminal_count": sum(1 for n in nodes if n["kind"] == "terminal"),
    }


def connections(graph: dict) -> list[dict]:
    """Flatten graph edges to normalized connection rows."""
    out = []
    for e in graph.get("edges", []):
        out.append({
            "wire_uid": e.get("id"), "from_node": e.get("from"),
            "to_node": e.get("to"), "from_terminal": e.get("from_terminal"),
            "to_terminal": e.get("to_terminal"), "color": e.get("color"),
        })
    return out


def _center(c: dict) -> tuple[Optional[float], Optional[float]]:
    if c.get("cx") is not None and c.get("cy") is not None:
        return float(c["cx"]), float(c["cy"])
    bbox = c.get("bbox")
    if bbox and len(bbox) == 4:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return None, None


def adjacency(graph: dict) -> dict[str, set]:
    """Undirected adjacency map node_id -> set(neighbour node_ids)."""
    adj: dict[str, set] = {n["id"]: set() for n in graph.get("nodes", [])}
    for e in graph.get("edges", []):
        a, b = e.get("from"), e.get("to")
        if a is not None:
            adj.setdefault(a, set())
        if b is not None:
            adj.setdefault(b, set())
        if a is not None and b is not None:
            adj[a].add(b)
            adj[b].add(a)
    return adj
