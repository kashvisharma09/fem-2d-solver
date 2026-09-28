"""Plate-with-a-hole mesh via gmsh (unstructured triangles)."""
import numpy as np

from .elements import signed_area


def plate_with_hole_mesh(W: float, H: float, r: float,
                          mesh_size_far: float, mesh_size_hole: float = None):
    """Mesh a rectangular plate [0, W] x [-H/2, H/2] with a circular hole of
    radius r centered at (W/2, 0), using gmsh's OpenCASCADE kernel. The mesh
    is refined near the hole and coarser far from it.

    Returns
    -------
    nodes : (N, 2) float array
    elements : (M, 3) int array, triangle connectivity (0-indexed, CCW)
    """
    import gmsh

    if mesh_size_hole is None:
        mesh_size_hole = mesh_size_far / 6.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("plate_with_hole")
        cx, cy = W / 2.0, 0.0

        rect = gmsh.model.occ.addRectangle(0.0, -H / 2.0, 0.0, W, H)
        hole = gmsh.model.occ.addDisk(cx, cy, 0.0, r, r)
        gmsh.model.occ.cut([(2, rect)], [(2, hole)])
        gmsh.model.occ.synchronize()

        # Refine mesh size based on distance from the hole center.
        radius_field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            radius_field, "F", f"sqrt((x-({cx}))^2 + (y-({cy}))^2)"
        )

        thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh, "InField", radius_field)
        gmsh.model.mesh.field.setNumber(thresh, "SizeMin", mesh_size_hole)
        gmsh.model.mesh.field.setNumber(thresh, "SizeMax", mesh_size_far)
        gmsh.model.mesh.field.setNumber(thresh, "DistMin", r)
        gmsh.model.mesh.field.setNumber(thresh, "DistMax", r + 4 * mesh_size_far)
        gmsh.model.mesh.field.setAsBackgroundMesh(thresh)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        elements = []
        for etype, nodetags in zip(elem_types, elem_node_tags):
            if etype == 2:  # 3-node triangle
                tri = np.array(nodetags).reshape(-1, 3)
                for t in tri:
                    elements.append([tag_to_index[n] for n in t])

        nodes = node_coords
        elements = np.array(elements, dtype=int)

        # Ensure every triangle is counter-clockwise (flip if needed).
        for i, tri in enumerate(elements):
            if signed_area(nodes[tri]) < 0:
                elements[i] = elements[i][[0, 2, 1]]

        return nodes, elements
    finally:
        gmsh.finalize()