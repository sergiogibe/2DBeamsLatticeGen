import numpy as np
import gmsh, json, sys
import tkinter as tk
from tkinter import messagebox
from collections import OrderedDict
from typing import Iterable
from collections import defaultdict
import scipy


def _qkey_xyz(x, y, z, tol):
    """Quantize coords so points within tol share a key."""
    if tol <= 0:
        return (x, y, z)
    s = 1.0 / tol
    return (round(x * s), round(y * s), round(z * s))

def _point_xyz(pt_tag):
    # For a point, bbox min == max == (x,y,z)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(0, pt_tag)
    return xmax, ymax, zmax

def find_duplicate_geom_points(tol=1e-9):
    """
    Return a list of {'coord': (x,y,z), 'points': [pt_tags]} for groups with >1 points.
    Call AFTER gmsh.model.occ.synchronize().
    """
    groups = defaultdict(list)
    pts = gmsh.model.getEntities(0)  # [(0, pt_tag), ...]
    for _, ptag in pts:
        x, y, z = _point_xyz(ptag)
        groups[_qkey_xyz(x, y, z, tol)].append(ptag)

    dups = []
    for key, tags in groups.items():
        if len(tags) > 1:
            # representative coord (de-quantize best-effort)
            x, y, z = _point_xyz(tags[0])
            dups.append({'coord': (x, y, z), 'points': sorted(tags)})
    return dups

def find_duplicate_geom_lines(tol=1e-9):
    """
    Return a list of {'endpoints': ((x1,y1,z1),(x2,y2,z2)), 'lines': [line_tags]}
    for groups of lines that share the same pair of endpoint coordinates (order-independent).
    Assumes straight OCC lines created with occ.addLine (two endpoints).
    """
    groups = defaultdict(list)
    lines = gmsh.model.getEntities(1)  # [(1, line_tag), ...]
    for _, ltag in lines:
        # get boundary points (two endpoints)
        bnd = gmsh.model.getBoundary([(1, ltag)], oriented=False, recursive=False)
        ptags = [t for dim, t in bnd if dim == 0]
        if len(ptags) != 2:
            continue  # skip composite curves etc.
        x1, y1, z1 = _point_xyz(ptags[0])
        x2, y2, z2 = _point_xyz(ptags[1])
        k1 = _qkey_xyz(x1, y1, z1, tol)
        k2 = _qkey_xyz(x2, y2, z2, tol)
        key = tuple(sorted((k1, k2)))  # order-independent
        groups[key].append(ltag)

    dups = []
    for key, ltags in groups.items():
        if len(ltags) > 1:
            # reconstruct representative endpoint coords
            (kx1, ky1, kz1), (kx2, ky2, kz2) = key
            # These are quantized; for display we can fetch from one line's real endpoints:
            l0 = ltags[0]
            bnd = gmsh.model.getBoundary([(1, l0)], oriented=False, recursive=False)
            p0, p1 = [t for dim, t in bnd if dim == 0]
            ep1 = _point_xyz(p0); ep2 = _point_xyz(p1)
            # sort for consistent output
            if ep2 < ep1: ep1, ep2 = ep2, ep1
            dups.append({'endpoints': (ep1, ep2), 'lines': sorted(ltags)})
    return dups

def coords_by_tag(tag: int):
    """Return (x,y,z) for a Gmsh *node tag* (tags are 1-based)."""
    # Ensure mesh exists
    # gmsh.model.mesh.generate(1)  # <- already done earlier in your script

    tags, coords, _ = gmsh.model.mesh.getNodes()
    # Build a fast tag->index map once and cache it on the function
    cache = getattr(coords_by_tag, "_cache", None)
    if not cache or cache["n"] != len(tags):
        tag2i = {int(t): i for i, t in enumerate(tags)}
        coords_by_tag._cache = {"map": tag2i, "n": len(tags)}
    else:
        tag2i = cache["map"]

    t = int(tag)
    if t not in tag2i:
        raise ValueError(
            f"Node tag {t} not found. "
            "Did you pass a 0-based index instead of a Gmsh tag, "
            "or did tags change after boolean ops/cleanup?"
        )
    i = tag2i[t]
    x = float(coords[3*i])
    y = float(coords[3*i+1])
    z = float(coords[3*i+2])
    return x, y, z

def group_indices_by_tag(indices: Iterable[int], tags:    Iterable[int]):
    """
    Returns:
      groups      -> [[indices with tag1], [indices with tag2], ...]
      group_tags  -> [tag1, tag2, ...] in order of first appearance
      groups_dict -> {tag: [indices], ...}

    """
    idx_iter = list(indices)
    tag_iter = list(tags)
    if len(idx_iter) != len(tag_iter):
        raise ValueError("indices and tags must have the same length")

    groups = OrderedDict()  
    for i, t in zip(idx_iter, tag_iter):
        groups.setdefault(t, []).append(i)

    group_tags = list(groups.keys())
    grouped_indices = list(groups.values())
    return grouped_indices, group_tags, dict(groups)

def ask_yes_no(title="Save mesh", message="Do you want to save the mesh?") -> bool:
    root = tk.Tk()
    root.withdraw()                 # hide main window
    root.attributes("-topmost", True)
    try:
        result = messagebox.askyesno(title, message, parent=root)
    finally:
        root.destroy()
    return bool(result)

def find_node_gmsh(node_tags: np.ndarray, node_coords: np.ndarray, x: float, y:float , z:float = 0.0, tol: float = 1e-6):
    target = np.array([x, y, z])
    node_found_tag, node_found_coord = None, None
    for i, tag in enumerate(node_tags):
        coord = np.array([
            node_coords[3 * i],
            node_coords[3 * i + 1],
            node_coords[3 * i + 2]
        ])
        if np.linalg.norm(coord - target) < tol:
            node_found_tag = tag
            node_found_coord = coord
            break
    return node_found_tag, node_found_coord

def mesher_beams1D_OLD(case: str, 
                   fimport: str, 
                   fexport: str = None, 
                   ui: bool = True):
    """
    Mesher utilizing Gmsh reading a cell case in cases.json.

    |It generates a mesh of 1D elements.

    |Provides tesselation considering the lattice vectors directions.

    |By defing physical groups one can generate periodic boundary conditions,
    specifying the nodes (done in the GUI). Generates the condenser matrix. (still needs implementation)

    8/6/2025 - S. Britto

    """

    '''Load setup from .json file'''
    try:
        with open(fimport) as json_file:
            setup = json.load(json_file)[case]
    except KeyError:
        print("No setup found for this case. Please check cells.json. ")

    lc                      = setup["lc"]
    rescale_factor          = setup["rescale_factor"]
    ncels_lv1: int          = setup["n_cells"][0]
    ncels_lv2: int          = setup["n_cells"][1]
    n_nodes: int            = setup["n_nodes"]
    n_connections: int      = setup["n_connections"]
    nodes: list             = [[coord * rescale_factor for coord in node] for node in setup["nodes"]]
    connections: list       = setup["connections"]
    lattice_vectors: list   = setup["lattice_vectors"]

    '''Initialize Gmsh and creating cell using OCC kernel'''
    gmsh.initialize()
    gmsh.model.add(case)
    factory = gmsh.model.occ
    factory_points = []
    factory_connections = []
    factory_connection_type = []
    for node in nodes:
        factory_points.append(factory.addPoint(node[0], node[1], 0.0, lc))
    for connection in connections:
        factory_connections.append(factory.addLine(factory_points[connection[0]], factory_points[connection[1]]))
        factory_connection_type.append(connection[2])
    factory.synchronize()

    '''Set connection type as physical groups'''
    group_pg, group_pgt, _ = group_indices_by_tag(indices=factory_connections, tags=factory_connection_type)
    g_index = 0
    for g in group_pg:
        gmsh.model.addPhysicalGroup(1, g, tag=group_pgt[g_index])
        g_index += 1
    
    factory.synchronize()
    gmsh.model.setColor([(1, t) for t in group_pg[0]], 255, 0, 0) 
    gmsh.model.setColor([(1, t) for t in group_pg[1]], 0, 255, 0) 

    '''Tessellation'''
    dx1 = nodes[lattice_vectors[0][1]][0] - nodes[lattice_vectors[0][0]][0]
    dy1 = nodes[lattice_vectors[0][1]][1] - nodes[lattice_vectors[0][0]][1]
    dx2 = nodes[lattice_vectors[1][1]][0] - nodes[lattice_vectors[1][0]][0]
    dy2 = nodes[lattice_vectors[1][1]][1] - nodes[lattice_vectors[1][0]][1]

    all_connections = factory_connections[:]
    for i in range(ncels_lv1):
        for j in range(ncels_lv2):
            if i == 0 and j == 0:
                continue
            dx_i = i * dx1 + j * dx2
            dy_j = i * dy1 + j * dy2
            new_cell = factory.copy([(1, connection) for connection in factory_connections])
            factory.translate(new_cell, dx=dx_i, dy=dy_j, dz=0)
            all_connections.extend([connection[1] for connection in new_cell])

    '''Check duplicates'''
    factory.synchronize()
    points_total = gmsh.model.getEntities(dim=0)
    lines_total = gmsh.model.getEntities(dim=1)
    n_points_total = len(points_total)
    n_lines_total = len(lines_total)
    expected_points_total = (ncels_lv1 * ncels_lv2) * n_nodes
    expected_lines_total = (ncels_lv1 * ncels_lv2) * n_connections
    gmsh.model.occ.removeAllDuplicates()
    factory.synchronize()
    points_total_after_clean = gmsh.model.getEntities(dim=0)
    lines_total_after_clean = gmsh.model.getEntities(dim=1)
    n_points_total_after_clean = len(points_total_after_clean)
    n_lines_total_after_clean = len(lines_total_after_clean)
    if n_points_total == n_points_total_after_clean and n_lines_total == n_lines_total_after_clean:
        if ncels_lv1 == 1 and ncels_lv2 == 1:
            pass
        else:
            print("WARNING: It seems that the tesselated geometry has duplicate points or lines! ")
            sys.exit()

    '''Mesh generation'''
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc/1.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc*1.5)
    gmsh.model.mesh.generate(1)

    '''Bloch analysis (w/ some comments)'''
    factory.synchronize()
    mesh_node_tags, mesh_node_coords, _ = gmsh.model.mesh.getNodes()
    bloch_permutation_vector = mesh_node_tags
    n_mesh_nodes = len(mesh_node_tags)
    bloch_mode: str = None

    if ncels_lv1 == 1 and ncels_lv2 == 1:
        bloch_mode = "cell"
        prime_nodes, connected_nodes = [], []
        for lv in range(2):
            prime_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, x=nodes[lattice_vectors[lv][0]][0], y=nodes[lattice_vectors[lv][0]][1])[0])
            connected_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, x=nodes[lattice_vectors[lv][1]][0], y=nodes[lattice_vectors[lv][1]][1])[0])

    elif ncels_lv1 > 1 and ncels_lv2 == 1:
        bloch_mode = "supercellx"
        lv = 1  # always second lattice vector used in this case
        prime_nodes, connected_nodes = [], []
        for i in range(ncels_lv1):
            prime_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                x=dx1*i + nodes[lattice_vectors[lv][0]][0], 
                                                y=dy1*i + nodes[lattice_vectors[lv][0]][1])[0])
            connected_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                    x=dx1*i + nodes[lattice_vectors[lv][1]][0], 
                                                    y=dy1*i + nodes[lattice_vectors[lv][1]][1])[0])
    elif ncels_lv2 > 1 and ncels_lv1 == 1:
        bloch_mode = "supercelly"
        lv = 0  # always first lattice vector used in this case
        prime_nodes, connected_nodes = [], []
        for i in range(ncels_lv2):
            prime_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                x=dx2*i + nodes[lattice_vectors[lv][0]][0], 
                                                y=dy2*i + nodes[lattice_vectors[lv][0]][1])[0])
            connected_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                    x=dx2*i + nodes[lattice_vectors[lv][1]][0], 
                                                    y=dy2*i + nodes[lattice_vectors[lv][1]][1])[0])

    elif ncels_lv1 > 1 and ncels_lv2 > 1:
        bloch_mode = "fullscale"
        prime_nodes, connected_nodes = [], []

    else:
        print("Invalid number of cells during tessellation (check if you didn't enter either 0 or negative). ")
        sys.exit()

    # Create the list of permutation considering the prime nodes, then internal, then connected
    bloch_permutation_vector = [x for x in bloch_permutation_vector if x not in prime_nodes] 
    bloch_permutation_vector = [x for x in bloch_permutation_vector if x not in connected_nodes] 
    bloch_permutation_vector[:0] = prime_nodes
    bloch_permutation_vector.extend(connected_nodes)

    # Correct the values to Python (substract 1 from all the list)
    for i in range(len(bloch_permutation_vector)):
        bloch_permutation_vector[i] -= 1 

    # Create invert permutation and correct dof (= 3 for Timoshenko beams -> x, y, theta)
    dof = 3
    bloch_inv_permutation_vector = []
    for i in range(n_mesh_nodes):
        bloch_inv_permutation_vector.append(bloch_permutation_vector.index(i))
    bloch_permutation_vector_dof3 = [-1 for j in range(n_mesh_nodes * dof)]
    bloch_inv_permutation_vector_dof3 = [-1 for j in range(n_mesh_nodes * dof)]
    for i in range(n_mesh_nodes):
        bloch_permutation_vector_dof3[dof * i]     = dof * bloch_permutation_vector[i]
        bloch_permutation_vector_dof3[dof * i + 1] = dof * bloch_permutation_vector[i] + 1
        bloch_permutation_vector_dof3[dof * i + 2] = dof * bloch_permutation_vector[i] + 2
        bloch_inv_permutation_vector_dof3[dof * i] = dof * bloch_inv_permutation_vector[i]
        bloch_inv_permutation_vector_dof3[dof * i + 1] = dof * bloch_inv_permutation_vector[i] + 1
        bloch_inv_permutation_vector_dof3[dof * i + 2] = dof * bloch_inv_permutation_vector[i] + 2

    n_nodes_prime = len(prime_nodes)
    n_nodes_connected = len(connected_nodes)
    n_nodes_internal = n_mesh_nodes - n_nodes_prime - n_nodes_connected
    nnt = [n_nodes_prime, n_nodes_internal, n_nodes_connected]
    print(f"Done connecting bloch nodes for the {bloch_mode} case! ")

    '''Visibility options'''
    gmsh.option.setNumber("Geometry.Points", 1)
    gmsh.option.setNumber("Geometry.PointSize", 12)
    gmsh.option.setNumber("Geometry.Lines", 0)
    gmsh.option.setNumber("Geometry.LineWidth", 1)
    gmsh.option.setNumber("Mesh.Points", 1)
    gmsh.option.setNumber("Mesh.PointSize", 8)
    gmsh.option.setNumber("Mesh.Lines", 1)
    gmsh.option.setNumber("Mesh.LineWidth", 2)

    if bloch_mode == "cell":
        view_tag = gmsh.view.add("Arrows")
        x1, y1, z1 = coords_by_tag(prime_nodes[0])
        gmsh.view.addListData(view_tag, "VP", 1, [x1, y1, z1, dx1, dy1, 0.0])
        view_tag = gmsh.view.add("Arrows")
        x2, y2, z2 = coords_by_tag(prime_nodes[1])
        gmsh.view.addListData(view_tag, "VP", 1, [x2, y2, z2, dx2, dy2, 0.0])

    elif bloch_mode == "supercellx":
        for i in range(ncels_lv1):
            view_tag = gmsh.view.add("Arrows")
            x1, y1, z1 = coords_by_tag(prime_nodes[i])
            gmsh.view.addListData(view_tag, "VP", 1, [x1, y1, z1, dx2, dy2, 0.0])

    elif bloch_mode == "supercelly":
        for i in range(ncels_lv2):
            view_tag = gmsh.view.add("Arrows")
            x1, y1, z1 = coords_by_tag(prime_nodes[i])
            gmsh.view.addListData(view_tag, "VP", 1, [x1, y1, z1, dx1, dy1, 0.0])

    elif bloch_mode == "fullscale":
        pass
   
    if ui:
        gmsh.fltk.run()

    '''Export mesh file'''
    save = ask_yes_no("Export mesh", "Export Medit/INRIA .mesh now?")
    if save:
        gmsh.option.setNumber("Mesh.SaveAll", 1)     
        gmsh.write(f"{str(fexport)}/output.mesh") 

    '''Close Gmsh'''
    gmsh.finalize()

def mesher_beams1D(case: str, fimport: str, fexport: str = None, ui: bool = True):
    """
    Mesher utilizing Gmsh reading a cell case in cases.json.

    |It generates a mesh of 1D elements.

    |Provides tesselation considering the lattice vectors directions.

    |By defing physical groups one can generate periodic boundary conditions,
    specifying the nodes (done in the GUI). Generates the condenser matrix. (still needs implementation)

    8/12/2025 - S. Britto

    """

    '''Load setup from .json file'''
    try:
        with open(fimport) as json_file:
            setup = json.load(json_file)[case]
    except KeyError:
        print("No setup found for this case. Please check cells.json. ")

    lc                      = setup["lc"]
    rescale_factor          = setup["rescale_factor"]
    ncels_lv1: int          = setup["n_cells"][0]
    ncels_lv2: int          = setup["n_cells"][1]
    n_nodes: int            = setup["n_nodes"]
    n_connections: int      = setup["n_connections"]
    nodes: list             = [[coord * rescale_factor for coord in node] for node in setup["nodes"]]
    connections: list       = setup["connections"]
    lattice_vectors: list   = setup["lattice_vectors"]

    '''Initialize Gmsh and creating cell using OCC kernel'''
    gmsh.initialize()
    gmsh.model.add(case)
    factory = gmsh.model.occ
    factory_points = []
    factory_connections = []
    factory_connection_type = []
    for node in nodes:
        factory_points.append(factory.addPoint(node[0], node[1], 0.0, lc))

    for connection in connections:
        L = factory.addLine(factory_points[connection[0]], factory_points[connection[1]])
        factory_connections.append(L)
        factory_connection_type.append(int(connection[2]))  # ensure int tags
    factory.synchronize()

    type_to_base_idxs = {}
    for k, t in enumerate(factory_connection_type):
        type_to_base_idxs.setdefault(t, []).append(k)

    type_to_lines = {t: [factory_connections[i] for i in idxs]
                    for t, idxs in type_to_base_idxs.items()}

    '''Tessellation'''
    dx1 = nodes[lattice_vectors[0][1]][0] - nodes[lattice_vectors[0][0]][0]
    dy1 = nodes[lattice_vectors[0][1]][1] - nodes[lattice_vectors[0][0]][1]
    dx2 = nodes[lattice_vectors[1][1]][0] - nodes[lattice_vectors[1][0]][0]
    dy2 = nodes[lattice_vectors[1][1]][1] - nodes[lattice_vectors[1][0]][1]

    all_connections = factory_connections[:]
    base_dimtags = [(1, L) for L in factory_connections]

    for i in range(ncels_lv1):
        for j in range(ncels_lv2):
            if i == 0 and j == 0:
                continue
            dx_i = i * dx1 + j * dx2
            dy_j = i * dy1 + j * dy2

            # Copy returns lines in the SAME ORDER as base
            new_cell = factory.copy(base_dimtags)
            factory.translate(new_cell, dx=dx_i, dy=dy_j, dz=0.0)
            new_line_tags = [dt[1] for dt in new_cell]
            all_connections.extend(new_line_tags)

            # Propagate per-type membership using base indices
            for t, idxs in type_to_base_idxs.items():
                for k in idxs:
                    type_to_lines[t].append(new_line_tags[k])

    # Clean duplicates, then create Physical Groups once (final tags)
    factory.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    factory.synchronize()

    # Some tags may have been merged; filter to existing lines
    existing_lines = {t for (_, t) in gmsh.model.getEntities(1)}
    for t in list(type_to_lines.keys()):
        type_to_lines[t] = sorted({L for L in type_to_lines[t] if L in existing_lines})

    for t, ltags in type_to_lines.items():
        if not ltags:
            continue
        gmsh.model.addPhysicalGroup(1, ltags, tag=int(t))
        gmsh.model.setPhysicalName(1, int(t), f"type:{t}")

    palette = [(230,57,70), (29,53,87), (69,123,157), (131,197,190), (168,218,220), (241,143,1), (99,199,77), (156,39,176), (255,111,0)]
    for i, t in enumerate(sorted(type_to_lines.keys())):
        r,g,b = palette[i % len(palette)]
        gmsh.model.setColor([(1, L) for L in type_to_lines[t]], r, g, b)

    '''Check duplicates'''
    factory.synchronize()  
    dup_pts = find_duplicate_geom_points()
    dup_lin = find_duplicate_geom_lines()
    if dup_pts:
        print(f"WARNING: Duplicate geometry points: {len(dup_pts)} groups")
        for g in dup_pts:
            print("  at", g['coord'], "points:", g['points'])
        sys.exit()
    
    if dup_lin:
        print(f"WARNING: Duplicate geometry lines: {len(dup_lin)} groups")
        for g in dup_lin:
            print("  between", g['endpoints'], "lines:", g['lines'])
        sys.exit()

    '''Mesh generation'''
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc/1.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc*1.5)
    gmsh.model.mesh.generate(1)

    '''Bloch analysis (w/ some comments)'''
    factory.synchronize()
    mesh_node_tags, mesh_node_coords, _ = gmsh.model.mesh.getNodes()
    bloch_permutation_vector = mesh_node_tags
    n_mesh_nodes = len(mesh_node_tags)
    bloch_mode: str = None

    if ncels_lv1 == 1 and ncels_lv2 == 1:
        bloch_mode = "cell"
        prime_nodes, connected_nodes = [], []
        for lv in range(2):
            prime_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, x=nodes[lattice_vectors[lv][0]][0], y=nodes[lattice_vectors[lv][0]][1])[0])
            connected_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, x=nodes[lattice_vectors[lv][1]][0], y=nodes[lattice_vectors[lv][1]][1])[0])

    elif ncels_lv1 > 1 and ncels_lv2 == 1:
        bloch_mode = "supercellx"
        lv = 1  # always second lattice vector used in this case
        prime_nodes, connected_nodes = [], []
        for i in range(ncels_lv1):
            prime_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                x=dx1*i + nodes[lattice_vectors[lv][0]][0], 
                                                y=dy1*i + nodes[lattice_vectors[lv][0]][1])[0])
            connected_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                    x=dx1*i + nodes[lattice_vectors[lv][1]][0], 
                                                    y=dy1*i + nodes[lattice_vectors[lv][1]][1])[0])
    elif ncels_lv2 > 1 and ncels_lv1 == 1:
        bloch_mode = "supercelly"
        lv = 0  # always first lattice vector used in this case
        prime_nodes, connected_nodes = [], []
        for i in range(ncels_lv2):
            prime_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                x=dx2*i + nodes[lattice_vectors[lv][0]][0], 
                                                y=dy2*i + nodes[lattice_vectors[lv][0]][1])[0])
            connected_nodes.append(find_node_gmsh(mesh_node_tags, mesh_node_coords, 
                                                    x=dx2*i + nodes[lattice_vectors[lv][1]][0], 
                                                    y=dy2*i + nodes[lattice_vectors[lv][1]][1])[0])

    elif ncels_lv1 > 1 and ncels_lv2 > 1:
        bloch_mode = "fullscale"
        prime_nodes, connected_nodes = [], []

    else:
        print("Invalid number of cells during tessellation (check if you didn't enter either 0 or negative). ")
        sys.exit()

    # Create the list of permutation considering the prime nodes, then internal, then connected
    bloch_permutation_vector = [x for x in bloch_permutation_vector if x not in prime_nodes] 
    bloch_permutation_vector = [x for x in bloch_permutation_vector if x not in connected_nodes] 
    bloch_permutation_vector[:0] = prime_nodes
    bloch_permutation_vector.extend(connected_nodes)

    # Correct the values to Python (substract 1 from all the list)
    for i in range(len(bloch_permutation_vector)):
        bloch_permutation_vector[i] -= 1 

    # Create invert permutation and correct dof (= 3 for Timoshenko beams -> x, y, theta)
    dof = 3
    bloch_inv_permutation_vector = []
    for i in range(n_mesh_nodes):
        bloch_inv_permutation_vector.append(bloch_permutation_vector.index(i))
    bloch_permutation_vector_dof3 = [-1 for j in range(n_mesh_nodes * dof)]
    bloch_inv_permutation_vector_dof3 = [-1 for j in range(n_mesh_nodes * dof)]
    for i in range(n_mesh_nodes):
        bloch_permutation_vector_dof3[dof * i]     = dof * bloch_permutation_vector[i]
        bloch_permutation_vector_dof3[dof * i + 1] = dof * bloch_permutation_vector[i] + 1
        bloch_permutation_vector_dof3[dof * i + 2] = dof * bloch_permutation_vector[i] + 2
        bloch_inv_permutation_vector_dof3[dof * i] = dof * bloch_inv_permutation_vector[i]
        bloch_inv_permutation_vector_dof3[dof * i + 1] = dof * bloch_inv_permutation_vector[i] + 1
        bloch_inv_permutation_vector_dof3[dof * i + 2] = dof * bloch_inv_permutation_vector[i] + 2

    n_nodes_prime = len(prime_nodes)
    n_nodes_connected = len(connected_nodes)
    n_nodes_internal = n_mesh_nodes - n_nodes_prime - n_nodes_connected
    nnt = [n_nodes_prime, n_nodes_internal, n_nodes_connected]
    print(f"Done connecting bloch nodes for the {bloch_mode} case! ")

    '''Visibility options'''
    gmsh.option.setNumber("Geometry.Points", 1)
    gmsh.option.setNumber("Geometry.PointSize", 13)
    gmsh.option.setNumber("Geometry.Lines", 0)
    gmsh.option.setNumber("Geometry.LineWidth", 1)
    gmsh.option.setNumber("Mesh.Points", 1)
    gmsh.option.setNumber("Mesh.PointSize", 11)
    gmsh.option.setNumber("Mesh.Lines", 1)
    gmsh.option.setNumber("Mesh.LineWidth", 5)

    if bloch_mode == "cell":
        view_tag = gmsh.view.add("Arrows")
        x1, y1, z1 = coords_by_tag(prime_nodes[0])
        gmsh.view.addListData(view_tag, "VP", 1, [x1, y1, z1, dx1, dy1, 0.0])
        view_tag = gmsh.view.add("Arrows")
        x2, y2, z2 = coords_by_tag(prime_nodes[1])
        gmsh.view.addListData(view_tag, "VP", 1, [x2, y2, z2, dx2, dy2, 0.0])

    elif bloch_mode == "supercellx":
        for i in range(ncels_lv1):
            view_tag = gmsh.view.add("Arrows")
            x1, y1, z1 = coords_by_tag(prime_nodes[i])
            gmsh.view.addListData(view_tag, "VP", 1, [x1, y1, z1, dx2, dy2, 0.0])

    elif bloch_mode == "supercelly":
        for i in range(ncels_lv2):
            view_tag = gmsh.view.add("Arrows")
            x1, y1, z1 = coords_by_tag(prime_nodes[i])
            gmsh.view.addListData(view_tag, "VP", 1, [x1, y1, z1, dx1, dy1, 0.0])

    elif bloch_mode == "fullscale":
        pass
   
    if ui:
        gmsh.fltk.run()

    '''Export mesh file'''
    save = ask_yes_no("Export mesh", "Export Medit/INRIA .mesh now?")
    if save:
        gmsh.option.setNumber("Mesh.SaveAll", 1)     
        gmsh.write(f"{str(fexport)}/output.mesh") 
        scipy.io.savemat(f'{str(fexport)}/bloch.mat', {
            'PERMUT': np.array(bloch_permutation_vector_dof3) + 1,
            'IPERMUT': np.array(bloch_inv_permutation_vector_dof3) + 1,
            'NNT': nnt}
            )

    '''Close Gmsh'''
    gmsh.finalize()







