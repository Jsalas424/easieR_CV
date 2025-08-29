# isotropic_remeshing.py
# Standalone script for isotropic remeshing and cleaning of a 3D triangular mesh using PyMeshLab.
# Saves BOTH:
#   - PLY (portable to R via Rvcg/rgl, and back to Python)
#   - NPZ (fast Python reload: V, F arrays)
#
# Runtime requirements:
#   - Python >= 3.13.x
#   - pymeshlab >= 2025.0
#
# Python deps: pymeshlab>=2025.0, numpy, pandas, plotly
#
# Usage:
#   CSV input:
#     python isotropic_remeshing.py --input-points points.csv --input-faces faces.csv
#   PLY input:
#     python isotropic_remeshing.py --input-ply mesh.ply
#   With custom output prefix:
#     python isotropic_remeshing.py --input-ply mesh.ply --output-prefix my_mesh
#
# Outputs (auto):
#   Original:
#     - {prefix}_original.ply
#     - {prefix}_original.npz   (keys: V, F; 0-indexed)
#   Remeshed:
#     - {prefix}_remeshed_points.csv, {prefix}_remeshed_faces.csv (faces 1-indexed)
#     - {prefix}_remeshed.ply
#     - {prefix}_remeshed.npz   (keys: V, F; 0-indexed)
#
# Notes:
#   - Absolute quantities are passed using pymeshlab.PureValue (2025+ API).
#   - Surface distance checking is DISABLED by default (checksurfdist=False) to speed up remeshing.
#   - IMPORTANT (default behavior): we PRUNE small disconnected components ("tiny islands").
#       This WILL remove intentionally separate structures smaller than the threshold.
#       To keep all components someday, set MIN_COMPONENT_FACES = 0 (or comment out that filter).

from __future__ import annotations
import sys, re, argparse, pathlib, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymeshlab
from pymeshlab import PureValue  # absolute quantities in 2025+ API

# ----------- Defaults you can tune -----------
# Target remeshing parameters (surface-distance check disabled by default for speed)
TARGET_EDGE_LENGTH = 3.0
TARGET_AREA = 3.0  # Default target area for equilateral triangle
ITERATIONS = 5
FEATURE_ANGLE_DEG = 90.0
MAX_SURF_DIST = 2.0
SURF_DIST_CHECK = False
ADAPTIVE = False

# CLEANING: prune tiny disconnected components by face count.
#   ⚠️ This WILL remove intentionally separate structures smaller than this threshold.
#   To keep everything, set to 0.
MIN_COMPONENT_FACES = 25

# ----------- Environment guards -----------
def _version_tuple(v: str) -> tuple[int, int, int]:
    nums = re.findall(r"\d+", v)
    major = int(nums[0]) if len(nums) > 0 else 0
    minor = int(nums[1]) if len(nums) > 1 else 0
    patch = int(nums[2]) if len(nums) > 2 else 0
    return (major, minor, patch)

if sys.version_info < (3, 13):
    raise RuntimeError("This script requires Python >= 3.13.x")
# Require 2025+ API by feature, not string parsing
try:
    from pymeshlab import PureValue  # noqa: F401
except Exception:
    ver = getattr(pymeshlab, "__version__", "unknown")
    raise RuntimeError(
        f"This script requires PyMeshLab >= 2025.0 (PureValue not found). Detected version: {ver}"
    )


# ----------- I/O helpers -----------
def ensure_results_directory():
    """Create results directory if it doesn't exist and return the path."""
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def calculate_edge_length_from_area(target_area):
    """Calculate edge length of equilateral triangle from target area.
    
    For an equilateral triangle with side length s:
    Area = (s² * sqrt(3)) / 4
    
    Solving for s:
    s = sqrt((4 * Area) / sqrt(3))
    
    Args:
        target_area: Desired area of equilateral triangle
    
    Returns:
        Edge length for equilateral triangle with given area
    """
    import math
    return math.sqrt((4.0 * target_area) / math.sqrt(3.0))

def load_mesh_csv(points_file='points.csv', faces_file='faces.csv'):
    """Load vertices (surf_x,y,z) and faces (v1,v2,v3; 1-indexed) from CSVs.
       Returns: vertices (float32, n x 3), faces (int32, m x 3, 0-indexed)"""
    points_df = pd.read_csv(points_file)
    need_p = ['surf_x', 'surf_y', 'surf_z']
    if points_df.shape[1] < 3 or not all(col in points_df.columns for col in need_p):
        raise ValueError("points.csv must have columns: surf_x, surf_y, surf_z")
    vertices = points_df[need_p].to_numpy(dtype=np.float32)

    faces_df = pd.read_csv(faces_file)
    need_f = ['v1', 'v2', 'v3']
    if faces_df.shape[1] < 3 or not all(col in faces_df.columns for col in need_f):
        raise ValueError("faces.csv must have columns: v1, v2, v3 (1-indexed)")
    faces = faces_df[need_f].to_numpy(dtype=np.int32) - 1  # to 0-indexed

    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Invalid face indices detected (out of vertex range).")

    return vertices, faces

def load_mesh_ply(ply_file):
    """Load vertices and faces from a PLY file using PyMeshLab.
       Returns: vertices (float32, n x 3), faces (int32, m x 3, 0-indexed)"""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_file)
    mesh = ms.current_mesh()
    
    vertices = mesh.vertex_matrix().astype(np.float32)
    faces = mesh.face_matrix().astype(np.int32)
    
    return vertices, faces

def load_mesh(input_file=None, points_file=None, faces_file=None):
    """Load mesh from either PLY file or CSV files.
       Args:
           input_file: Path to PLY file (if provided, overrides CSV options)
           points_file: Path to points CSV file
           faces_file: Path to faces CSV file
       Returns: vertices (float32, n x 3), faces (int32, m x 3, 0-indexed)"""
    
    if input_file is not None:
        # Load from PLY file
        input_path = pathlib.Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if input_path.suffix.lower() != '.ply':
            raise ValueError(f"Input file must be a .ply file, got: {input_path.suffix}")
        
        print(f"Loading mesh from PLY: {input_file}")
        return load_mesh_ply(input_file)
    
    elif points_file is not None and faces_file is not None:
        # Load from CSV files
        points_path = pathlib.Path(points_file)
        faces_path = pathlib.Path(faces_file)
        
        if not points_path.exists():
            raise FileNotFoundError(f"Points file not found: {points_file}")
        if not faces_path.exists():
            raise FileNotFoundError(f"Faces file not found: {faces_file}")
        
        print(f"Loading mesh from CSV: {points_file}, {faces_file}")
        return load_mesh_csv(points_file, faces_file)
    
    else:
        raise ValueError("Must provide either --input-ply or both --input-points and --input-faces")

def ensure_vertex_normals(ms: pymeshlab.MeshSet):
    """Recompute normals (face, then vertex) so exported PLYs carry normals."""
    ms.apply_filter('compute_normal_per_face')
    ms.apply_filter('compute_normal_per_vertex')

def save_mesh_csv(vertices, faces, points_file='remeshed_points.csv', faces_file='remeshed_faces.csv'):
    """Save vertices and faces as CSVs. Faces written 1-indexed for convenience."""
    pd.DataFrame(vertices, columns=['surf_x', 'surf_y', 'surf_z']).to_csv(str(points_file), index=False)
    pd.DataFrame(faces + 1, columns=['v1', 'v2', 'v3']).to_csv(str(faces_file), index=False)
    print(f"[CSV] Saved: {points_file}, {faces_file}")

def save_mesh_ply(vertices, faces, ply_path):
    """Save a mesh as PLY via PyMeshLab."""
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(m, 'mesh')
    ensure_vertex_normals(ms)
    ms.save_current_mesh(str(ply_path))  # Convert Path to string for pymeshlab
    print(f"[PLY] Saved: {ply_path}")

def save_mesh_npz(vertices, faces, npz_path):
    """Save numpy arrays for fast Python reload."""
    np.savez(str(npz_path), V=vertices.astype(np.float32), F=faces.astype(np.int32))  # Convert Path to string
    print(f"[NPZ] Saved: {npz_path}")

# ----------- Viz helper -----------
def visualize_mesh(vertices, faces, title='Mesh Visualization'):
    """Plotly: surface + wireframe (edges reconstructed on the fly).
       Camera matches the provided R layout (eye & up)."""
    fig = go.Figure()

    # surface
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=1, colorscale='Viridis', name='Surface'
    ))

    # wireframe from faces (unique undirected edges)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
    x_lines, y_lines, z_lines = [], [], []
    for a, b in unique_edges:
        x_lines.extend([vertices[a, 0], vertices[b, 0], None])
        y_lines.extend([vertices[a, 1], vertices[b, 1], None])
        z_lines.extend([vertices[a, 2], vertices[b, 2], None])
    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines', line=dict(color='black', width=1), name='Wire'
    ))

    # R -> Plotly(Python) camera conversion from your snippet
    fig.update_layout(
        title=title,
        showlegend=True,
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=-1.5, y=0.2, z=1.5),
                up=dict(x=0, y=1, z=0)
            )
        )
    )
    fig.show()

# ----------- Cleaning to mirror R's vcgClean(iterate=TRUE) -----------
def clean_like_rvcg(ms: pymeshlab.MeshSet,
                    max_passes: int = 10,
                    fix_non_manifold: bool = True,
                    min_component_faces: int = MIN_COMPONENT_FACES):
    """
    Approximate Rvcg::vcgClean(..., iterate=TRUE):
      - Update normals
      - Remove duplicate verts/faces, zero-area & folded (degenerate) faces
      - Remove unreferenced vertices
      - Optionally attempt non-manifold edge/vertex repair
      - Iterate until vertex/face counts no longer change or max_passes reached
      - THEN prune disconnected components smaller than `min_component_faces`
        (⚠️ removes intentionally separate small structures)
      - Finally, drop any unreferenced vertices and update normals
    """
    def counts():
        m = ms.current_mesh()
        return m.vertex_number(), m.face_number()

    # ensure normals present before cleaning
    ms.apply_filter('compute_normal_per_face')
    ms.apply_filter('compute_normal_per_vertex')

    v_prev, f_prev = counts()
    for _ in range(max_passes):
        # core face/vertex cleanups
        ms.apply_filter('meshing_remove_duplicate_vertices')
        ms.apply_filter('meshing_remove_duplicate_faces')
        ms.apply_filter('meshing_remove_null_faces')            # zero-area faces
        ms.apply_filter('meshing_remove_folded_faces')          # folded/degenerate faces
        ms.apply_filter('meshing_remove_unreferenced_vertices')

        if fix_non_manifold:
            ms.apply_filter('meshing_repair_non_manifold_edges')
            ms.apply_filter('meshing_repair_non_manifold_vertices')

        # re-normals after geometry edits
        ms.apply_filter('compute_normal_per_face')
        ms.apply_filter('compute_normal_per_vertex')

        v_cur, f_cur = counts()
        if (v_cur, f_cur) == (v_prev, f_prev):
            break
        v_prev, f_prev = v_cur, f_cur

    # --- prune disconnected components by size ---
    # ⚠️ This WILL remove intentionally separate structures smaller than the threshold.
    if min_component_faces and min_component_faces > 0:
        ms.apply_filter(
            'meshing_remove_connected_component_by_face_number',
            mincomponentsize=int(min_component_faces),
            removeunref=True
        )

    # final tidy-up
    ms.apply_filter('meshing_remove_unreferenced_vertices')
    ms.apply_filter('compute_normal_per_face')
    ms.apply_filter('compute_normal_per_vertex')

# ----------- Remeshing (PyMeshLab 2025 API) -----------
def remesh_isotropic(vertices, faces,
                     target_edge_length=TARGET_EDGE_LENGTH,
                     iterations=ITERATIONS,
                     feature_angle_deg=FEATURE_ANGLE_DEG,
                     max_surf_dist=MAX_SURF_DIST,
                     surf_dist_check=SURF_DIST_CHECK,
                     adaptive=ADAPTIVE,
                     min_component_faces=MIN_COMPONENT_FACES,
                     split=True, collapse=True, swap=True, smooth=True, reproject=True):
    """Run PyMeshLab isotropic remeshing; return new (V, F)."""
    m = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, 'input_mesh')

    # meshing_isotropic_explicit_remeshing(
    #   iterations, adaptive, selectedonly, targetlen, featuredeg,
    #   checksurfdist, maxsurfdist, splitflag, collapseflag, swapflag,
    #   smoothflag, reprojectflag )
    ms.apply_filter(
        'meshing_isotropic_explicit_remeshing',
        iterations=int(iterations),
        adaptive=bool(adaptive),
        selectedonly=False,
        targetlen=PureValue(float(target_edge_length)),
        featuredeg=float(feature_angle_deg),
        checksurfdist=bool(surf_dist_check),          # default False (disabled) for this workflow
        maxsurfdist=PureValue(float(max_surf_dist)),
        splitflag=bool(split),
        collapseflag=bool(collapse),
        swapflag=bool(swap),
        smoothflag=bool(smooth),
        reprojectflag=bool(reproject)
    )

    # Post-remeshing cleaning (iterative), then prune small disconnected components
    clean_like_rvcg(ms, max_passes=10, fix_non_manifold=True, min_component_faces=min_component_faces)

    remeshed = ms.current_mesh()
    new_vertices = remeshed.vertex_matrix().astype(np.float32)
    new_faces = remeshed.face_matrix().astype(np.int32)

    # compact indices (safety; usually already compact)
    used = np.unique(new_faces)
    if len(used) < new_vertices.shape[0]:
        remap = np.zeros(new_vertices.shape[0], dtype=np.int32) - 1
        remap[used] = np.arange(len(used), dtype=np.int32)
        new_vertices = new_vertices[used]
        new_faces = remap[new_faces]

    return new_vertices, new_faces

# ----------- Command line argument parsing -----------
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Isotropic remeshing of 3D triangular meshes using PyMeshLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from PLY file with default target area (4.0)
  python isotropic_remeshing.py --input-ply mesh.ply
  
  # Load from CSV files
  python isotropic_remeshing.py --input-points points.csv --input-faces faces.csv
  
  # Custom target area for edge length calculation
  python isotropic_remeshing.py --input-ply mesh.ply --target-area 2.5
  
  # Direct edge length specification
  python isotropic_remeshing.py --input-ply mesh.ply --target-edge-length 1.5
  
  # Custom output prefix and parameters
  python isotropic_remeshing.py --input-ply mesh.ply --output-prefix my_mesh --target-area 3.0
  
  # No visualization
  python isotropic_remeshing.py --input-ply mesh.ply --no-viz
        """
    )
    
    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-ply', type=str,
                            help='Input PLY file path')
    
    csv_group = parser.add_argument_group('CSV input (use both together)')
    csv_group.add_argument('--input-points', type=str,
                          help='Input points CSV file (surf_x, surf_y, surf_z)')
    csv_group.add_argument('--input-faces', type=str,
                          help='Input faces CSV file (v1, v2, v3; 1-indexed)')
    
    # Output options
    parser.add_argument('--output-prefix', type=str, default='mesh',
                       help='Output file prefix (default: mesh)')
    
    # Remeshing parameters
    remesh_group = parser.add_mutually_exclusive_group()
    remesh_group.add_argument('--target-edge-length', type=float,
                             help=f'Target edge length for remeshing (default: calculated from area)')
    remesh_group.add_argument('--target-area', type=float, default=TARGET_AREA,
                             help=f'Target area of equilateral triangle for edge length calculation (default: {TARGET_AREA})')
    
    parser.add_argument('--iterations', type=int, default=ITERATIONS,
                       help=f'Number of remeshing iterations (default: {ITERATIONS})')
    parser.add_argument('--feature-angle', type=float, default=FEATURE_ANGLE_DEG,
                       help=f'Feature angle in degrees (default: {FEATURE_ANGLE_DEG})')
    parser.add_argument('--adaptive', action='store_true',
                       help='Use adaptive remeshing')
    parser.add_argument('--surf-dist-check', action='store_true',
                       help='Enable surface distance checking (slower but more accurate)')
    
    # Cleaning parameters
    parser.add_argument('--min-component-faces', type=int, default=MIN_COMPONENT_FACES,
                       help=f'Minimum faces in connected components to keep (default: {MIN_COMPONENT_FACES})')
    
    # Visualization
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip mesh visualization')
    
    args = parser.parse_args()
    
    # Validate CSV input combination
    csv_provided = (args.input_points is not None) or (args.input_faces is not None)
    if csv_provided and not (args.input_points and args.input_faces):
        parser.error("Both --input-points and --input-faces must be provided together")
    
    # Set PLY vs CSV mode
    if args.input_ply:
        args.input_mode = 'ply'
    else:
        args.input_mode = 'csv'
    
    return args

# ----------- Main -----------
if __name__ == "__main__":
    args = parse_arguments()
    
    # Load mesh based on input mode
    if args.input_mode == 'ply':
        V, F = load_mesh(input_file=args.input_ply)
    else:
        V, F = load_mesh(points_file=args.input_points, faces_file=args.input_faces)

    # Calculate target edge length
    if args.target_edge_length is not None:
        target_edge_length = args.target_edge_length
        print(f"Using specified target edge length: {target_edge_length}")
    else:
        target_edge_length = calculate_edge_length_from_area(args.target_area)
        print(f"Calculated target edge length: {target_edge_length:.4f} from target area: {args.target_area}")

    # Create results directory and generate output filenames
    results_dir = ensure_results_directory()
    prefix = args.output_prefix
    original_ply = results_dir / f"{prefix}_original.ply"
    original_npz = results_dir / f"{prefix}_original.npz"
    remeshed_points_csv = results_dir / f"{prefix}_remeshed_points.csv"
    remeshed_faces_csv = results_dir / f"{prefix}_remeshed_faces.csv"
    remeshed_ply = results_dir / f"{prefix}_remeshed.ply"
    remeshed_npz = results_dir / f"{prefix}_remeshed.npz"

    # --- Save ORIGINAL as PLY + NPZ ---
    print("Saving ORIGINAL mesh (PLY + NPZ)...")
    save_mesh_ply(V, F, original_ply)
    save_mesh_npz(V, F, original_npz)

    if not args.no_viz:
        print("Visualizing original mesh...")
        visualize_mesh(V, F, title='Original Mesh')

    print("Performing isotropic remeshing...")
    V2, F2 = remesh_isotropic(
        V, F,
        target_edge_length=target_edge_length,
        iterations=args.iterations,
        feature_angle_deg=args.feature_angle,
        adaptive=args.adaptive,
        surf_dist_check=args.surf_dist_check,
        min_component_faces=args.min_component_faces
    )

    if not args.no_viz:
        print("Visualizing remeshed mesh...")
        visualize_mesh(V2, F2, title='Remeshed Mesh')

    # --- Save REMESHED as CSV, PLY, NPZ ---
    print("Saving REMESHED mesh (CSV + PLY + NPZ)...")
    save_mesh_csv(V2, F2, points_file=remeshed_points_csv, faces_file=remeshed_faces_csv)
    save_mesh_ply(V2, F2, remeshed_ply)
    save_mesh_npz(V2, F2, remeshed_npz)

    print("Done. Outputs in 'results' folder:")
    print(f"  ORIGINAL:  {original_ply.name}, {original_npz.name}")
    print(f"  REMESHED:  {remeshed_points_csv.name}, {remeshed_faces_csv.name}, {remeshed_ply.name}, {remeshed_npz.name}")
    
    # Print mesh statistics
    print(f"\nMesh Statistics:")
    print(f"  Original:  {V.shape[0]} vertices, {F.shape[0]} faces")
    print(f"  Remeshed:  {V2.shape[0]} vertices, {F2.shape[0]} faces")
    print(f"  Target edge length used: {target_edge_length:.4f}")
    print(f"\nAll files saved to: {results_dir.absolute()}")
