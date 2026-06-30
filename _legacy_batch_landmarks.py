import os
import pyvista as pv
import numpy as np

# =========================
# CONFIG
# =========================
input_folder = "data"
output_folder = "results"
output_file = "ALL_Landmarks.csv"
n_semilandmarks = 50

os.makedirs(output_folder, exist_ok=True)

# =========================
# FUNCIONS
# =========================
def detect_cusps(points, min_distance=4.5, max_cusps=5, alpha=0.35):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    cx = np.mean(x)
    cy = np.mean(y)

    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_norm = (r - r.min()) / (r.max() - r.min())

    score = z - alpha * r_norm * (z.max() - z.min())

    r_limit = np.percentile(r, 95)
    valid_mask = r < r_limit

    valid_points = points[valid_mask]
    valid_score = score[valid_mask]

    order = np.argsort(valid_score)[::-1]
    sorted_points = valid_points[order]

    selected_cusps = []

    for p in sorted_points:
        if len(selected_cusps) == 0:
            selected_cusps.append(p)
        else:
            dists = [np.linalg.norm(p[:2] - q[:2]) for q in selected_cusps]
            if min(dists) > min_distance:
                selected_cusps.append(p)

        if len(selected_cusps) >= max_cusps:
            break

    selected_cusps = np.array(selected_cusps)

    if len(selected_cusps) < 5:
        raise ValueError(f"Només s'han detectat {len(selected_cusps)} cúspides.")

    return selected_cusps


def reorder_cusps_manual(selected_cusps):
    return np.array([
        selected_cusps[4],  # 1
        selected_cusps[2],  # 2
        selected_cusps[3],  # 3
        selected_cusps[0],  # 4
        selected_cusps[1],  # 5
    ])


def line_minimum_on_mesh(points, c1, c2, n_samples=120, t_min=0.20, t_max=0.80):
    sampled_indices = []

    for t in np.linspace(t_min, t_max, n_samples):
        xy = (1.0 - t) * c1[:2] + t * c2[:2]
        dists = np.linalg.norm(points[:, :2] - xy, axis=1)
        idx = np.argmin(dists)
        sampled_indices.append(idx)

    sampled_indices = list(dict.fromkeys(sampled_indices))
    sampled_points = points[sampled_indices]

    min_idx = np.argmin(sampled_points[:, 2])
    return sampled_points[min_idx]


def detect_intercuspal_points(points, ordered_cusps):
    intercuspal_points = []

    for i in range(len(ordered_cusps)):
        c1 = ordered_cusps[i]
        c2 = ordered_cusps[(i + 1) % len(ordered_cusps)]
        p = line_minimum_on_mesh(points, c1, c2, n_samples=120)
        intercuspal_points.append(p)

    return np.array(intercuspal_points)


def detect_central_point(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    cx = np.mean(x)
    cy = np.mean(y)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    r_limit_center = np.percentile(r, 18)
    center_mask = r < r_limit_center
    center_points = points[center_mask]

    if len(center_points) == 0:
        raise ValueError("No s'han trobat punts centrals.")

    return center_points[np.argmin(center_points[:, 2])]


def polygon_signed_area_xy(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def order_points_closed_loop(points):
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def ensure_clockwise(points):
    area = polygon_signed_area_xy(points)
    if area > 0:
        return points[::-1]
    return points


def rotate_curve_start(points, reference_point):
    dists = np.linalg.norm(points[:, :2] - reference_point[:2], axis=1)
    start_idx = np.argmin(dists)
    return np.vstack([points[start_idx:], points[:start_idx]])


def resample_closed_curve(points, n_samples=50):
    if len(points) < 3:
        raise ValueError("No hi ha prou punts per remostrejar la corba.")

    closed = np.vstack([points, points[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length == 0:
        raise ValueError("La longitud total de la corba és zero.")

    target_distances = np.linspace(0, total_length, n_samples, endpoint=False)

    sampled = []
    seg_idx = 0

    for td in target_distances:
        while seg_idx < len(seg_lengths) - 1 and cumulative[seg_idx + 1] < td:
            seg_idx += 1

        p0 = closed[seg_idx]
        p1 = closed[seg_idx + 1]
        seg_start = cumulative[seg_idx]
        seg_len = seg_lengths[seg_idx]

        if seg_len == 0:
            sampled.append(p0)
            continue

        t = (td - seg_start) / seg_len
        p = (1 - t) * p0 + t * p1
        sampled.append(p)

    return np.array(sampled)


def extract_lowest_boundary_loop(mesh):
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )

    if boundary.n_points == 0:
        raise ValueError("No s'ha detectat cap boundary edge. La malla pot estar tancada.")

    connected = boundary.connectivity()
    region_ids = connected["RegionId"]
    unique_regions = np.unique(region_ids)

    loops = []
    z_means = []

    for rid in unique_regions:
        pts = connected.points[region_ids == rid]
        if len(pts) >= 3:
            loops.append(pts)
            z_means.append(np.mean(pts[:, 2]))

    if len(loops) == 0:
        raise ValueError("No s'ha pogut extreure cap loop de la boundary.")

    lowest_idx = np.argmin(z_means)
    return loops[lowest_idx]


def detect_semilandmarks_on_lower_border(mesh, reference_point, n_samples=50):
    border_pts = extract_lowest_boundary_loop(mesh)
    ordered_border = order_points_closed_loop(border_pts)
    ordered_border = ensure_clockwise(ordered_border)
    ordered_border = rotate_curve_start(ordered_border, reference_point)
    semilandmarks = resample_closed_curve(ordered_border, n_samples=n_samples)
    return semilandmarks


# =========================
# LOOP PRINCIPAL
# =========================
print("Looking in:", input_folder)
files = sorted(os.listdir(input_folder))
print("Files found:", files)

all_landmarks = []
failed_files = []

for filename in files:
    if not filename.lower().endswith(".ply"):
        continue

    print("Processing:", filename)

    filepath = os.path.join(input_folder, filename)

    try:
        mesh = pv.read(filepath)
        points = mesh.points

        # 1. cúspides
        selected_cusps = detect_cusps(points)
        ordered_cusps = reorder_cusps_manual(selected_cusps)

        # 2. intercuspídis
        intercuspal_points = detect_intercuspal_points(points, ordered_cusps)

        # 3. punt central
        central_point = detect_central_point(points)

        # 4. semilandmarks vora
        reference_cusp1 = ordered_cusps[0]
        semilandmarks = detect_semilandmarks_on_lower_border(
            mesh,
            reference_point=reference_cusp1,
            n_samples=n_semilandmarks
        )

        # Guardar cúspides
        for i, p in enumerate(ordered_cusps, start=1):
            all_landmarks.append([filename, f"cusp{i}", p[0], p[1], p[2]])

        # Guardar intercuspídis
        for i, p in enumerate(intercuspal_points, start=1):
            all_landmarks.append([filename, f"inter{i}", p[0], p[1], p[2]])

        # Guardar centre
        all_landmarks.append([filename, "center", central_point[0], central_point[1], central_point[2]])

        # Guardar semilandmarks
        for i, p in enumerate(semilandmarks, start=1):
            all_landmarks.append([filename, f"semi{i}", p[0], p[1], p[2]])

    except Exception as e:
        print(f"  ERROR amb {filename}: {e}")
        failed_files.append([filename, str(e)])


# =========================
# EXPORT FINAL
# =========================
global_file = os.path.join(output_folder, output_file)

with open(global_file, "w", encoding="utf-8") as f:
    f.write("file,type,x,y,z\n")
    for row in all_landmarks:
        f.write(",".join(map(str, row)) + "\n")

print("\nDONE")
print("Saved:", global_file)

if failed_files:
    failed_log = os.path.join(output_folder, "failed_files.txt")
    with open(failed_log, "w", encoding="utf-8") as f:
        for row in failed_files:
            f.write(f"{row[0]}\t{row[1]}\n")
    print("Failed files log saved:", failed_log)