import torch, numpy as np, os, random, re, sys
sys.path.insert(0, '/data5/jl/project/tokenizer_seg/cosmo3d_other_dirs_excluding_main')
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import trimesh, open3d as o3d
from pipes_tools.generate_superpoint_train import superpoint
from pipes_tools.find_unified_rotation import load_mesh_concat, normalize
R_UNIFIED = np.array([[0,-1,0],[0,0,1],[-1,0,0]], np.float32)
MESH_ROOT = '/data3/jl/dataset/3DCoMPaT200/datasets--CoMPaT--3DCoMPaT200/snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/Compat200/models'
FOLDER_RE = re.compile(r"^(?:coarse|fine)_b'(?P<id>[^']+)'$")
def parse_id(name):
    m = FOLDER_RE.match(name); return m.group('id') if m else None
def mesh_normals_at_points(mesh, pts):
    try: _, _, face_idx = trimesh.proximity.closest_point(mesh, pts)
    except Exception:
        tri = np.asarray(mesh.triangles_center); nn = NearestNeighbors(n_neighbors=1).fit(tri)
        face_idx = nn.kneighbors(pts, return_distance=False).reshape(-1)
    fn = np.asarray(mesh.face_normals, dtype=np.float32); n = fn[np.clip(face_idx.astype(int), 0, len(fn)-1)]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
def best_yaw(xyz, mesh, n=2048):
    p = xyz[np.random.choice(xyz.shape[0], min(n, xyz.shape[0]), replace=False)]
    mpts, _ = trimesh.sample.sample_surface(mesh, min(n, max(512, p.shape[0])), seed=0)
    m = normalize(torch.from_numpy(mpts.astype(np.float32))).numpy(); p = normalize(torch.from_numpy(p)).numpy()
    best_cd, best = 1e9, 0
    for deg in [0, 90, -90, 180]:
        th = np.deg2rad(deg); c,s = np.cos(th), np.sin(th)
        Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32); pr = p @ Ry.T
        d = ((pr[:,None]-m[None,:])**2).sum(-1).min(1).mean() + ((m[:,None]-pr[None,:])**2).sum(-1).min(1).mean()
        if d < best_cd: best_cd, best = d, deg
    return best
with open('/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat/train.txt') as f:
    dirs = [l.strip() for l in f if l.strip() and os.path.exists(l.strip()+'/parts.pt')]
random.seed(0); samples = random.sample(dirs, 12)
print('obj | gt | ARI_pt | ARI_mesh | ARI_o3d | yaw')
aris = {'pt':[], 'mesh':[], 'o3d':[]}
for d in samples:
    mid = parse_id(os.path.basename(d)); gltf = os.path.join(MESH_ROOT, f'{mid}.gltf')
    if not mid or not os.path.exists(gltf): continue
    xyz = torch.load(d+'/points.pt', weights_only=True).numpy().astype(np.float32)
    nrm_pt = torch.load(d+'/normals.pt', weights_only=True).numpy().astype(np.float32)
    gt = torch.load(d+'/parts.pt', weights_only=True).numpy().reshape(-1)
    mesh = load_mesh_concat(gltf); yaw = best_yaw(xyz, mesh)
    th = np.deg2rad(yaw); c,s = np.cos(th), np.sin(th)
    Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32); R = Ry @ R_UNIFIED
    xyz_a = xyz @ R.T; nrm_mesh = mesh_normals_at_points(mesh, xyz_a)
    tok_pt = superpoint(xyz, nrm_pt); tok_mesh = superpoint(xyz_a, nrm_mesh)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    tok_o3d = superpoint(xyz, np.asarray(pcd.normals, np.float32))
    a_pt,a_mesh,a_o3d = adjusted_rand_score(gt,tok_pt), adjusted_rand_score(gt,tok_mesh), adjusted_rand_score(gt,tok_o3d)
    aris['pt'].append(a_pt); aris['mesh'].append(a_mesh); aris['o3d'].append(a_o3d)
    print(f"{os.path.basename(d)} | {len(np.unique(gt))} | {a_pt:.3f} | {a_mesh:.3f} | {a_o3d:.3f} | {yaw}")
print('avg', np.mean(aris['pt']), np.mean(aris['mesh']), np.mean(aris['o3d']))
