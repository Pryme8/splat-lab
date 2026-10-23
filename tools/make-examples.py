# Generates the files in examples/. Needs: pip install trimesh pillow numpy
# Run from the examples/ directory: python ../tools/make-examples.py
import numpy as np, trimesh, struct
from PIL import Image
# 1) textured plane + vertex-colored sphere in one GLB
img = Image.new('RGB', (64, 64)); px = img.load()
for y in range(64):
    for x in range(64):
        px[x, y] = (40, 200, 60) if y < 32 else (220, 40, 200)   # top half green, bottom half magenta
verts = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float)
uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)  # trimesh uses OpenGL-style v (flipped on export)
plane = trimesh.Trimesh(verts, [[0, 2, 1], [0, 3, 2]], visual=trimesh.visual.TextureVisuals(uv=uv, image=img), process=False)
sph = trimesh.creation.icosphere(subdivisions=3, radius=0.6); sph.apply_translation([0, 0, 1.2])
c = ((sph.vertices[:, 0:1] + 0.6) / 1.2 * np.array([[255, 60, 20]]) + np.array([[0, 80, 200]])).clip(0, 255).astype(np.uint8)
sph.visual.vertex_colors = np.hstack([c, np.full((len(c), 1), 255, np.uint8)])
trimesh.Scene([plane, sph]).export('textured.glb')
# 2) OBJ and STL
trimesh.creation.torus(1.0, 0.35).export('torus.obj')
trimesh.creation.capsule(1.5, 0.5).export('capsule.stl')
# 3) 3DGS-style PLY (helix of Gaussians)
n = 20000; rs = np.random.default_rng(1)
t = rs.uniform(0, 6 * np.pi, n); ang = rs.uniform(0, 2 * np.pi, n)
R0 = 0.35
cen = np.stack([np.cos(t), t / 6 - 1.5, np.sin(t)], 1)
nrm = np.stack([np.cos(t) * np.cos(ang), np.sin(ang), np.sin(t) * np.cos(ang)], 1)
pos = cen + R0 * nrm
cols = np.stack([0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(ang), 0.5 - 0.5 * np.cos(t)], 1)
fdc = (cols - 0.5) / 0.28209479
op = np.full(n, 3.0)  # logit -> ~0.95
sc = np.log(np.stack([np.full(n, 0.04), np.full(n, 0.03), np.full(n, 0.005)], 1))
# rotation: identity-ish random quats
q = rs.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
fields = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
data = np.hstack([pos, np.zeros((n, 3)), fdc, op[:, None], sc, q]).astype('<f4')
hdr = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join('property float %s\n' % f for f in fields) + 'end_header\n'
open('helix_gaussians.ply', 'wb').write(hdr.encode() + data.tobytes())
# 4) .splat (antimatter15 layout) of same data
buf = bytearray()
lin = np.exp(sc); a = 1 / (1 + np.exp(-op))
for i in range(n):
    buf += struct.pack('<6f', *pos[i], *lin[i])
    buf += bytes([int(np.clip(cols[i, 0] * 255, 0, 255)), int(np.clip(cols[i, 1] * 255, 0, 255)), int(np.clip(cols[i, 2] * 255, 0, 255)), int(a[i] * 255)])
    buf += bytes([int(np.clip(v * 128 + 128, 0, 255)) for v in q[i]])
open('helix.splat', 'wb').write(bytes(buf))
# 5) point cloud PLY: torus knot surface points with colors
m = 60000; u = rs.uniform(0, 2 * np.pi, m); v = rs.uniform(0, 2 * np.pi, m)
r = np.cos(3 * u) + 2
C = np.stack([r * np.cos(2 * u), r * np.sin(2 * u), -np.sin(3 * u)], 1) * 0.6
C2 = np.stack([(np.cos(3 * (u + 1e-3)) + 2) * np.cos(2 * (u + 1e-3)), (np.cos(3 * (u + 1e-3)) + 2) * np.sin(2 * (u + 1e-3)), -np.sin(3 * (u + 1e-3))], 1) * 0.6
T = C2 - C; T /= np.linalg.norm(T, axis=1, keepdims=True)
B = np.cross(T, [0, 0, 1]); B /= np.linalg.norm(B, axis=1, keepdims=True); N = np.cross(B, T)
P = C + 0.22 * (np.cos(v)[:, None] * N + np.sin(v)[:, None] * B)
rgb = (np.stack([0.5 + 0.5 * np.cos(u), 0.4 + 0.4 * np.cos(v), 0.6 + 0.4 * np.sin(u)], 1) * 255).astype(np.uint8)
rec = np.zeros(m, dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
rec['x'], rec['y'], rec['z'] = P[:, 0], P[:, 1], P[:, 2]; rec['red'], rec['green'], rec['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
hdr = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % m
open('knot_points.ply', 'wb').write(hdr.encode() + rec.tobytes())
print('ok')
