/* =========================================================================
   SplatTrain: a WebGPU differentiable tile rasterizer and trainer for three
   primitive types, written from the papers' image-formation models:
     type 0  3D Gaussian splatting (Kerbl et al. 2023)
     type 1  triangle splatting, window ReLU(phi/phi(incenter))^sigma (Held et al. 2025)
     type 2  3D convex splatting, sigmoid(-sigma * LSE_delta(edge distances)) (Held et al. 2025)
   Pipeline per view: project -> tile counts -> prefix sum -> duplicate keys
   (tile | depth) -> 4-bit LSD radix sort -> per-tile ranges -> front-to-back
   compositing. Training adds: L1/L2 loss -> back-to-front gradient pass with
   float atomics -> chain rule to 3D parameters -> Adam. The key count never
   leaves the GPU: sort and ranges are sized by indirect dispatch.
   Standalone: it creates its own GPUDevice and exchanges plain Float32Arrays.
   ========================================================================= */
const SplatTrain = (() => {
  "use strict";
  const PS_OF = [16, 16, 24];      // parameter stride per type
  const JS_OF = [16, 24, 28];      // projected-record stride per type
  const CAM_BYTES = 144;
  const MAX_BLOCKS = 65535;

  /* ---------------------------------------------------------------- WGSL */
  const camStruct = `
struct Cam {
  view: mat4x4<f32>,
  intr: vec4<f32>,   // fx fy cx cy
  dims: vec4<u32>,   // W H tilesX tilesY
  clip: vec4<f32>,   // near far sigmaTri deltaCvx
  misc: vec4<f32>,   // sigmaCvx bgR bgG bgB
  cnt: vec4<u32>,    // N - depthBits cap
};`;
  function header(type) {
    return `${camStruct}
const PS: u32 = ${PS_OF[type]}u;
const JS: u32 = ${JS_OF[type]}u;
const GS: u32 = 20u;
const COLI: u32 = ${[5, 11, 15][type]}u;
const OPI: u32 = ${[8, 14, 18][type]}u;
const ZI: u32 = ${[9, 15, 19][type]}u;
const RECTI: u32 = ${[10, 16, 21][type]}u;
fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }
`;
  }
  const viewFns = `
fn W3() -> mat3x3<f32> { return mat3x3<f32>(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz); }
fn toView(p: vec3f) -> vec3f { return (cam.view * vec4f(p, 1.0)).xyz; }
fn toPix(t: vec3f) -> vec2f { return vec2f(cam.intr.x * t.x / t.z + cam.intr.z, cam.intr.w - cam.intr.y * t.y / t.z); }
fn jrow0(t: vec3f) -> vec3f { return vec3f(cam.intr.x / t.z, 0.0, -cam.intr.x * t.x / (t.z * t.z)); }
fn jrow1(t: vec3f) -> vec3f { return vec3f(0.0, -cam.intr.y / t.z, cam.intr.y * t.y / (t.z * t.z)); }
fn tileRect(lo: vec2f, hi: vec2f) -> vec4f {
  let tx = f32(cam.dims.z); let ty = f32(cam.dims.w);
  let x0 = clamp(floor(lo.x / 16.0), 0.0, tx); let y0 = clamp(floor(lo.y / 16.0), 0.0, ty);
  let x1 = clamp(floor(hi.x / 16.0) + 1.0, x0, tx); let y1 = clamp(floor(hi.y / 16.0) + 1.0, y0, ty);
  return vec4f(x0, y0, x1, y1);
}
fn quatMat(q0: vec4f) -> mat3x3<f32> {
  let q = q0 / max(length(q0), 1e-12);
  let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  return mat3x3<f32>(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}`;

  /* window functions on a projected record q (copied into a var for dynamic indexing) */
  const windowFns = [
`fn evalW(q: array<f32, JS>, x: vec2f) -> f32 {
  let d = x - vec2f(q[0], q[1]);
  let power = -0.5 * (q[2] * d.x * d.x + q[4] * d.y * d.y) - q[3] * d.x * d.y;
  if (power > 0.0) { return 0.0; }
  return exp(power);
}`,
`fn edgeD(P: vec2f, Q: vec2f, invL: f32, s: f32, x: vec2f) -> f32 { return s * cross2(Q - P, x - P) * invL; }
fn evalW(q: array<f32, JS>, x: vec2f) -> f32 {
  let a = vec2f(q[0], q[1]); let b = vec2f(q[2], q[3]); let c = vec2f(q[4], q[5]); let s = q[10];
  let m = min(min(edgeD(a, b, q[6], s, x), edgeD(b, c, q[7], s, x)), edgeD(c, a, q[8], s, x));
  if (m <= 0.0) { return 0.0; }
  return pow(m / q[9], cam.clip.z);
}`,
`fn evalW(q0: array<f32, JS>, x: vec2f) -> f32 {
  var q = q0;
  let n = u32(q[12]); let Sz = q[13]; let s = q[14]; let dl = cam.clip.w; let sg = cam.misc.x;
  var L: array<f32, 6>; var m = -1e9;
  for (var j = 0u; j < n; j++) {
    let k = select(j + 1u, 0u, j + 1u == n);
    let P = vec2f(q[2u * j], q[2u * j + 1u]); let Q = vec2f(q[2u * k], q[2u * k + 1u]);
    let e = Q - P; let len = length(e);
    var l = -1e9;
    if (len > 1e-6) { l = -s * cross2(e, x - P) / (len * Sz); }
    L[j] = l; m = max(m, l);
  }
  if (sg * m > 9.0) { return 0.0; }
  var acc = 0.0;
  for (var j = 0u; j < n; j++) { acc += exp(dl * (L[j] - m)); }
  let phi = m + log(acc) / dl;
  return 1.0 / (1.0 + exp(sg * phi));
}`];

  /* ---- projection, one thread per primitive ---- */
  const projectBodies = [
`  let mean = vec3f(P[b], P[b + 1u], P[b + 2u]);
  let t = toView(mean);
  if (t.z < cam.clip.x) { return; }
  let s = exp(vec3f(P[b + 3u], P[b + 4u], P[b + 5u]));
  let R = quatMat(vec4f(P[b + 6u], P[b + 7u], P[b + 8u], P[b + 9u]));
  let M = mat3x3<f32>(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let Wm = W3();
  let Sv = Wm * (M * transpose(M)) * transpose(Wm);
  let j0 = jrow0(t); let j1 = jrow1(t);
  let a = dot(j0, Sv * j0) + 0.3; let bb = dot(j0, Sv * j1); let c = dot(j1, Sv * j1) + 0.3;
  let det = a * c - bb * bb;
  if (det <= 1e-12) { return; }
  let mid = 0.5 * (a + c); let lam = mid + sqrt(max(0.1, mid * mid - det)); let rad = ceil(3.0 * sqrt(lam));
  let px = toPix(t);
  let rect = tileRect(px - vec2f(rad), px + vec2f(rad));
  let n = u32((rect.z - rect.x) * (rect.w - rect.y));
  if (n == 0u) { return; }
  J[o] = px.x; J[o + 1u] = px.y; J[o + 2u] = c / det; J[o + 3u] = -bb / det; J[o + 4u] = a / det;
  J[o + 5u] = max(P[b + 10u], 0.0); J[o + 6u] = max(P[b + 11u], 0.0); J[o + 7u] = max(P[b + 12u], 0.0);
  J[o + 8u] = sigm(P[b + 13u]); J[o + 9u] = t.z;
  J[o + 10u] = rect.x; J[o + 11u] = rect.y; J[o + 12u] = rect.z; J[o + 13u] = rect.w;
  tiles[i] = n;`,
`  let t0 = toView(vec3f(P[b], P[b + 1u], P[b + 2u]));
  let t1 = toView(vec3f(P[b + 3u], P[b + 4u], P[b + 5u]));
  let t2 = toView(vec3f(P[b + 6u], P[b + 7u], P[b + 8u]));
  if (min(min(t0.z, t1.z), t2.z) < cam.clip.x) { return; }
  let a = toPix(t0); let bq = toPix(t1); let c = toPix(t2);
  let cr = cross2(bq - a, c - a);
  if (abs(cr) < 1e-4) { return; }
  let s = sign(cr);
  let L0 = length(bq - a); let L1 = length(c - bq); let L2 = length(a - c);
  let r = abs(cr) / (L0 + L1 + L2);
  let rect = tileRect(min(min(a, bq), c), max(max(a, bq), c));
  let n = u32((rect.z - rect.x) * (rect.w - rect.y));
  if (n == 0u) { return; }
  J[o] = a.x; J[o + 1u] = a.y; J[o + 2u] = bq.x; J[o + 3u] = bq.y; J[o + 4u] = c.x; J[o + 5u] = c.y;
  J[o + 6u] = 1.0 / L0; J[o + 7u] = 1.0 / L1; J[o + 8u] = 1.0 / L2; J[o + 9u] = r; J[o + 10u] = s;
  J[o + 11u] = max(P[b + 9u], 0.0); J[o + 12u] = max(P[b + 10u], 0.0); J[o + 13u] = max(P[b + 11u], 0.0);
  J[o + 14u] = sigm(P[b + 12u]); J[o + 15u] = (t0.z + t1.z + t2.z) / 3.0;
  J[o + 16u] = rect.x; J[o + 17u] = rect.y; J[o + 18u] = rect.z; J[o + 19u] = rect.w;
  tiles[i] = n;`,
`  var S: array<vec2f, 6>; var zs = 0.0;
  for (var k = 0u; k < 6u; k++) {
    let t = toView(vec3f(P[b + 3u * k], P[b + 3u * k + 1u], P[b + 3u * k + 2u]));
    if (t.z < cam.clip.x) { return; }
    S[k] = toPix(t); zs += t.z;
  }
  var start = 0u;
  for (var k = 1u; k < 6u; k++) { if (S[k].x < S[start].x || (S[k].x == S[start].x && S[k].y < S[start].y)) { start = k; } }
  var H: array<vec2f, 6>; var hid: array<u32, 6>; var n = 0u; var cur = start;
  for (var k = 0u; k < 6u; k++) {
    H[k] = S[cur]; hid[k] = cur; n = k + 1u;
    var nxt = select(cur + 1u, 0u, cur == 5u);
    for (var j = 0u; j < 6u; j++) {
      if (j == cur) { continue; }
      let ea = S[nxt] - S[cur]; let eb = S[j] - S[cur];
      let c = cross2(ea, eb);
      if (c < 0.0 || (c == 0.0 && dot(eb, eb) > dot(ea, ea))) { nxt = j; }
    }
    cur = nxt;
    if (cur == start) { break; }
  }
  var area2 = 0.0; var lo = H[0]; var hi = H[0];
  for (var k = 0u; k < n; k++) {
    let q = H[select(k + 1u, 0u, k + 1u == n)];
    area2 += cross2(H[k], q); lo = min(lo, H[k]); hi = max(hi, H[k]);
  }
  if (n < 3u || abs(area2) < 1e-4) { return; }
  let Sz = sqrt(abs(area2) * 0.5);
  let mg = min((5.5 / cam.misc.x + log(6.0) / cam.clip.w) * Sz, 4.0 * Sz) + 1.0;
  let rect = tileRect(lo - vec2f(mg), hi + vec2f(mg));
  let cnt = u32((rect.z - rect.x) * (rect.w - rect.y));
  if (cnt == 0u) { return; }
  var packed = 0u;
  for (var k = 0u; k < 6u; k++) {
    let kk = min(k, n - 1u);
    J[o + 2u * k] = H[kk].x; J[o + 2u * k + 1u] = H[kk].y;
    packed = packed | (hid[kk] << (3u * k));
  }
  J[o + 12u] = f32(n); J[o + 13u] = Sz; J[o + 14u] = sign(area2);
  J[o + 15u] = max(P[b + 18u], 0.0); J[o + 16u] = max(P[b + 19u], 0.0); J[o + 17u] = max(P[b + 20u], 0.0);
  J[o + 18u] = sigm(P[b + 21u]); J[o + 19u] = zs / 6.0; J[o + 20u] = f32(packed);
  J[o + 21u] = rect.x; J[o + 22u] = rect.y; J[o + 23u] = rect.z; J[o + 24u] = rect.w;
  tiles[i] = cnt;`];
  const projectWGSL = type => `${header(type)}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read_write> J: array<f32>;
@group(0) @binding(3) var<storage, read_write> tiles: array<u32>;
${viewFns}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= cam.cnt.x) { return; }
  let b = i * PS; let o = i * JS;
  tiles[i] = 0u;
${projectBodies[type]}
}`;

  /* ---- scan (Blelloch, 512 per block) ---- */
  const scanWGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;
@group(0) @binding(2) var<uniform> U: vec4<u32>;
var<workgroup> s: array<u32, 512>;
@compute @workgroup_size(256) fn scanBlock(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let n = U.x; let blk = w.x + w.y * 65535u; let base = blk * 512u; let t = l.x;
  let i0 = base + 2u * t; let i1 = i0 + 1u;
  s[2u * t] = select(0u, data[min(i0, n - 1u)], i0 < n);
  s[2u * t + 1u] = select(0u, data[min(i1, n - 1u)], i1 < n);
  var off = 1u;
  for (var d = 256u; d > 0u; d = d >> 1u) {
    workgroupBarrier();
    if (t < d) { let ai = off * (2u * t + 1u) - 1u; let bi = off * (2u * t + 2u) - 1u; s[bi] = s[bi] + s[ai]; }
    off = off << 1u;
  }
  workgroupBarrier();
  if (t == 0u) { sums[blk] = s[511]; s[511] = 0u; }
  for (var d = 1u; d < 512u; d = d << 1u) {
    off = off >> 1u;
    workgroupBarrier();
    if (t < d) { let ai = off * (2u * t + 1u) - 1u; let bi = off * (2u * t + 2u) - 1u; let tmp = s[ai]; s[ai] = s[bi]; s[bi] = s[bi] + tmp; }
  }
  workgroupBarrier();
  if (i0 < n) { data[i0] = s[2u * t]; }
  if (i1 < n) { data[i1] = s[2u * t + 1u]; }
}
@compute @workgroup_size(256) fn addSums(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let n = U.x; let blk = w.x + w.y * 65535u; let add = sums[blk];
  let i0 = blk * 512u + 2u * l.x; let i1 = i0 + 1u;
  if (i0 < n) { data[i0] = data[i0] + add; }
  if (i1 < n) { data[i1] = data[i1] + add; }
}`;

  /* key count -> sort size, kept on the GPU so a step never waits for a readback.
     info: K (capped), K (raw), blocks of 256 keys, projections that overflowed the cap (sticky).
     ind: dispatchWorkgroupsIndirect args covering those blocks (same 2D split as dispatchDims). */
  const argsWGSL = `${camStruct}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> offs: array<u32>;
@group(0) @binding(2) var<storage, read> tiles: array<u32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;
@group(0) @binding(4) var<storage, read_write> ind: array<u32>;
@compute @workgroup_size(1) fn main() {
  let n = cam.cnt.x;
  var total = 0u;
  if (n > 0u) { total = offs[n - 1u] + tiles[n - 1u]; }
  let K = min(total, cam.cnt.w); let nb = (K + 255u) / 256u;
  info[0] = K; info[1] = total; info[2] = nb;
  if (total > K) { info[3] = info[3] + 1u; }
  ind[0] = min(nb, 65535u); ind[1] = (nb + 65534u) / 65535u; ind[2] = 1u;
}`;

  const duplicateWGSL = type => `${header(type)}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> J: array<f32>;
@group(0) @binding(2) var<storage, read> tiles: array<u32>;
@group(0) @binding(3) var<storage, read> offs: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= cam.cnt.x || tiles[i] == 0u) { return; }
  let o = i * JS; let db = cam.cnt.z;
  let x0 = u32(J[o + RECTI]); let y0 = u32(J[o + RECTI + 1u]); let x1 = u32(J[o + RECTI + 2u]); let y1 = u32(J[o + RECTI + 3u]);
  let zq = u32(clamp((J[o + ZI] - cam.clip.x) / (cam.clip.y - cam.clip.x), 0.0, 1.0) * f32((1u << db) - 1u));
  var k = offs[i];
  for (var y = y0; y < y1; y++) {
    for (var x = x0; x < x1; x++) {
      if (k < cam.cnt.w) { keys[k] = ((y * cam.dims.z + x) << db) | zq; vals[k] = i; }
      k++;
    }
  }
}`;

  /* digit counts are stored digit-major with a fixed stride (blocks for the full key cap), so the
     count scan has a fixed size; blocks past the live key count stay zero from the per-pass clear */
  const rsWGSL = `
struct RS { shift: u32, stride: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> R: RS;
@group(0) @binding(1) var<storage, read> kin: array<u32>;
@group(0) @binding(2) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;
var<workgroup> h: array<atomic<u32>, 16>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let blk = w.x + w.y * 65535u;
  if (l.x < 16u) { atomicStore(&h[l.x], 0u); }
  workgroupBarrier();
  let i = blk * 256u + l.x;
  if (i < info[0]) { atomicAdd(&h[(kin[i] >> R.shift) & 15u], 1u); }
  workgroupBarrier();
  if (l.x < 16u && blk < info[2]) { cnt[l.x * R.stride + blk] = atomicLoad(&h[l.x]); }
}`;
  const rsScatterWGSL = `
struct RS { shift: u32, stride: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> R: RS;
@group(0) @binding(1) var<storage, read> kin: array<u32>;
@group(0) @binding(2) var<storage, read> vin: array<u32>;
@group(0) @binding(3) var<storage, read_write> kout: array<u32>;
@group(0) @binding(4) var<storage, read_write> vout: array<u32>;
@group(0) @binding(5) var<storage, read> cnt: array<u32>;
@group(0) @binding(6) var<storage, read> info: array<u32>;
var<workgroup> sc: array<u32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let blk = w.x + w.y * 65535u;
  let i = blk * 256u + l.x;
  let valid = i < info[0];
  var k = 0u; var v = 0u;
  if (valid) { k = kin[i]; v = vin[i]; }
  let d = (k >> R.shift) & 15u;
  var rank = 0u;
  for (var dig = 0u; dig < 16u; dig++) {
    let f = select(0u, 1u, valid && d == dig);
    sc[l.x] = f;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
      var add = 0u;
      if (l.x >= off) { add = sc[l.x - off]; }
      workgroupBarrier();
      sc[l.x] = sc[l.x] + add;
      workgroupBarrier();
    }
    if (f == 1u) { rank = sc[l.x] - 1u; }
    workgroupBarrier();
  }
  if (valid) { let dst = cnt[d * R.stride + blk] + rank; kout[dst] = k; vout[dst] = v; }
}`;

  const rangesWGSL = `${camStruct}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> ranges: array<u32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 256u; let K = info[0]; let db = cam.cnt.z;
  if (i >= K) { return; }
  let tile = keys[i] >> db;
  if (i == 0u) { ranges[2u * tile] = 0u; }
  else { let prev = keys[i - 1u] >> db; if (prev != tile) { ranges[2u * prev + 1u] = i; ranges[2u * tile] = i; } }
  if (i == K - 1u) { ranges[2u * tile + 1u] = K; }
}`;

  const BATCH = 64;
  const renderWGSL = type => `${header(type)}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> J: array<f32>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> ranges: array<u32>;
@group(0) @binding(4) var<storage, read_write> img: array<f32>;
@group(0) @binding(5) var<storage, read_write> tfin: array<f32>;
@group(0) @binding(6) var<storage, read_write> lastc: array<u32>;
var<workgroup> sh: array<array<f32, JS>, ${BATCH}>;
${windowFns[type]}
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let W = cam.dims.x; let H = cam.dims.y;
  let tile = w.y * cam.dims.z + w.x;
  let px = w.x * 16u + l.x; let py = w.y * 16u + l.y;
  let inside = px < W && py < H;
  let r0 = ranges[2u * tile]; let r1 = ranges[2u * tile + 1u];
  let pix = vec2f(f32(px) + 0.5, f32(py) + 0.5);
  var T = 1.0; var C = vec3f(0.0); var done = !inside; var last = 0u;
  let total = r1 - r0;
  let nb = (total + ${BATCH - 1}u) / ${BATCH}u;
  for (var bi = 0u; bi < nb; bi++) {
    workgroupBarrier();
    if (li < ${BATCH}u) {
      let k = r0 + bi * ${BATCH}u + li;
      if (k < r1) { let id = vals[k]; for (var f = 0u; f < JS; f++) { sh[li][f] = J[id * JS + f]; } }
    }
    workgroupBarrier();
    let cnt = min(${BATCH}u, total - bi * ${BATCH}u);
    for (var j = 0u; j < cnt; j++) {
      if (done) { break; }
      let q = sh[j];
      let a = min(0.99, q[OPI] * evalW(q, pix));
      if (a < 1.0 / 255.0) { continue; }
      let tt = T * (1.0 - a);
      if (tt < 0.0001) { done = true; break; }
      C += vec3f(q[COLI], q[COLI + 1u], q[COLI + 2u]) * a * T;
      T = tt; last = bi * ${BATCH}u + j + 1u;
    }
  }
  if (inside) {
    let p = py * W + px;
    let bg = vec3f(cam.misc.y, cam.misc.z, cam.misc.w);
    let o = C + T * bg;
    img[4u * p] = o.x; img[4u * p + 1u] = o.y; img[4u * p + 2u] = o.z; img[4u * p + 3u] = 1.0;
    tfin[p] = T; lastc[p] = last;
  }
}`;

  const packWGSL = `${camStruct}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read_write> gt: array<u32>;
@group(0) @binding(3) var<uniform> U: vec4<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let p = g.x; let n = cam.dims.x * cam.dims.y;
  if (p >= n) { return; }
  let c = clamp(vec3f(img[4u * p], img[4u * p + 1u], img[4u * p + 2u]), vec3f(0.0), vec3f(1.0));
  gt[U.x + p] = u32(c.x * 255.0 + 0.5) | (u32(c.y * 255.0 + 0.5) << 8u) | (u32(c.z * 255.0 + 0.5) << 16u) | (255u << 24u);
}`;

  /* loss: 0.8 L1 + 0.2 L2 per channel, mean over pixels; gradient to dLdC */
  const lossWGSL = `${camStruct}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read> gt: array<u32>;
@group(0) @binding(3) var<storage, read_write> dLdC: array<f32>;
@group(0) @binding(4) var<storage, read_write> lossv: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> U: vec4<u32>;
var<workgroup> red: array<f32, 256>;
fn atomicAddF(idx: u32, v: f32) {
  var old = atomicLoad(&lossv[idx]);
  loop {
    let nv = bitcast<u32>(bitcast<f32>(old) + v);
    let r = atomicCompareExchangeWeak(&lossv[idx], old, nv);
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let p = g.x; let n = cam.dims.x * cam.dims.y;
  var lv = 0.0;
  if (p < n) {
    let u = gt[U.x + p];
    let G = vec3f(f32(u & 255u), f32((u >> 8u) & 255u), f32((u >> 16u) & 255u)) / 255.0;
    let C = vec3f(img[4u * p], img[4u * p + 1u], img[4u * p + 2u]);
    let d = C - G;
    let inv = 1.0 / f32(n * 3u);
    let gr = (0.8 * sign(d) + 0.4 * d) * inv;
    dLdC[4u * p] = gr.x; dLdC[4u * p + 1u] = gr.y; dLdC[4u * p + 2u] = gr.z;
    lv = dot(0.8 * abs(d) + 0.2 * d * d, vec3f(1.0)) * inv;
  }
  red[l.x] = lv;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (l.x < s) { red[l.x] += red[l.x + s]; } workgroupBarrier(); }
  if (l.x == 0u) { atomicAddF(0u, red[0]); }
}`;

  /* ---- backward render: per-pixel reverse traversal, float atomics into G2 ---- */
  const backPixel = [
// gaussian: G2 = [dmean2d.x, dmean2d.y, dA, dB, dC, dop, dR, dG, dB]
`      let d = pix - vec2f(q[0], q[1]);
      let power = -0.5 * (q[2] * d.x * d.x + q[4] * d.y * d.y) - q[3] * d.x * d.y;
      let G = exp(power);
      let gG = q[OPI] * dLda * G;
      add(gb + 0u, gG * (q[2] * d.x + q[3] * d.y));
      add(gb + 1u, gG * (q[3] * d.x + q[4] * d.y));
      add(gb + 2u, -0.5 * gG * d.x * d.x);
      add(gb + 3u, -gG * d.x * d.y);
      add(gb + 4u, -0.5 * gG * d.y * d.y);
      add(gb + 5u, G * dLda);
      add(gb + 6u, gcol.x); add(gb + 7u, gcol.y); add(gb + 8u, gcol.z);`,
// triangle: G2 = [a.xy, b.xy, c.xy, dr, dop, dR, dG, dB]
`      var V: array<vec2f, 3>; V[0] = vec2f(q[0], q[1]); V[1] = vec2f(q[2], q[3]); V[2] = vec2f(q[4], q[5]);
      let s = q[10];
      var dm = 1e9; var kk = 0u;
      for (var e = 0u; e < 3u; e++) {
        let P = V[e]; let Q = V[select(e + 1u, 0u, e == 2u)];
        let dd = s * cross2(Q - P, pix - P) * q[6u + e];
        if (dd < dm) { dm = dd; kk = e; }
      }
      let w = pow(dm / q[9], cam.clip.z);
      let gw = q[OPI] * dLda;
      let gm = gw * cam.clip.z * w / max(dm, 1e-8);
      add(gb + 6u, -gw * cam.clip.z * w / q[9]);
      let P = V[kk]; let ki = select(kk + 1u, 0u, kk == 2u); let Q = V[ki];
      let e = Q - P; let u = pix - P; let L = length(e); let c = cross2(e, u);
      let dcP = vec2f(e.y - u.y, u.x - e.x); let dcQ = vec2f(u.y, -u.x);
      let gP = s * (dcP / L + c * e / (L * L * L));
      let gQ = s * (dcQ / L - c * e / (L * L * L));
      add(gb + 2u * kk, gm * gP.x); add(gb + 2u * kk + 1u, gm * gP.y);
      add(gb + 2u * ki, gm * gQ.x); add(gb + 2u * ki + 1u, gm * gQ.y);
      add(gb + 7u, w * dLda);
      add(gb + 8u, gcol.x); add(gb + 9u, gcol.y); add(gb + 10u, gcol.z);`,
// convex: G2 = [p0..p5 xy (12), dop, dR, dG, dB]
`      let n = u32(q[12]); let Sz = q[13]; let s = q[14]; let dl = cam.clip.w; let sg = cam.misc.x;
      let packed = u32(q[20]);
      var L: array<f32, 6>; var m = -1e9;
      for (var j = 0u; j < n; j++) {
        let k = select(j + 1u, 0u, j + 1u == n);
        let P = vec2f(q[2u * j], q[2u * j + 1u]); let Q = vec2f(q[2u * k], q[2u * k + 1u]);
        let e = Q - P; let len = length(e);
        var lv = -1e9;
        if (len > 1e-6) { lv = -s * cross2(e, pix - P) / (len * Sz); }
        L[j] = lv; m = max(m, lv);
      }
      var acc = 0.0;
      for (var j = 0u; j < n; j++) { acc += exp(dl * (L[j] - m)); }
      let phi = m + log(acc) / dl;
      let I = 1.0 / (1.0 + exp(sg * phi));
      let gphi = q[OPI] * dLda * (-sg * I * (1.0 - I));
      // accumulate per input point (hull vertex j is point ij), then add every slot once
      var gv: array<vec2f, 6>; var gSz = 0.0;
      for (var j = 0u; j < n; j++) {
        if (L[j] < -1e8) { continue; }
        let k = select(j + 1u, 0u, j + 1u == n);
        let P = vec2f(q[2u * j], q[2u * j + 1u]); let Q = vec2f(q[2u * k], q[2u * k + 1u]);
        let gL = gphi * exp(dl * (L[j] - m)) / acc;
        gSz += gL * (-L[j] / Sz);
        let e = Q - P; let u = pix - P; let len = length(e); let c = cross2(e, u);
        let f = -s / Sz;
        let gP = f * (vec2f(e.y - u.y, u.x - e.x) / len + c * e / (len * len * len));
        let gQ = f * (vec2f(u.y, -u.x) / len - c * e / (len * len * len));
        let ij = (packed >> (3u * j)) & 7u; let ik = (packed >> (3u * k)) & 7u;
        gv[ij] += gL * gP; gv[ik] += gL * gQ;
      }
      for (var k = 0u; k < 6u; k++) { add(gb + 2u * k, gv[k].x); add(gb + 2u * k + 1u, gv[k].y); }
      add(gb + 12u, I * dLda);
      add(gb + 13u, gcol.x); add(gb + 14u, gcol.y); add(gb + 15u, gcol.z);
      add(gb + 16u, gSz);`];
  /* Gradients are accumulated per batch in workgroup memory (acc, one GS-float record per batch slot) and
     flushed to G2 by all threads in parallel at the end of each batch. Adding straight into G2 made every
     warp wait out 9-17 dependent global compare-exchange round trips per primitive, which dominated the step. */
  const backwardWGSL = type => `${header(type)}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> J: array<f32>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> ranges: array<u32>;
@group(0) @binding(4) var<storage, read> tfin: array<f32>;
@group(0) @binding(5) var<storage, read> lastc: array<u32>;
@group(0) @binding(6) var<storage, read> dLdC: array<f32>;
@group(0) @binding(7) var<storage, read_write> G2: array<atomic<u32>>;
var<workgroup> sh: array<array<f32, JS>, ${BATCH}>;
var<workgroup> shid: array<u32, ${BATCH}>;
var<workgroup> acc: array<atomic<u32>, ${BATCH}u * GS>;
fn addGlobal(idx: u32, v: f32) {
  var old = atomicLoad(&G2[idx]);
  loop {
    let r = atomicCompareExchangeWeak(&G2[idx], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}
fn addShared(idx: u32, v: f32) {
  var old = atomicLoad(&acc[idx]);
  loop {
    let r = atomicCompareExchangeWeak(&acc[idx], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}
/* idx: batch slot * GS + gradient slot */
fn add(idx: u32, v: f32) {
  if (v == 0.0 || v != v) { return; }
  addShared(idx, v);
}
${windowFns[type]}
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let W = cam.dims.x; let H = cam.dims.y;
  let tile = w.y * cam.dims.z + w.x;
  let px = w.x * 16u + l.x; let py = w.y * 16u + l.y;
  let inside = px < W && py < H;
  let p = min(py * W + px, W * H - 1u);
  let r0 = ranges[2u * tile]; let r1 = ranges[2u * tile + 1u];
  let pix = vec2f(f32(px) + 0.5, f32(py) + 0.5);
  let Tfin = tfin[p]; var T = Tfin;
  var last = lastc[p];
  if (!inside) { last = 0u; }
  let dpix = vec3f(dLdC[4u * p], dLdC[4u * p + 1u], dLdC[4u * p + 2u]);
  let bg = vec3f(cam.misc.y, cam.misc.z, cam.misc.w);
  let bgdot = dot(bg, dpix);
  var accum = vec3f(0.0); var lastA = 0.0; var lastCol = vec3f(0.0);
  let total = r1 - r0;
  let nb = (total + ${BATCH - 1}u) / ${BATCH}u;
  for (var bb = 0u; bb < nb; bb++) {
    let bi = nb - 1u - bb;
    workgroupBarrier();
    if (li < ${BATCH}u) {
      let k = r0 + bi * ${BATCH}u + li;
      if (k < r1) { let id = vals[k]; shid[li] = id; for (var f = 0u; f < JS; f++) { sh[li][f] = J[id * JS + f]; } }
    }
    workgroupBarrier();
    let cnt = min(${BATCH}u, total - bi * ${BATCH}u);
    for (var jj = 0u; jj < cnt; jj++) {
      let j = cnt - 1u - jj;
      let rel = bi * ${BATCH}u + j;
      if (rel >= last) { continue; }
      var q = sh[j];
      let wv = evalW(q, pix);
      let araw = q[OPI] * wv;
      let a = min(0.99, araw);
      if (a < 1.0 / 255.0) { continue; }
      T = T / (1.0 - a);
      let col = vec3f(q[COLI], q[COLI + 1u], q[COLI + 2u]);
      let gcol = a * T * dpix;
      accum = lastA * lastCol + (1.0 - lastA) * accum;
      lastA = a; lastCol = col;
      var dLda = dot(col - accum, dpix) * T + (-Tfin / (1.0 - a)) * bgdot;
      if (araw > 0.99) { dLda = 0.0; }
      let gb = j * GS;
${backPixel[type]}
    }
    // flush this batch's gradients, spread over all threads; also resets acc for the next batch
    workgroupBarrier();
    for (var e = li; e < cnt * GS; e += 256u) {
      let v = bitcast<f32>(atomicExchange(&acc[e], 0u));
      if (v != 0.0) { addGlobal(shid[e / GS] * GS + e % GS, v); }
    }
  }
}`;

  /* ---- 2D gradients -> parameter gradients ---- */
  const backprojBodies = [
`  let mean = vec3f(P[b], P[b + 1u], P[b + 2u]);
  let t = toView(mean);
  if (t.z < cam.clip.x) { return; }
  let ls = vec3f(P[b + 3u], P[b + 4u], P[b + 5u]); let s = exp(ls);
  let q0 = vec4f(P[b + 6u], P[b + 7u], P[b + 8u], P[b + 9u]); let ql = max(length(q0), 1e-12); let qn = q0 / ql;
  let R = quatMat(qn);
  let M = mat3x3<f32>(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let Sig = M * transpose(M);
  let Wm = W3();
  let Sv = Wm * Sig * transpose(Wm);
  let j0 = jrow0(t); let j1 = jrow1(t);
  let a = dot(j0, Sv * j0) + 0.3; let bb = dot(j0, Sv * j1); let c = dot(j1, Sv * j1) + 0.3;
  let det = a * c - bb * bb;
  if (det <= 1e-12) { return; }
  let Q = mat2x2<f32>(c / det, -bb / det, -bb / det, a / det);
  let Gq = mat2x2<f32>(g(2u), 0.5 * g(3u), 0.5 * g(3u), g(4u));
  let Gc = (Q * Gq * Q) * -1.0;
  // d cov2 / d Sv : cov2 = Jm Sv Jm^T, Jm rows j0, j1
  let gSv = outerP(j0, j0) * Gc[0][0] + (outerP(j0, j1) + outerP(j1, j0)) * Gc[0][1] + outerP(j1, j1) * Gc[1][1];
  let gSig = transpose(Wm) * gSv * Wm;
  // dL/dJm rows = 2 * Gc * Jm * Sv
  let sj0 = Sv * j0; let sj1 = Sv * j1;
  let gJ0 = 2.0 * (Gc[0][0] * sj0 + Gc[1][0] * sj1);
  let gJ1 = 2.0 * (Gc[0][1] * sj0 + Gc[1][1] * sj1);
  let fx = cam.intr.x; let fy = cam.intr.y; let z2 = t.z * t.z; let z3 = z2 * t.z;
  var gt = vec3f(0.0);
  gt.x += gJ0.z * (-fx / z2);
  gt.y += gJ1.z * (fy / z2);
  gt.z += gJ0.x * (-fx / z2) + gJ0.z * (2.0 * fx * t.x / z3) + gJ1.y * (fy / z2) + gJ1.z * (-2.0 * fy * t.y / z3);
  gt += g(0u) * j0 + g(1u) * j1;
  let gmean = transpose(Wm) * gt;
  // Sig = M M^T -> dM = 2 gSig M ; M_ik = R_ik s_k
  let gM = 2.0 * gSig * M;
  let gs = vec3f(dot(R[0], gM[0]), dot(R[1], gM[1]), dot(R[2], gM[2]));
  let gR = mat3x3<f32>(gM[0] * s.x, gM[1] * s.y, gM[2] * s.z);
  let w = qn.x; let x = qn.y; let y = qn.z; let z = qn.w;
  // R[col][row]
  let r00 = gR[0][0]; let r10 = gR[0][1]; let r20 = gR[0][2];
  let r01 = gR[1][0]; let r11 = gR[1][1]; let r21 = gR[1][2];
  let r02 = gR[2][0]; let r12 = gR[2][1]; let r22 = gR[2][2];
  let gw = 2.0 * (-z * r01 + y * r02 + z * r10 - x * r12 - y * r20 + x * r21);
  let gx = 2.0 * (y * r01 + z * r02 + y * r10 - 2.0 * x * r11 - w * r12 + z * r20 + w * r21 - 2.0 * x * r22);
  let gy = 2.0 * (-2.0 * y * r00 + x * r01 + w * r02 + x * r10 + z * r12 - w * r20 + z * r21 - 2.0 * y * r22);
  let gz = 2.0 * (-2.0 * z * r00 - w * r01 + x * r02 + w * r10 - 2.0 * z * r11 + y * r12 + x * r20 + y * r21);
  let gqn = vec4f(gw, gx, gy, gz);
  let gq = (gqn - qn * dot(qn, gqn)) / ql;
  let op = sigm(P[b + 13u]);
  PG[b] = gmean.x; PG[b + 1u] = gmean.y; PG[b + 2u] = gmean.z;
  PG[b + 3u] = gs.x * s.x; PG[b + 4u] = gs.y * s.y; PG[b + 5u] = gs.z * s.z;
  PG[b + 6u] = gq.x; PG[b + 7u] = gq.y; PG[b + 8u] = gq.z; PG[b + 9u] = gq.w;
  PG[b + 10u] = g(6u); PG[b + 11u] = g(7u); PG[b + 12u] = g(8u);
  PG[b + 13u] = g(5u) * op * (1.0 - op);
  dens[i] += length(vec2f(g(0u), g(1u)));`,
`  var T: array<vec3f, 3>;
  for (var k = 0u; k < 3u; k++) { T[k] = toView(vec3f(P[b + 3u * k], P[b + 3u * k + 1u], P[b + 3u * k + 2u])); if (T[k].z < cam.clip.x) { return; } }
  let a = toPix(T[0]); let bq = toPix(T[1]); let c = toPix(T[2]);
  let cr = cross2(bq - a, c - a);
  if (abs(cr) < 1e-4) { return; }
  let s = sign(cr);
  let lab = length(bq - a); let lbc = length(c - bq); let lca = length(a - c);
  let per = lab + lbc + lca; let r = abs(cr) / per;
  let dcrb = vec2f((c - a).y, -(c - a).x); let dcrc = vec2f(-(bq - a).y, (bq - a).x); let dcra = -(dcrb + dcrc);
  let dpa = (a - bq) / lab + (a - c) / lca; let dpb = (bq - a) / lab + (bq - c) / lbc; let dpc = (c - bq) / lbc + (c - a) / lca;
  let gr = g(6u);
  var g2: array<vec2f, 3>;
  g2[0] = vec2f(g(0u), g(1u)) + gr * (s * dcra / per - r * dpa / per);
  g2[1] = vec2f(g(2u), g(3u)) + gr * (s * dcrb / per - r * dpb / per);
  g2[2] = vec2f(g(4u), g(5u)) + gr * (s * dcrc / per - r * dpc / per);
  let Wt = transpose(W3());
  var gsum = 0.0;
  for (var k = 0u; k < 3u; k++) {
    let gw = Wt * (g2[k].x * jrow0(T[k]) + g2[k].y * jrow1(T[k]));
    PG[b + 3u * k] = gw.x; PG[b + 3u * k + 1u] = gw.y; PG[b + 3u * k + 2u] = gw.z;
    gsum += length(g2[k]);
  }
  let op = sigm(P[b + 12u]);
  PG[b + 9u] = g(8u); PG[b + 10u] = g(9u); PG[b + 11u] = g(10u);
  PG[b + 12u] = g(7u) * op * (1.0 - op);
  dens[i] += gsum;`,
`  let Wt = transpose(W3());
  var gsum = 0.0;
  var gp: array<vec2f, 6>;
  for (var k = 0u; k < 6u; k++) { gp[k] = vec2f(g(2u * k), g(2u * k + 1u)); }
  let o = i * JS;
  let n = u32(J[o + 12u]); let Sz = J[o + 13u]; let sgn = J[o + 14u]; let packed = u32(J[o + 20u]);
  if (n >= 3u && Sz > 0.0) {
    // Sz = sqrt(|area2| / 2), area2 = sum cross2(H_k, H_k+1)
    let gS = g(16u) * sgn / (4.0 * Sz);
    for (var k = 0u; k < n; k++) {
      let kn = select(k + 1u, 0u, k + 1u == n); let kp = select(k - 1u, n - 1u, k == 0u);
      let Hn = vec2f(J[o + 2u * kn], J[o + 2u * kn + 1u]); let Hp = vec2f(J[o + 2u * kp], J[o + 2u * kp + 1u]);
      let ik = (packed >> (3u * k)) & 7u;
      gp[ik] += gS * vec2f(Hn.y - Hp.y, Hp.x - Hn.x);
    }
  }
  for (var k = 0u; k < 6u; k++) {
    let t = toView(vec3f(P[b + 3u * k], P[b + 3u * k + 1u], P[b + 3u * k + 2u]));
    if (t.z < cam.clip.x) { return; }
    let g2 = gp[k];
    let gw = Wt * (g2.x * jrow0(t) + g2.y * jrow1(t));
    PG[b + 3u * k] = gw.x; PG[b + 3u * k + 1u] = gw.y; PG[b + 3u * k + 2u] = gw.z;
    gsum += length(g2);
  }
  let op = sigm(P[b + 21u]);
  PG[b + 18u] = g(13u); PG[b + 19u] = g(14u); PG[b + 20u] = g(15u);
  PG[b + 21u] = g(12u) * op * (1.0 - op);
  dens[i] += gsum;`];
  const backprojWGSL = type => `${header(type)}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> G2: array<u32>;
@group(0) @binding(3) var<storage, read_write> PG: array<f32>;
@group(0) @binding(4) var<storage, read_write> dens: array<f32>;
${type === 2 ? "@group(0) @binding(5) var<storage, read> J: array<f32>;" : ""}
${viewFns}
var<private> gbase: u32;
fn g(k: u32) -> f32 { return bitcast<f32>(G2[gbase + k]); }
fn outerP(a: vec3f, b: vec3f) -> mat3x3<f32> { return mat3x3<f32>(a * b.x, a * b.y, a * b.z); }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= cam.cnt.x) { return; }
  let b = i * PS; gbase = i * GS;
  for (var k = 0u; k < PS; k++) { PG[b + k] = 0.0; }
${backprojBodies[type]}
}`;

  const adamWGSL = type => `
struct AU { lr: array<vec4f, 6>, hp: vec4f, ip: vec4<u32> };   // hp: beta1 beta2 eps opFloor ; ip: N step clampColors -
@group(0) @binding(0) var<uniform> U: AU;
@group(0) @binding(1) var<storage, read_write> P: array<f32>;
@group(0) @binding(2) var<storage, read> PG: array<f32>;
@group(0) @binding(3) var<storage, read_write> M1: array<f32>;
@group(0) @binding(4) var<storage, read_write> M2: array<f32>;
const PS: u32 = ${PS_OF[type]}u;
const OPP: u32 = ${[13, 12, 21][type]}u;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x + gid.y * 65535u * 256u;
  if (e >= U.ip.x * PS) { return; }
  let k = e % PS;
  let lr = U.lr[k / 4u][k % 4u];
  if (lr == 0.0) { return; }
  var gr = PG[e];
  if (gr != gr) { gr = 0.0; }
  let b1 = U.hp.x; let b2 = U.hp.y; let st = f32(U.ip.y);
  let m = b1 * M1[e] + (1.0 - b1) * gr; let v = b2 * M2[e] + (1.0 - b2) * gr * gr;
  M1[e] = m; M2[e] = v;
  let mh = m / (1.0 - pow(b1, st)); let vh = v / (1.0 - pow(b2, st));
  var p = P[e] - lr * mh / (sqrt(vh) + U.hp.z);
  if (k == OPP) { p = clamp(max(p, U.hp.w), -12.0, 12.0); }
  P[e] = p;
}`;

  /* ---------------------------------------------------------------- JS */
  /* timestamps: request timestamp-query for profiling (tests/bench.html) */
  async function createDevice({ timestamps = false } = {}) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found.");
    const lim = adapter.limits;
    const requiredFeatures = timestamps && adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
    return adapter.requestDevice({ requiredFeatures, requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(10, lim.maxStorageBuffersPerShaderStage),
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
    } });
  }
  const U = GPUBufferUsage;
  function buf(dev, size, usage, label) { return dev.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label }); }
  function dispatchDims(groups) { return groups <= 65535 ? [Math.max(1, groups), 1] : [65535, Math.ceil(groups / 65535)]; }

  class Scan {
    constructor(dev, pipes, data, maxN) {
      this.dev = dev; this.pipes = pipes; this.levels = [];
      let n = maxN, d = data;
      while (true) {
        const nb = Math.ceil(n / 512);
        const sums = buf(dev, nb * 4, U.STORAGE, "scanSums");
        const uni = buf(dev, 16, U.UNIFORM | U.COPY_DST, "scanU");
        const bg = dev.createBindGroup({ layout: pipes.scanBlock.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: d } }, { binding: 1, resource: { buffer: sums } }, { binding: 2, resource: { buffer: uni } }] });
        const bg2 = dev.createBindGroup({ layout: pipes.addSums.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: d } }, { binding: 1, resource: { buffer: sums } }, { binding: 2, resource: { buffer: uni } }] });
        this.levels.push({ uni, bg, bg2 });
        if (nb <= 1) break;
        n = nb; d = sums;
      }
    }
    /* exclusive scan of the first n entries: block scans down the levels, then add block sums back up */
    encode(enc, n) {
      if (n <= 0) return;
      const sizes = [n];
      while (sizes[sizes.length - 1] > 512) sizes.push(Math.ceil(sizes[sizes.length - 1] / 512));
      if (sizes.length > this.levels.length) throw new Error("scan too large");
      sizes.forEach((m, l) => this.dev.queue.writeBuffer(this.levels[l].uni, 0, new Uint32Array([m, 0, 0, 0])));
      for (let l = 0; l < sizes.length; l++) {
        const p = enc.beginComputePass(); p.setPipeline(this.pipes.scanBlock); p.setBindGroup(0, this.levels[l].bg);
        p.dispatchWorkgroups(...dispatchDims(Math.ceil(sizes[l] / 512))); p.end();
      }
      for (let l = sizes.length - 2; l >= 0; l--) {
        const p = enc.beginComputePass(); p.setPipeline(this.pipes.addSums); p.setBindGroup(0, this.levels[l].bg2);
        p.dispatchWorkgroups(...dispatchDims(Math.ceil(sizes[l] / 512))); p.end();
      }
    }
  }

  const pipeCache = new WeakMap();
  function pipelines(dev, type) {
    let byType = pipeCache.get(dev); if (!byType) { byType = {}; pipeCache.set(dev, byType); }
    if (byType[type]) return byType[type];
    const mk = (code, entryPoint = "main", label) => {
      const module = dev.createShaderModule({ code, label });
      return dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint }, label });
    };
    const scanMod = dev.createShaderModule({ code: scanWGSL, label: "scan" });
    const p = {
      project: mk(projectWGSL(type), "main", "project" + type),
      duplicate: mk(duplicateWGSL(type), "main", "dup" + type),
      render: mk(renderWGSL(type), "main", "render" + type),
      backward: mk(backwardWGSL(type), "main", "backward" + type),
      backproj: mk(backprojWGSL(type), "main", "backproj" + type),
      adam: mk(adamWGSL(type), "main", "adam" + type),
      scanBlock: dev.createComputePipeline({ layout: "auto", compute: { module: scanMod, entryPoint: "scanBlock" } }),
      addSums: dev.createComputePipeline({ layout: "auto", compute: { module: scanMod, entryPoint: "addSums" } }),
      args: mk(argsWGSL, "main", "args"), rsCount: mk(rsWGSL, "main", "rsCount"), rsScatter: mk(rsScatterWGSL, "main", "rsScatter"),
      ranges: mk(rangesWGSL, "main", "ranges"), pack: mk(packWGSL, "main", "pack"), loss: mk(lossWGSL, "main", "loss"),
    };
    byType[type] = p;
    return p;
  }
  const bindAll = (dev, pipe, list) => dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: list.map((b, i) => ({ binding: i, resource: { buffer: b } })) });

  /* one rasterizer instance: fixed primitive count N, image W x H */
  class Raster {
    constructor(dev, type, N, W, H, { train = false, keyCap } = {}) {
      this.dev = dev; this.type = type; this.N = N; this.W = W; this.H = H; this.train = train;
      const PSz = PS_OF[type], JSz = JS_OF[type];
      this.pipes = pipelines(dev, type);
      this.tilesX = Math.ceil(W / 16); this.tilesY = Math.ceil(H / 16);
      const nt = this.tilesX * this.tilesY;
      this.depthBits = 32 - Math.ceil(Math.log2(nt + 1));
      this.cap = Math.min(keyCap || Math.max(1 << 20, N * 12), MAX_BLOCKS * 256);
      const S = U.STORAGE, D = U.COPY_DST, Sr = U.COPY_SRC;
      this.cam = buf(dev, CAM_BYTES, U.UNIFORM | D, "cam");
      this.P = buf(dev, N * PSz * 4, S | D | Sr, "params");
      this.J = buf(dev, N * JSz * 4, S, "proj");
      this.tiles = buf(dev, N * 4, S | Sr, "tiles");
      this.offs = buf(dev, N * 4, S | D, "offs");
      this.info = buf(dev, 16, S | Sr, "keyInfo");
      this.ind = buf(dev, 16, S | U.INDIRECT, "sortDispatch");
      this.stage = buf(dev, 16, U.MAP_READ | D, "stage");
      this.keysA = buf(dev, this.cap * 4, S, "keysA"); this.valsA = buf(dev, this.cap * 4, S, "valsA");
      this.keysB = buf(dev, this.cap * 4, S, "keysB"); this.valsB = buf(dev, this.cap * 4, S, "valsB");
      const nbMax = this.nbMax = Math.ceil(this.cap / 256);
      this.cnt = buf(dev, 16 * nbMax * 4, S | D, "rsCnt");
      this.ranges = buf(dev, nt * 2 * 4, S | D, "ranges");
      this.img = buf(dev, W * H * 16, S | Sr, "img");
      this.tfin = buf(dev, W * H * 4, S, "tfin");
      this.lastc = buf(dev, W * H * 4, S, "lastc");
      this.imgStage = buf(dev, W * H * 16, U.MAP_READ | D, "imgStage");
      this.scanTiles = new Scan(dev, this.pipes, this.offs, N);
      this.scanCnt = new Scan(dev, this.pipes, this.cnt, 16 * nbMax);
      this.rsU = Array.from({ length: 8 }, (_, pass) => {
        const b = buf(dev, 16, U.UNIFORM | D, "rsU"); dev.queue.writeBuffer(b, 0, new Uint32Array([pass * 4, nbMax, 0, 0])); return b;
      });
      const p = this.pipes;
      this.bg = {
        project: bindAll(dev, p.project, [this.cam, this.P, this.J, this.tiles]),
        args: bindAll(dev, p.args, [this.cam, this.offs, this.tiles, this.info, this.ind]),
        duplicate: bindAll(dev, p.duplicate, [this.cam, this.J, this.tiles, this.offs, this.keysA, this.valsA]),
        ranges: bindAll(dev, p.ranges, [this.cam, this.keysA, this.ranges, this.info]),
        render: bindAll(dev, p.render, [this.cam, this.J, this.valsA, this.ranges, this.img, this.tfin, this.lastc]),
      };
      // radix uniforms differ per pass: build per-pass bind groups
      this.bg.rsCount = []; this.bg.rsScatter = [];
      for (let pass = 0; pass < 8; pass++) {
        const src = pass % 2 === 0 ? [this.keysA, this.valsA] : [this.keysB, this.valsB];
        const dst = pass % 2 === 0 ? [this.keysB, this.valsB] : [this.keysA, this.valsA];
        this.bg.rsCount.push(bindAll(dev, p.rsCount, [this.rsU[pass], src[0], this.cnt, this.info]));
        this.bg.rsScatter.push(bindAll(dev, p.rsScatter, [this.rsU[pass], src[0], src[1], dst[0], dst[1], this.cnt, this.info]));
      }
      if (train) {
        this.dLdC = buf(dev, W * H * 16, S, "dLdC");
        this.G2 = buf(dev, N * 20 * 4, S | D | Sr, "G2");
        this.PG = buf(dev, N * PSz * 4, S | Sr, "PG");
        this.M1 = buf(dev, N * PSz * 4, S | D, "M1"); this.M2 = buf(dev, N * PSz * 4, S | D, "M2");
        this.dens = buf(dev, N * 4, S | D | Sr, "dens");
        this.lossv = buf(dev, 16, S | D | Sr, "loss");
        this.lossStage = buf(dev, 16, U.MAP_READ | D, "lossStage");
        this.lossU = buf(dev, 16, U.UNIFORM | D, "lossU");
        this.adamU = buf(dev, 128, U.UNIFORM | D, "adamU");
        this.bg.backward = bindAll(dev, p.backward, [this.cam, this.J, this.valsA, this.ranges, this.tfin, this.lastc, this.dLdC, this.G2]);
        this.bg.backproj = bindAll(dev, p.backproj, type === 2 ? [this.cam, this.P, this.G2, this.PG, this.dens, this.J] : [this.cam, this.P, this.G2, this.PG, this.dens]);
        this.bg.adam = bindAll(dev, p.adam, [this.adamU, this.P, this.PG, this.M1, this.M2]);
      }
      this.sigmaTri = 1; this.deltaCvx = 6; this.sigmaCvx = 18; this.bg3 = [0.059, 0.071, 0.086];
    }
    setCamera(cam) {
      const a = new ArrayBuffer(CAM_BYTES), f = new Float32Array(a), u = new Uint32Array(a);
      f.set(cam.view, 0);
      f[16] = cam.fx; f[17] = cam.fy; f[18] = cam.cx; f[19] = cam.cy;
      u[20] = this.W; u[21] = this.H; u[22] = this.tilesX; u[23] = this.tilesY;
      f[24] = cam.near; f[25] = cam.far; f[26] = this.sigmaTri; f[27] = this.deltaCvx;
      f[28] = this.sigmaCvx; f[29] = this.bg3[0]; f[30] = this.bg3[1]; f[31] = this.bg3[2];
      u[32] = this.N; u[33] = 0; u[34] = this.depthBits; u[35] = this.cap;
      this.dev.queue.writeBuffer(this.cam, 0, a);
    }
    setParams(params) { this.dev.queue.writeBuffer(this.P, 0, params); }
    /* phase 1: project, count tile overlaps, size the sort on the GPU (encodes into enc) */
    encodeProject(enc, cam) {
      this.setCamera(cam);
      let p = enc.beginComputePass(); p.setPipeline(this.pipes.project); p.setBindGroup(0, this.bg.project); p.dispatchWorkgroups(...dispatchDims(Math.ceil(this.N / 256))); p.end();
      enc.copyBufferToBuffer(this.tiles, 0, this.offs, 0, this.N * 4);
      this.scanTiles.encode(enc, this.N);
      p = enc.beginComputePass(); p.setPipeline(this.pipes.args); p.setBindGroup(0, this.bg.args); p.dispatchWorkgroups(1); p.end();
    }
    /* phase 2: sort + rasterize (encodes into enc); sort and ranges are sized by indirect dispatch */
    encodeRaster(enc) {
      enc.clearBuffer(this.ranges);
      let p = enc.beginComputePass(); p.setPipeline(this.pipes.duplicate); p.setBindGroup(0, this.bg.duplicate); p.dispatchWorkgroups(...dispatchDims(Math.ceil(this.N / 256))); p.end();
      for (let pass = 0; pass < 8; pass++) {
        enc.clearBuffer(this.cnt);
        p = enc.beginComputePass(); p.setPipeline(this.pipes.rsCount); p.setBindGroup(0, this.bg.rsCount[pass]); p.dispatchWorkgroupsIndirect(this.ind, 0); p.end();
        this.scanCnt.encode(enc, 16 * this.nbMax);
        p = enc.beginComputePass(); p.setPipeline(this.pipes.rsScatter); p.setBindGroup(0, this.bg.rsScatter[pass]); p.dispatchWorkgroupsIndirect(this.ind, 0); p.end();
      }
      p = enc.beginComputePass(); p.setPipeline(this.pipes.ranges); p.setBindGroup(0, this.bg.ranges); p.dispatchWorkgroupsIndirect(this.ind, 0); p.end();
      p = enc.beginComputePass(); p.setPipeline(this.pipes.render); p.setBindGroup(0, this.bg.render); p.dispatchWorkgroups(this.tilesX, this.tilesY); p.end();
    }
    render(cam) {
      const enc = this.dev.createCommandEncoder(); this.encodeProject(enc, cam); this.encodeRaster(enc); this.dev.queue.submit([enc.finish()]);
    }
    /* key counts of the last projection, for diagnostics; rendering and training never wait on this */
    async readKeyInfo() {
      const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.info, 0, this.stage, 0, 16); this.dev.queue.submit([enc.finish()]);
      await this.stage.mapAsync(GPUMapMode.READ); const v = new Uint32Array(this.stage.getMappedRange().slice(0)); this.stage.unmap();
      return { K: v[0], Kraw: v[1], overflows: v[3] };
    }
    async readImage() {
      const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.img, 0, this.imgStage, 0, this.W * this.H * 16); this.dev.queue.submit([enc.finish()]);
      await this.imgStage.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(this.imgStage.getMappedRange().slice(0)); this.imgStage.unmap();
      return out;
    }
    async readBuffer(src, bytes) {
      const st = this.dev.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
      const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, st, 0, bytes); this.dev.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ); const out = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      return out;
    }
    /* one optimization step against gt image `viewIdx` of gtBuf; encodes and submits without waiting */
    step(cam, gtBuf, viewIdx, adamParams) {
      const dev = this.dev, enc = dev.createCommandEncoder();
      this.encodeProject(enc, cam);
      this.encodeRaster(enc);
      if (!this._lossBG || this._lossGT !== gtBuf) { this._lossBG = bindAll(dev, this.pipes.loss, [this.cam, this.img, gtBuf, this.dLdC, this.lossv, this.lossU]); this._lossGT = gtBuf; }
      dev.queue.writeBuffer(this.lossU, 0, new Uint32Array([viewIdx * this.W * this.H, 0, 0, 0]));
      let p = enc.beginComputePass(); p.setPipeline(this.pipes.loss); p.setBindGroup(0, this._lossBG); p.dispatchWorkgroups(Math.ceil(this.W * this.H / 256)); p.end();
      enc.clearBuffer(this.G2);
      p = enc.beginComputePass(); p.setPipeline(this.pipes.backward); p.setBindGroup(0, this.bg.backward); p.dispatchWorkgroups(this.tilesX, this.tilesY); p.end();
      p = enc.beginComputePass(); p.setPipeline(this.pipes.backproj); p.setBindGroup(0, this.bg.backproj); p.dispatchWorkgroups(...dispatchDims(Math.ceil(this.N / 256))); p.end();
      if (adamParams) {
        dev.queue.writeBuffer(this.adamU, 0, adamParams);
        p = enc.beginComputePass(); p.setPipeline(this.pipes.adam); p.setBindGroup(0, this.bg.adam); p.dispatchWorkgroups(...dispatchDims(Math.ceil(this.N * PS_OF[this.type] / 256))); p.end();
      }
      dev.queue.submit([enc.finish()]);
    }
    encodePack(enc, gtBuf, viewIdx) {
      if (!this._packU) this._packU = buf(this.dev, 16, U.UNIFORM | U.COPY_DST, "packU");
      if (this._packGT !== gtBuf) { this._packBG = bindAll(this.dev, this.pipes.pack, [this.cam, this.img, gtBuf, this._packU]); this._packGT = gtBuf; }
      this.dev.queue.writeBuffer(this._packU, 0, new Uint32Array([viewIdx * this.W * this.H, 0, 0, 0]));
      const p = enc.beginComputePass(); p.setPipeline(this.pipes.pack); p.setBindGroup(0, this._packBG); p.dispatchWorkgroups(Math.ceil(this.W * this.H / 256)); p.end();
    }
    async readLoss(reset = true) {
      const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.lossv, 0, this.lossStage, 0, 4);
      if (reset) enc.clearBuffer(this.lossv);
      this.dev.queue.submit([enc.finish()]);
      await this.lossStage.mapAsync(GPUMapMode.READ); const v = new Float32Array(this.lossStage.getMappedRange().slice(0))[0]; this.lossStage.unmap();
      return v;
    }
    destroy() { for (const k in this) { if (k === "_lossGT" || k === "_packGT") continue; const b = this[k]; if (b && b instanceof GPUBuffer) b.destroy(); } }
  }

  /* --------------------------------------------------- cameras + params */
  function lookAtLH(eye, target, up) {
    const z = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let x = cross(up, z); if (Math.hypot(...x) < 1e-6) x = cross([0, 0, 1], z); x = norm(x);
    const y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  }
  const norm = v => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function makeViews(center, radius, count, W, H, fovDeg = 50) {
    const f = (H / 2) / Math.tan(fovDeg * Math.PI / 360), dist = radius / Math.sin(fovDeg * Math.PI / 360) * 1.05, views = [];
    for (let i = 0; i < count; i++) {
      const y = 1 - 2 * (i + 0.5) / count, r = Math.sqrt(1 - y * y), th = i * Math.PI * (3 - Math.sqrt(5));
      const d = [Math.cos(th) * r, y * 0.92, Math.sin(th) * r];
      const eye = [center[0] + d[0] * dist, center[1] + d[1] * dist, center[2] + d[2] * dist];
      views.push({ view: lookAtLH(eye, center, [0, 1, 0]), fx: f, fy: f, cx: W / 2, cy: H / 2, near: Math.max(1e-3, dist - radius * 1.6), far: dist + radius * 1.6, eye });
    }
    return views;
  }
  /* rotation matrix (columns a1,a2,a3) -> quaternion (w,x,y,z) */
  function matToQuat(c0, c1, c2) {
    const m00 = c0[0], m10 = c0[1], m20 = c0[2], m01 = c1[0], m11 = c1[1], m21 = c1[2], m02 = c2[0], m12 = c2[1], m22 = c2[2];
    const tr = m00 + m11 + m22; let w, x, y, z;
    if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
    else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
    else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
    else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
    return [w, x, y, z];
  }
  const logit = p => Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
  /* convert a shared splat set (pos, ax, sig, col, op) into gaussian params */
  function gaussianParams(S, scale = 1, minSig = 0) {
    const P = new Float32Array(S.n * 16);
    for (let i = 0; i < S.n; i++) {
      const a1 = Array.from(S.ax.subarray(i * 9, i * 9 + 3)), a2 = Array.from(S.ax.subarray(i * 9 + 3, i * 9 + 6));
      let a3 = Array.from(S.ax.subarray(i * 9 + 6, i * 9 + 9));
      if (dot(cross(a1, a2), a3) < 0) a3 = a3.map(v => -v);
      const q = matToQuat(a1, a2, a3), b = i * 16;
      P[b] = S.pos[i * 3]; P[b + 1] = S.pos[i * 3 + 1]; P[b + 2] = S.pos[i * 3 + 2];
      for (let k = 0; k < 3; k++) P[b + 3 + k] = Math.log(Math.max(minSig, S.sig[i * 3 + k] * scale, 1e-7));
      P[b + 6] = q[0]; P[b + 7] = q[1]; P[b + 8] = q[2]; P[b + 9] = q[3];
      P[b + 10] = S.col[i * 3]; P[b + 11] = S.col[i * 3 + 1]; P[b + 12] = S.col[i * 3 + 2];
      P[b + 13] = logit(Math.min(0.99, S.op[i]));
    }
    return P;
  }
  function triangleParams(S, scale, rnd) {
    const P = new Float32Array(S.n * 16);
    for (let i = 0; i < S.n; i++) {
      const th = rnd() * Math.PI * 2, b = i * 16;
      for (let k = 0; k < 3; k++) {
        const a = th + k * Math.PI * 2 / 3, f = 2.2 * scale;
        const ca = Math.cos(a) * f * S.sig[i * 3], sa = Math.sin(a) * f * S.sig[i * 3 + 1];
        for (let j = 0; j < 3; j++) P[b + k * 3 + j] = S.pos[i * 3 + j] + ca * S.ax[i * 9 + j] + sa * S.ax[i * 9 + 3 + j];
      }
      P[b + 9] = S.col[i * 3]; P[b + 10] = S.col[i * 3 + 1]; P[b + 11] = S.col[i * 3 + 2];
      P[b + 12] = logit(Math.min(0.95, S.op[i] * 0.9));
    }
    return P;
  }
  function convexParams(S, scale, rnd) {
    const P = new Float32Array(S.n * 24);
    for (let i = 0; i < S.n; i++) {
      const b = i * 24;
      for (let k = 0; k < 6; k++) {
        const a = (k + rnd() * 0.6) * Math.PI / 3, f = 2.0 * scale * (0.7 + 0.3 * rnd());
        const ca = Math.cos(a) * f * S.sig[i * 3], sa = Math.sin(a) * f * S.sig[i * 3 + 1], l = (rnd() - 0.5) * 0.1 * S.sig[i * 3];
        for (let j = 0; j < 3; j++) P[b + k * 3 + j] = S.pos[i * 3 + j] + ca * S.ax[i * 9 + j] + sa * S.ax[i * 9 + 3 + j] + l * S.ax[i * 9 + 6 + j];
      }
      P[b + 18] = S.col[i * 3]; P[b + 19] = S.col[i * 3 + 1]; P[b + 20] = S.col[i * 3 + 2];
      P[b + 21] = logit(Math.min(0.95, S.op[i] * 0.9));
    }
    return P;
  }

  /* ------------------------------------------------------------- trainer */
  const sig = x => 1 / (1 + Math.exp(-x));
  class Trainer {
    constructor(dev) { this.dev = dev; this.gt = null; }
    /* render the teacher (dense Gaussians) from numViews cameras into packed RGBA8 targets */
    async prepare({ teacherParams, nTeacher, center, radius, W, H, numViews, onProgress = () => {}, isStale = () => false }) {
      const dev = this.dev;
      this.W = W; this.H = H; this.radius = radius; this.center = center;
      this.views = makeViews(center, radius, numViews, W, H);
      if (this.gt) this.gt.destroy();
      this.gt = dev.createBuffer({ size: numViews * W * H * 4, usage: U.STORAGE | U.COPY_SRC });
      const T = new Raster(dev, 0, nTeacher, W, H, { keyCap: Math.max(1 << 20, nTeacher * 4) });
      T.setParams(teacherParams);
      for (let v = 0; v < numViews; v++) {
        if (isStale()) { T.destroy(); throw new Error("stale"); }
        const enc = dev.createCommandEncoder(); T.encodeProject(enc, this.views[v]); T.encodeRaster(enc); T.encodePack(enc, this.gt, v); dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
        onProgress((v + 1) / numViews);
      }
      this.preview = await T.readImage();
      const { overflows } = await T.readKeyInfo();
      T.destroy();
      return { dropped: overflows };
    }
    async readTeacher(v) {
      const st = this.dev.createBuffer({ size: this.W * this.H * 4, usage: U.MAP_READ | U.COPY_DST });
      const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.gt, v * this.W * this.H * 4, st, 0, this.W * this.H * 4); this.dev.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ); const out = new Uint8Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      return out;
    }
    /* optimize `params` (n primitives of `type`) against the teacher views */
    async train({ type, params, n, steps, sigmaTri = 1, deltaCvx = 6, sigmaCvx = 18, opaqueFinish = false, onProgress = () => {}, onSnapshot = () => {}, snapshotEvery = 300, isStale = () => false }) {
      const dev = this.dev, PS = PS_OF[type], ext = this.radius;
      const R = new Raster(dev, type, n, this.W, this.H, { train: true });
      R.sigmaTri = sigmaTri; R.deltaCvx = deltaCvx; R.sigmaCvx = sigmaCvx;
      R.setParams(params);
      this.raster = R;
      const lrBase = new Float32Array(24);
      let posIdx;
      if (type === 0) { posIdx = [0, 1, 2]; lrBase.set([0, 0, 0, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.03]); }
      else if (type === 1) { posIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8]; lrBase.set([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005, 0.005, 0.005, 0.03]); }
      else { posIdx = Array.from({ length: 18 }, (_, i) => i); lrBase.set([0.005, 0.005, 0.005, 0.03], 18); }
      const posStart = (type === 0 ? 3e-4 : 1.5e-3) * ext, posEnd = posStart * 0.08;
      const ab = new ArrayBuffer(128), af = new Float32Array(ab), au = new Uint32Array(ab);
      const relocEvery = 250, relocStop = Math.floor(steps * (opaqueFinish ? 0.55 : 0.7));
      const opaqueStart = Math.floor(steps * 0.65);
      let lossFrom = 0, t0 = performance.now();
      const inflight = [];
      for (let step = 1; step <= steps; step++) {
        if (isStale()) break;
        const prog = step / steps;
        const lrPos = posStart * Math.pow(posEnd / posStart, prog);
        af.set(lrBase, 0); for (const k of posIdx) af[k] = lrPos;
        let opFloor = -100;
        if (type === 1 && opaqueFinish && step >= opaqueStart) {
          const u = (step - opaqueStart) / Math.max(1, steps - opaqueStart);
          R.sigmaTri = sigmaTri + (0.08 - sigmaTri) * Math.min(1, u * 1.4);
          opFloor = logit(Math.min(0.995, 0.3 + 0.7 * Math.min(1, u * 1.6)));
          if (step === opaqueStart) await this.pruneTriangles(R, n, 0.3);
        }
        af[24] = 0.9; af[25] = 0.999; af[26] = 1e-12; af[27] = opFloor;
        au[28] = n; au[29] = step; au[30] = 0; au[31] = 0;
        const vi = (Math.random() * this.views.length) | 0;
        R.step(this.views[vi], this.gt, vi, af);
        // keep a few steps queued so the GPU never idles waiting for JS, without letting the queue grow unbounded
        inflight.push(dev.queue.onSubmittedWorkDone());
        if (inflight.length > 3) await inflight.shift();
        if (step % 100 === 0 || step === steps) {
          // the loss buffer sums per-step losses on the GPU; one readback per report
          const loss = await R.readLoss() / (step - lossFrom); lossFrom = step;
          onProgress({ step, steps, loss, ms: performance.now() - t0, sigmaTri: R.sigmaTri });
        }
        if (step % relocEvery === 0 && step <= relocStop) await this.relocate(R, type, n);
        if (step % snapshotEvery === 0 && step < steps) onSnapshot(await R.readBuffer(R.P, n * PS * 4), { step, sigmaTri: R.sigmaTri, final: false });
      }
      const P = await R.readBuffer(R.P, n * PS * 4);
      onSnapshot(P, { step: steps, sigmaTri: R.sigmaTri, final: true });
      R.destroy(); this.raster = null;
      return P;
    }
    /* MCMC-style densification: dead primitives are respawned as split copies of busy ones */
    async relocate(R, type, n) {
      const PS = PS_OF[type], P = await R.readBuffer(R.P, n * PS * 4), dens = await R.readBuffer(R.dens, n * 4);
      const opI = [13, 12, 21][type], dead = [], w = new Float64Array(n);
      let tot = 0;
      for (let i = 0; i < n; i++) {
        const op = sig(P[i * PS + opI]);
        if (op < 0.01) dead.push(i);
        else { tot += dens[i] * op + 1e-12; }
        w[i] = tot;
      }
      if (dead.length && tot > 0) {
        const pick = () => { const t = Math.random() * tot; let a = 0, b = n - 1; while (a < b) { const m = (a + b) >> 1; if (w[m] < t) a = m + 1; else b = m; } return a; };
        for (const d of dead) {
          const s = pick(), bs = s * PS, bd = d * PS;
          const op = sig(P[bs + opI]), opN = logit(1 - Math.sqrt(Math.max(1e-6, 1 - op)));
          for (let k = 0; k < PS; k++) P[bd + k] = P[bs + k];
          if (type === 0) {
            for (let k = 3; k < 6; k++) { P[bs + k] += Math.log(0.8); P[bd + k] += Math.log(0.8); }
            const sc = Math.exp(P[bs + 3]);
            for (let k = 0; k < 3; k++) P[bd + k] += (Math.random() - 0.5) * sc * 1.2;
          } else {
            const nv = type === 1 ? 3 : 6, c = [0, 0, 0];
            for (let v = 0; v < nv; v++) for (let k = 0; k < 3; k++) c[k] += P[bs + v * 3 + k] / nv;
            let rad = 0; for (let v = 0; v < nv; v++) rad = Math.max(rad, Math.hypot(P[bs + v * 3] - c[0], P[bs + v * 3 + 1] - c[1], P[bs + v * 3 + 2] - c[2]));
            const off = [0, 1, 2].map(() => (Math.random() - 0.5) * rad * 0.6);
            for (let v = 0; v < nv; v++) for (let k = 0; k < 3; k++) {
              const rel = P[bs + v * 3 + k] - c[k];
              P[bs + v * 3 + k] = c[k] + rel * 0.8;
              P[bd + v * 3 + k] = c[k] + off[k] + rel * 0.7;
            }
          }
          P[bs + opI] = opN; P[bd + opI] = opN;
        }
        this.dev.queue.writeBuffer(R.P, 0, P);
        const enc = this.dev.createCommandEncoder(); enc.clearBuffer(R.M1); enc.clearBuffer(R.M2); this.dev.queue.submit([enc.finish()]);
      }
      const enc = this.dev.createCommandEncoder(); enc.clearBuffer(R.dens); this.dev.queue.submit([enc.finish()]);
      this.lastRelocated = dead.length;
    }
    /* before forcing opacity up, collapse triangles that were nearly transparent so they vanish instead */
    async pruneTriangles(R, n, minOp) {
      const P = await R.readBuffer(R.P, n * 16 * 4);
      let pruned = 0;
      for (let i = 0; i < n; i++) {
        const b = i * 16;
        if (sig(P[b + 12]) < minOp) {
          const c = [0, 1, 2].map(k => (P[b + k] + P[b + 3 + k] + P[b + 6 + k]) / 3);
          for (let v = 0; v < 3; v++) for (let k = 0; k < 3; k++) P[b + v * 3 + k] = c[k];
          pruned++;
        }
      }
      this.dev.queue.writeBuffer(R.P, 0, P);
      this.pruned = pruned;
    }
    destroy() { if (this.gt) this.gt.destroy(); this.gt = null; }
  }

  return { createDevice, Raster, Trainer, makeViews, lookAtLH, gaussianParams, triangleParams, convexParams, PS_OF, logit, matToQuat };
})();
