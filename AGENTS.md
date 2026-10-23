# Agent handoff: Splat Lab

Read this before changing anything. It records how the project is put together, why some things are the way they are, and what is left to do. `README.md` is the user-facing description; this file is for whoever (human or agent) works on the code next.

## What this is

A static, build-free web app that converts 3D sources into four kinds of radiance-field primitives, renders them with Babylon.js, and trains them in the browser with a custom WebGPU differentiable rasterizer.

- **Sources:** SDF sculpt language, meshes (glb/gltf/obj/stl), trained Gaussian splats (ply/splat/spz), point clouds (ply).
- **Renderers (tabs):** `gs3d` (Babylon GaussianSplattingMesh, sorted), `points` (Babylon 9's native WebGPU point splatting; own WebGL2 fallback), `triangle`, `convex`.
- **Training:** `splat-train.js`, optimizing Gaussians (type 0), triangles (type 1) or convexes (type 2) against teacher views.

Origin: built in one long session with Claude; the owner wants it public on GitHub for discussion. Code quality bar: correct and readable over clever. Keep it build-free.

## Files

| File | Contents |
|---|---|
| `index.html` | Everything except training: CSS, UI, sources, SDF language (`sdfKernel`), samplers, renderers, app state, training UI. One `<script>` block, organized by `/* ===== */` section banners. |
| `splat-train.js` | `SplatTrain` IIFE: WGSL kernels as template strings, `Raster` (forward + backward), `Trainer` (teacher views, Adam loop, relocation), param conversion helpers. No Babylon dependency; creates its own `GPUDevice`. |
| `tests/gradcheck.html` | Gradient check, forward render, small training run. Open via a local server. |
| `tests/bench.html` | Training benchmark at the app's default sizes against a procedural torus-knot teacher: steps/s, loss curve, trained renders. Query parameters override the sizes. |
| `examples/` | Test assets. `tools/make-examples.py` regenerates them. |

Run: `python3 -m http.server 8000` in the repo root, open `http://localhost:8000/`. Add `?webgl` to force the WebGL2 engine. WebGPU needs a secure context (localhost or https).

## Data model (the contract between everything)

Every source becomes a "splat set" (`makeSet` in index.html):

```
n, kind ("surface" | "gaussian"),
pos[3n], ax[9n] (three orthonormal axes per element; a3 is the thinnest = normal),
sig[3n] (Gaussian std-devs along the axes, sig1 >= sig2 >= sig3),
col[3n] (display RGB 0..1), op[n] (opacity 0..1)
```

Renderers' `build(S, scale, reveal)` consume this. Trained results use `buildExplicit(...)` (triangles: `V[9n]`, convex: `pts[18n]`) or `gaussParamsToSet`.

### Trainer parameter layouts (`splat-train.js`, `PS_OF = [16, 16, 24]` floats per primitive)

- **type 0 Gaussian:** `[mx,my,mz, logσ1,logσ2,logσ3, qw,qx,qy,qz, r,g,b, opacityLogit, pad, pad]`
- **type 1 triangle:** `[v0xyz, v1xyz, v2xyz, r,g,b, opacityLogit, pad×3]`
- **type 2 convex:** `[p0..p5 xyz (18), r,g,b, opacityLogit, pad, pad]`

Projected records (`JS_OF = [16, 24, 28]`) and per-primitive 2D gradient slots (`GS = 20`) are documented next to the WGSL that writes them (`projectBodies`, `backPixel`). If you change a layout, change the projector, the window function, the backward pixel code, the backprojection and the Adam opacity index (`OPP`) together.

## Conventions that bite

- **Camera:** Babylon left-handed, view space x right / y up / z forward. Trainer views use `lookAtLH` with the same column-major layout as Babylon's `Matrix.m`. Trainer pixel coordinates are **y-down** (`py = cy - fy*y/z`); display shaders work **y-up**. Edge distances are orientation-invariant because every window recomputes the winding sign `s`. Keep it that way.
- **WGSL on Babylon's WebGPU engine:** custom ShaderMaterials must be WGSL (`ShaderStore.ShadersStoreWGSL`, `shaderLanguage: WGSL`). Creating *any* GLSL material on WebGPU makes Babylon fetch glslang/twgsl from cdn.babylonjs.com, which breaks under strict CSPs. `shaderMat()` picks the language; GLSL twins exist for the WebGL2 path.
  - Babylon multiplies clip-space y by `internals.yFactor_` (−1 on the canvas), so in WGSL fragments `fragmentInputs.position.xy` is already bottom-up like `gl_FragCoord`. Do **not** flip it again (that bug collapsed triangles to a line).
  - Babylon wraps WGSL `main`; don't `return` early from a vertex `main`. Use flags (see the convex WGSL vertex shader).
- **Display window must equal training window.** The triangle and convex display shaders recompute the window in screen space from all corners (each vertex carries all three triangle corners; convex hulls are built in the vertex shader). If you change one, change the other, or trained results will look wrong.
- **Convex size normalization:** distances are divided by `Sz = sqrt(hull area)`. Its gradient is included (`gSz` → backprojection through the hull area). Hull membership changes are not differentiable; that's expected.
- **Opacity clamp:** alpha = `min(0.99, op * window)`; backward zeroes the gradient when clamped. Alpha < 1/255 is skipped in both passes; transmittance stops at 1e-4. Forward and backward must agree on these.
- **Buffers owned by a `Raster`** are destroyed in `destroy()`, except borrowed ones (`_lossGT`, `_packGT`). Don't store other borrowed GPUBuffers on a Raster without adding them to that skip list (this exact bug once wiped the teacher images).
- **Radix sort keys:** `(tile << depthBits) | quantizedDepth`; `depthBits = 32 - ceil(log2(numTiles + 1))` so the padding tile id never collides with a real one. Sort is 8 passes of 4 bits, stable via per-digit workgroup scans.
- **No per-step readback.** The key count never reaches the CPU. The `args` kernel writes `info` (K capped, K raw, block count, overflow count) and `ind` (indirect dispatch args); radix sort and ranges use `dispatchWorkgroupsIndirect`. Digit counts are stored with a fixed stride (`nbMax`, blocks for the whole key cap) so the count scan has a fixed size; `cnt` is cleared before every pass so blocks past the live count stay zero. `Trainer.train` keeps up to 4 steps queued. Anything that maps a buffer (`readLoss`, `readBuffer`, relocation) drains that queue, so keep it off the per-step path.
- **SDF workers:** `sdfKernel` is serialized with `Function.prototype.toString()` into a Blob worker. It must stay self-contained: no references to outer-scope helpers.
- **Artifact/CSP constraints** (the app was first hosted in a sandbox): scripts only from jsDelivr/cdnjs/unpkg, no fetch to other hosts, no `window.print`, no `alert`. GitHub Pages doesn't need these, but keeping them makes the app embeddable anywhere.

## Testing

- `tests/gradcheck.html` → **Gradient check**. Expected agreement within 10%: Gaussians ~69/72, triangles ~64/66, convexes ~73/84. A drop means a backward-pass bug. Test primitives are spaced in depth on purpose: finite differences across a sort-order flip are meaningless.
- **Train test** → loss should fall from ~0.10 to ~0.012 (Gaussians) in 400 steps.
- `tests/bench.html` → **Run benchmark**, or `run('bench', { types: '1', steps: 500 })` from the console. A full default run takes about 3 minutes. RTX 5080, Chrome 152, Windows, defaults: Gaussians 54 steps/s, triangles 131, convexes 33 (35.6 / 52.8 / 25.1 before the readback was removed).
- **Timing GPU work:** `onSubmittedWorkDone()` and `mapAsync` resolve on Chrome's polling ticks (2.5–12 ms observed), so wall-clock timing of single phases measures the polling, not the GPU. Use timestamp queries (`timestamp-query` feature, `timestampWrites` per pass); Chrome quantizes them coarsely (~0.07 ms here) but sums over a step are reliable.
- Manual: load each file in `examples/`, switch all four tabs on both `?webgl` and WebGPU, train each type briefly, toggle "Show untrained", toggle the opaque mesh.
- Headless Chromium can run WebGPU **compute** but its WebGPU **canvas** loses the device; for canvas tests run headful under Xvfb and read pixels with `engine.readPixels` (canvas is BGRA on WebGPU).

## Open work, roughly by value

1. **Faster backward pass.** It dominates GPU time. Per step at the defaults on an RTX 5080 (timestamp queries), backward takes 14.0 / 4.9 / 18.2 ms of 15.7 / 6.1 / 21.6 ms total for Gaussians / triangles / convexes; render takes 0.9 / 0.8 / 3.0 ms; project, sort, loss, backprojection and Adam together stay under 1 ms. Every pixel CAS-adds into the same primitive's gradient slots, so the 256 pixels of a tile contend on the same addresses. Reduce within the tile first (for example `subgroupAdd` where the `subgroups` feature exists, keeping the current path as fallback), then issue one atomic per subgroup.
2. **More GPUs, learning rates, screenshots.** `tests/bench.html` exists and has numbers from one GPU (see Testing). Still to do: runs on integrated / Apple / AMD GPUs; tune learning rates in `Trainer.train` (`lrBase`, `posStart`) by looking at trained results in the app (convex loss is still falling at 3,000 steps); add app screenshots to the README.
3. **D-SSIM term** in the loss (3DGS uses 0.8 L1 + 0.2 D-SSIM). Needs a separable Gaussian blur pass over the 256² image; gradients are standard.
4. **Spherical harmonics** for view-dependent color (at least degree 1). Touches the param layout, projection, backward and the display shaders.
5. **Per-primitive σ (triangles) and δ/σ (convexes)** as learnable parameters, as in the papers. Currently global uniforms.
6. **Better densification.** Current relocation (`Trainer.relocate`) is a simple MCMC-style respawn every 250 steps. The papers' split/clone heuristics would help, especially for triangles.
7. **Export trained results:** `.ply`/`.splat` for Gaussians, `.glb` for opaque triangles (vertex colors). Babylon's serializers or a hand-written GLB writer both work.
8. **Teacher from real renders for meshes** (rasterize the original mesh with its textures instead of dense surfels) for sharper supervision.

## Things not to "fix"

- The untrained conversions look rough on purpose. They are the papers' starting point; training is the point of the app. Don't tune untrained visuals at the expense of matching the trainer's initialization.
- `RefSplats` uses Babylon's own GaussianSplattingMesh and point mode. Bugs there belong upstream in Babylon.js; don't reimplement them here.
