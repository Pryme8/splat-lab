# Splat Lab

Train and compare four radiance-field primitives in the browser, side by side, with [Babylon.js](https://www.babylonjs.com/) and WebGPU:

| Tab | Technique | Paper | How it's drawn here |
|---|---|---|---|
| **3DGS** | 3D Gaussian splatting | Kerbl et al., SIGGRAPH 2023 ([project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)) | Babylon's built-in `GaussianSplattingMesh`, sorted on a worker. The baseline. |
| **Points** | Gaussian point splatting | Rijsdijk et al., ACM TOG 2026 ([doi](https://doi.org/10.1145/3811272)) | Babylon 9's built-in WebGPU compute implementation (`pointSplattingRenderMode`). WebGL2 fallback included. |
| **Triangle** | Triangle splatting / Triangle Splatting+ | Held et al. 2025 ([arXiv](https://arxiv.org/abs/2505.19175), [+](https://arxiv.org/abs/2509.25122)) | Custom shaders (WGSL + GLSL), screen-space window identical to the trainer's. Opaque results draw as a plain mesh. |
| **Convex** | 3D convex splatting | Held et al., CVPR 2025 ([arXiv](https://arxiv.org/abs/2411.14974)) | Custom shaders; 2D hull built in the vertex shader. |

**Live demo:** `https://<your-user>.github.io/splat-lab/` (after enabling GitHub Pages, see below). Needs a WebGPU browser for training; rendering also works on WebGL2.

## What it does

1. **Load a source.** Meshes (`.glb .gltf .obj .stl`), trained Gaussian splats (`.ply .splat .spz`), point clouds (`.ply`), or a sculpt written in a small SDF language with primitives, smooth booleans, domain repetition, twist, noise and per-shape color. Try the files in [`examples/`](examples/).
2. **Convert.** Every source becomes the same set of oriented elements (position, three axes, three radii, color, opacity). Untrained, each renderer simply converts those elements into its own primitive. This looks rough, on purpose: it is the starting point the papers optimize from, not their result.
3. **Train.** `splat-train.js` renders 32–96 views of a dense "teacher" version of the source, then optimizes the current tab's primitives against them with Adam:
   - tile-based differentiable rasterizer in WGSL compute (project → prefix sum → key duplication → 4-bit LSD radix sort → per-tile front-to-back compositing);
   - analytic backward pass with float atomics, chained to 3D parameters (Gaussian mean/scale/rotation, triangle corners, all six convex points, colors, opacities);
   - L1 + L2 loss, MCMC-style relocation of dead primitives;
   - optional *opaque finish* for triangles (anneal σ → 0 and opacity → 1, the Triangle Splatting+ recipe), after which triangles render as an ordinary depth-tested mesh with no sorting or blending.

## Run locally

Any static server works. WebGPU requires a secure context, which `localhost` counts as.

```sh
git clone https://github.com/<your-user>/splat-lab.git
cd splat-lab
python3 -m http.server 8000      # or: npx serve .
# open http://localhost:8000/
```

Babylon.js 9.32.0 and its loaders load from jsDelivr; there is no build step.

## Tests

`tests/gradcheck.html` (open it through the same server) runs:

- **Gradient check:** analytic gradients vs. central finite differences for all three primitive types. Last run: Gaussians 69/72, triangles 64/66, convexes 73/84 within 10%. Remaining misses come from discontinuities (tile edges, opacity cutoffs, points crossing the convex hull).
- **Forward render:** teacher views of `examples/helix.splat`.
- **Train test:** 2,000 Gaussians and 2,000 triangles for 400 steps. Loss drops from ~0.10 to ~0.012 (Gaussians) and ~0.025 (triangles, opaque).

## Status and known limits

This is a research playground, not a production renderer. Open questions and good first issues:

- **Unverified at full scale.** Development happened on a software GPU, so full-size runs (100k+ primitives, 3,000 steps, 256²) have not been benchmarked on real hardware yet. Numbers and screenshots from real GPUs are very welcome.
- **One GPU→CPU readback per training step** (to size the sort). Removing it with indirect dispatch would speed training up.
- **View-independent color** (no spherical harmonics) and a plain L1/L2 loss (no D-SSIM).
- **Memory caps** on training: 800k Gaussians, 400k triangles, 250k convexes.
- **Convex training** treats hull membership as fixed per step (as the paper does); gradients are approximate where points cross the hull.
- The GaussianSplattingMesh point mode is Babylon's code; issues there belong upstream.

## Repository layout

```
index.html          the app (UI, sources, SDF language, renderers, training UI)
splat-train.js      standalone WebGPU trainer / differentiable rasterizer
examples/           small test assets (mesh, textured glTF, splats, point cloud)
tests/              gradient check, forward render and train tests
tools/              script that regenerates examples/
AGENTS.md           handoff notes for coding agents and contributors
```

## Publishing with GitHub Pages

Repository **Settings → Pages → Build and deployment → Deploy from a branch → `main` / root**. The site appears at `https://<your-user>.github.io/splat-lab/` within a minute or two.

## License

MIT, see [LICENSE](LICENSE). The papers cited above belong to their authors; this repository is an independent implementation for learning and comparison.
