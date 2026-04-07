# BTP Presentation Content
## SRAM-Fused Triton Kernel for Monte-Carlo Inventory Optimisation

**Target Duration: 7–8 minutes | 10 Slides**

---

## SLIDE 1 — Title Slide
**⏱ ~20 seconds**

---

### Title
**SRAM-Fused Triton Kernel for Monte-Carlo Inventory Optimisation:**
*A Multi-Echelon Stochastic Newsvendor Approach*

**Subtitle:**
Bachelor Thesis Project | Indian Institute of Technology Kharagpur

**Bottom row:**
[Your Name] | [Roll Number] | [Supervisor Name] | [Date]

---

### Suggested Visual
Clean IIT KGP branded slide with a subtle GPU chip or supply chain network graphic in the background.

---

---
---

## SLIDE 2 — The Business Problem
**⏱ ~45 seconds**

---

### Title
**Inventory Decisions Under Uncertainty: The Newsvendor Problem**

---

### Bullet Points

- A dealership network sells **tractors** and **generators** across multiple store locations
- Each season, a manager must decide **how much to order** *before* demand is observed
- **Order too little** → lost sales, unmet customers
- **Order too much** → unsold stock salvaged at a deep loss
- This classic trade-off is the **Newsvendor Problem** [Arrow et al., 1951]

**The challenge scales up:**
- **2,048 product-location nodes** across the network
- Demand at nearby stores is **spatially correlated** [Clark & Scarf, 1960]
- **131,072 Monte-Carlo scenarios** needed to faithfully model uncertainty

---

### Suggested Visual
*Left:* Simple diagram — "Order Q → Demand D arrives → Profit / Loss"
*Right:* Small network map of store locations with colour-coded correlation strength

---

---
---

## SLIDE 3 — The Mathematical Framework
**⏱ ~40 seconds**

---

### Title
**Formalising the Problem: Correlated Monte-Carlo Simulation**

---

### Bullet Points

**Expected profit per node i:**

$$E[\pi_i] = \frac{1}{S} \sum_{s=1}^{S} \Bigl[ p_i \cdot \min(D_{i,s},\, Q_i) \;-\; c_i \cdot Q_i \;+\; s_i \cdot \max(Q_i - D_{i,s},\, 0) \Bigr]$$

**Correlated demand via Cholesky decomposition [Glasserman, 2003]:**

$$D_{i,s} = \max\!\Bigl(\mu_i + \textstyle\sum_k L_{ik}\, Z_{ks},\; 0\Bigr) \qquad \Sigma = L L^\top$$

**Five-step computation graph:**

`Σ` → `Cholesky L` → `D = μ + L×Z` → `min(D, Q)` → `π` → `E[π]`

**The bottleneck:**
- L×Z produces intermediate matrix D of shape **[2048 × 131,072] = 1.07 GB** per forward pass

---

### Suggested Visual
> **[INSERT: `diagrams/newsvendor_math_flow.png`]**

---

---
---

## SLIDE 4 — The Bottleneck & Our Solution
**⏱ ~60 seconds**

---

### Title
**The Memory Wall — and How We Break It**

---

### Bullet Points

**Why `torch.compile` cannot solve this [Ansel et al., 2024]:**

```
D = L @ Z          ← cuBLAS (black box) writes 1.07 GB to HBM ✗
D = clamp(D, 0)    ← re-reads 1.07 GB from HBM
X = min(D, Q)      ← re-reads 1.07 GB from HBM
profit = p·X − c·Q + s·overage
E[π] = mean(profit)
```
- `torch.compile` fuses element-wise ops — but **cannot fuse across the matmul boundary**
- D[N, S] must be **fully materialised in HBM** between the matmul and everything that follows
- The 1.07 GB round-trip is the **dominant cost** [Wulf & McKee, 1995; Ivanov et al., 2021]

---

**Our solution — inspired by FlashAttention [Dao et al., 2022]:**

| | PyTorch-Compile | **Triton-Fused (Ours)** |
|---|---|---|
| D matrix in HBM | **1.07 GB** ✗ | **0 B** ✓ |
| Computation location | HBM → registers → HBM | Entirely in SRAM |
| HBM writes | ~1.07 GB | **8 KB** (output only) |

➡ **Tile the matmul. Apply newsvendor logic inside each tile. Write only the result.**

---

### Suggested Visual
> **[INSERT: `diagrams/memory_hierarchy.png`]**
Left panel = PyTorch path (large red D-matrix arrow through HBM). Right panel = Triton path (small green SRAM box, no D in HBM).

---

---
---

## SLIDE 5 — Data Pipeline
**⏱ ~35 seconds**

---

### Title
**Realistic Data Pipeline: From M5 Dataset to GPU Tensors**

---

### Bullet Points

**Four-stage ETL pipeline:**

| Stage | What it does | Output |
|---|---|---|
| 1. Topology | Hierarchical spatial correlations from M5 Kaggle dataset [Makridakis et al., 2022] | Correlation matrix R [N×N] |
| 2. Cholesky | L = chol(Σ) in FP64 for stability [Higham, 2002] | Factor L [N×N] = 16 MB |
| 3. Demand & Financials | Samples (μ, σ, p, c, s) per node by product category | Vectors [N×1] each |
| 4. Assembly | Z ~ N(0,I) generated on GPU; immutable TensorBundle packed | Z [N×S] = 1.07 GB on CUDA |

**Two product categories:**
- **Tractors (60%, N=1,228):** High-value (₹4.5–9 L), seasonal demand, very low salvage value
- **Generators (40%, N=820):** Moderate value, base sales + spare-parts failure demand (Poisson)

---

### Suggested Visual
> **[INSERT: `diagrams/data_pipeline_flow.png`]**

---

---
---

## SLIDE 6 — The Triton Kernel: Four Phases
**⏱ ~75 seconds**
*(core technical slide — spend the most time here)*

---

### Title
**The SRAM-Fused Triton Kernel: Four Phases**

---

### Bullet Points

**2-D kernel grid — each program instance handles one [BM × BN] tile:**

$$\text{Grid} = \Bigl(\bigl\lceil N/\text{BM}\bigr\rceil,\; \bigl\lceil S/\text{BN}\bigr\rceil\Bigr)$$

---

**Phase 1 — Tiled Matmul (K-loop, entirely in SRAM)**
```
for k in 0 .. ⌈K/BK⌉:
    L_tile [BM, BK]  ← load from HBM        # read-only
    Z_tile [BK, BN]  ← load from HBM        # read-only
    acc    [BM, BN]  += dot(L_tile, Z_tile)  # stays in SRAM
```

**Phase 2 — Fused Newsvendor Logic (zero HBM traffic)**
```
D       = max(μ + acc, 0)           # correlated demand — in SRAM
X       = min(D, Q)                 # constrained sales
overage = max(Q − D, 0)             # unsold inventory
profit  = p·X − c·Q + s·overage    # per-scenario profit
```

**Phase 3 — Partial Mean Reduction**
```
partial_mean = sum(profit, axis=1) / S    # [BM] — in registers
```

**Phase 4 — Atomic Accumulation (only HBM write: 8 KB total)**
```
atomic_add(out[offs_m], partial_mean)
```

---

### Suggested Visual
> **[INSERT: `diagrams/triton_kernel_grid.png`]**
Two-column layout: left = phases pseudocode (above), right = kernel grid diagram.
*Animate phases 1 → 2 → 3 → 4 one click at a time.*

---

---
---

## SLIDE 7 — Autotuning & Portability
**⏱ ~30 seconds**

---

### Title
**Autotuning: One Kernel, Any NVIDIA GPU**

---

### Bullet Points

**Challenge:** Optimal tile dimensions vary by GPU architecture

| GPU | SRAM / SM | Tile Budget |
|---|---|---|
| T4 (Turing) | 64 KB | Conservative |
| A100 (Ampere) | 164 KB | Larger tiles |
| H100 (Hopper) | 228 KB | Maximum |

**Solution:** `@triton.autotune` profiles **9 hand-curated configs** at first launch [Tillet et al., 2019]
- All 9 fit within **48 KB** — safe floor for T4 and above
- Best config on T4: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 → **48 KB SRAM**
- Winner is **cached** — zero overhead on subsequent kernel calls
- **Re-tunes automatically** only when N, S, or K changes

---

### Suggested Visual
Small two-column layout:
- Left: 9-config table (BLOCK_M, BLOCK_N, BLOCK_K, SRAM) with winning row highlighted
- Right: GPU architecture comparison (T4 / A100 / H100) with SRAM capacity bars

---

---
---

## SLIDE 8 — Results: Performance
**⏱ ~50 seconds**

---

### Title
**Results: 2.5× Faster, 34% Less Memory, 2.6× Higher Throughput**

---

### Bullet Points

**Benchmark setup:** N = 2,048 nodes | S = 131,072 scenarios | NVIDIA T4 GPU (16 GB)

---

| Metric | CPU-NumPy | PyTorch-Compile | **Triton-Fused** |
|---|---|---|---|
| Time (ms) | ~2,500 | ~180 | **~70** |
| Speedup vs CPU | 1× | ~14× | **~36×** |
| Peak GPU Mem | N/A | 3.2 GB | **2.1 GB** |
| TFLOPS | ~0.4 | ~6.1 | **~15.7** |

**Key takeaways:**
- Triton is **2.5× faster than PyTorch-Compile** — same arithmetic, eliminated memory traffic
- **34% memory reduction** — the 1.07 GB D matrix is entirely gone
- **15.7 TFLOPS** — 2.6× higher throughput than PyTorch-Compile

---

### Suggested Visual
> **[INSERT: Notebook plots — three side-by-side bar charts: Time (ms) | Memory (GB) | TFLOPS]**
Triton bar highlighted in a distinct colour (e.g., IIT KGP red/gold).

---

---
---

## SLIDE 9 — Correctness, Scaling & System
**⏱ ~45 seconds**

---

### Title
**Validation, Scaling Behaviour & System Overview**

---

### Bullet Points

**Numerical correctness — all three solvers agree within FP32 tolerance:**

| Comparison | Max \|Δ\| | Status |
|---|---|---|
| CPU vs PyTorch-Compile | < 0.00002 | ✅ PASS |
| CPU vs Triton-Fused | < 0.00002 | ✅ PASS |
| PyTorch vs Triton-Fused | < 0.00001 | ✅ PASS |

---

**Speedup grows with N** (S = 65,536 fixed):

| N | PyTorch (ms) | Triton (ms) | Speedup |
|---|---|---|---|
| 128 | ~5 | ~3 | 1.7× |
| 512 | ~31 | ~16 | 2.0× |
| **2048** | **~202** | **~75** | **2.7×** |

➡ As N↑, PyTorch's D matrix cost grows as **N×S**; Triton's HBM traffic grows as **N** only.

---

**System architecture — five clean modules:**
`config` → `data_pipeline` → `[CPU / PyTorch / Triton solvers]` → `benchmark`
Reproduced end-to-end in `BTP_Newsvendor_Full_Run.ipynb` (one click, Google Colab T4)

---

### Suggested Visual
> **[INSERT: Notebook plots — parity scatter (left) + scaling sweep line plot (right)]**
*Optionally: `diagrams/system_architecture.png` as a small inset at the bottom.*

---

---
---

## SLIDE 10 — Conclusions & Future Work
**⏱ ~40 seconds**

---

### Title
**Conclusions & Future Directions**

---

### Bullet Points

**What we demonstrated:**
- ✅ **2.5× speedup** over `torch.compile` PyTorch | **~36× over CPU** on NVIDIA T4
- ✅ **34% peak memory reduction** — 1.07 GB intermediate tensor eliminated
- ✅ **15.7 TFLOPS** — 2.6× higher throughput than PyTorch-Compile
- ✅ Kernel fusion is a **general, reusable strategy**: matmul → element-wise → reduce appears in attention, kernel density estimation, and many simulation workloads
- ✅ **Portable via autotuning** — works on T4, A100, H100 without code changes

**Future directions:**
1. **Differentiable simulation** — optimise order quantities Q via gradient descent end-to-end [Paszke et al., 2019]
2. **Multi-period model** — extend to carry-over inventory and lead times [Zipkin, 2000]
3. **FP16/BF16 tensor cores** on Ampere/Hopper for further speedup [Micikevicius et al., 2018]
4. **Multi-GPU scaling** — distribute L across devices for N > 10,000

---

### Suggested Visual
Two-column layout:
- **Left:** Three large bold numbers — `2.5×` / `34%` / `15.7 TFLOPS` — with brief labels
- **Right:** Bulleted future-work list

---

---
---

## SLIDE 11 — Q&A / Backup
**⏱ As needed**

---

### Title
**Thank You — Questions Welcome**

---

### Content

**Repository / Notebook:**
`BTP_Newsvendor_Full_Run.ipynb` — fully reproducible on Google Colab (T4 GPU)

**Key references:**
- Triton: Tillet et al., MAPL 2019
- FlashAttention: Dao et al., NeurIPS 2022
- Newsvendor: Arrow et al., Econometrica 1951
- torch.compile: Ansel et al., ASPLOS 2024

---

### Backup Slides (prepare but reveal only if asked)

| # | Topic | When to use |
|---|---|---|
| B1 | Full 9-config autotune table with SRAM breakdown | If asked about hardware portability |
| B2 | Cholesky factor L heatmap [2048×2048] | If asked about correlation structure |
| B3 | Profit distribution histograms (tractors vs generators) | If asked about business insights |
| B4 | Detailed kernel pseudocode walkthrough | If asked by GPU/HPC audience |
| B5 | Financial parameter ranges table | If asked about domain realism |

---
---

## PRESENTATION TIMING GUIDE

| # | Slide | Cumulative |
|---|---|---|
| 1 | Title | 0:20 |
| 2 | Business Problem | 1:05 |
| 3 | Mathematical Framework | 1:45 |
| 4 | Bottleneck & Solution *(merged)* | 2:45 |
| 5 | Data Pipeline | 3:20 |
| 6 | Triton Kernel: Four Phases *(spend time here)* | 4:35 |
| 7 | Autotuning & Portability | 5:05 |
| 8 | Results: Performance | 5:55 |
| 9 | Correctness, Scaling & System *(merged)* | 6:40 |
| 10 | Conclusions & Future Work | 7:20 |

**Total: ~7 min 20 sec** — comfortable margin before the 8-minute mark.

---

## DESIGN RECOMMENDATIONS

**Slide style:**
- Dark background (deep navy or charcoal) with white/light text
- IIT KGP colour accent (red or gold) for slide titles and key numbers
- Monospace font (Consolas / JetBrains Mono) for all code blocks
- Maximum **5–6 bullet points** per slide — never more

**Figures to embed:**

| Slide | Figure file |
|---|---|
| 3 | `diagrams/newsvendor_math_flow.png` |
| 4 | `diagrams/memory_hierarchy.png` |
| 5 | `diagrams/data_pipeline_flow.png` |
| 6 | `diagrams/triton_kernel_grid.png` (right column) |
| 8 | Notebook: Time / Memory / TFLOPS bar charts |
| 9 | Notebook: parity scatter + scaling sweep |

**Animation tip (Slide 6):**
Animate the four kernel phases one click at a time (Phase 1 → 2 → 3 → 4) so the audience follows along with your explanation rather than reading ahead.
