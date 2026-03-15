# SindbadTutorials.jl

Tutorials and examples for the [SINDBAD](https://github.com/LandEcosystems/Sindbad.jl) hybrid modelling framework: learning global parameterizations and process representations of ecosystem carbon and water fluxes using mechanistic models combined with machine learning.

---

## Installation

### 1. Install Julia

Use [Juliaup](https://github.com/JuliaLang/juliaup) (recommended) or install Julia 1.11.x manually:

```bash
# Via Juliaup
curl -fsSL https://install.julialang.org | sh
juliaup add 1.11.9
juliaup default 1.11.9
```

Optional: install the [Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-server) for VS Code.

### 2. Clone the repository

```bash
git clone https://github.com/LandEcosystems/SindbadTutorials.jl.git
cd SindbadTutorials.jl
```

### 3. Activate and instantiate the project environment

There is a **single project** at the repo root (one `Project.toml`); tutorial directories do not have their own environments. You only need to activate and instantiate that root project once.

**From the repo root** (recommended): open a terminal in the repo root, start Julia, then:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

In VS Code, open the repo as the workspace folder and set the active environment via the **Julia env:** dropdown to this project (root); the REPL and notebooks will then use it.

**If you run a notebook from inside a tutorial folder** (e.g. `tutorials/hybrid_parameter_learning/`): the notebook’s current directory may be that subfolder. Activate the repo root from there so the same project is used, e.g. in the first cell:

```julia
using Pkg
Pkg.activate("../../")   # from tutorials/hybrid_parameter_learning/ → repo root
Pkg.instantiate()
```

(Some notebooks already include this; use one or two `"../"` depending on depth.) You do **not** run `Pkg.activate(".")` with the current directory inside a tutorial folder, or you would activate a folder that has no `Project.toml`.

### 4. (Optional) Configure how Sindbad is loaded: `setup_sindbad.jl`

The script `setup_sindbad.jl` configures whether the Sindbad ecosystem (Sindbad.jl, ErrorMetrics, TimeSamplers, OmniTools) is used from the **registry** or from **local dev checkouts**. This is optional if you only want to run tutorials with released packages.

- **Run mode (default)**  
  Use released versions from the registry. No local clones. Good for just running tutorials.

- **Dev mode**  
  Clone packages into `dev/` and use `Pkg.develop()` so code changes in `dev/Sindbad.jl` (and the others) are used by the tutorials. Good for developing or debugging Sindbad while working through the repo.

**Configure mode** in `SindbadSetup.toml` (create it if missing):

```toml
[mode]
sindbad = "run"   # use registry (default)
# sindbad = "dev" # use local clones under dev/
```

Optional: override Git URLs for your forks:

```toml
[Sindbad]
# git_url = "https://github.com/your-user/Sindbad.jl.git"
```

**Apply the setup** (from the repo root):

```bash
julia setup_sindbad.jl
```

The script will:

- **Run mode:** Remove any dev checkouts and install Sindbad (and its dependencies) from the registry.
- **Dev mode:** Clone the packages into `dev/` (if not already present), run `Pkg.develop(path="dev/...")` for each, and set up Sindbad.jl’s environment to use the same local packages. You can then edit code in `dev/Sindbad.jl` and use `Revise` / re-running cells to pick up changes.

Afterwards, run `Pkg.instantiate()` (or let the script do it) so the environment is resolved. If you use dev mode, start Julia with `using Revise` then `using SindbadTutorials` so edits in `dev/` are reflected when you re-run.

---

## Tutorials overview

Tutorials live under `tutorials/` and are organized by topic. Configurations (experiments, optimization, model structure) are in `tutorials/setups/`.

| Tutorial | Description |
|----------|-------------|
| **hybrid_parameter_learning** | Learn WROASTED/LUE model parameters from synthetic FLUXNET data with ML. Covers running the baseline, changing covariates (`Step02_load_covariates.jl`), and using PCA (`Step03_use_pca.jl`). Entry: `Task_Parameter.md` and `run_hybrid_inversion.jl`. |
| **hybrid_process_learning** | Build a hybrid model for GPP sensitivity to air temperature: add a new model/approach in Sindbad, implement an external ML model, and link it (direct call or lookup table). Entry: `Task_Process.md`, `Step01_add_model.jl`, `Step02_implement_ml_model.jl`, `Step03_link_ml_model.jl` / `Step03_link_ml_model_lut.jl`. |
| **insitu_inversion** | Site-level inversion (e.g. single FLUXNET site) with CMA-ES optimization. Run `run_insitu_inversion.jl` or the matching notebook; experiment is defined in `setups/WROASTED_HB/experiment_insitu.json`. |
| **global_inversion** | Global-scale parameter inversion with optional PFT filtering and CMA-ES. Run `run_global_inversion.jl`; uses experiment and optimization configs from `setups/`. |
| **setups/** | Shared JSON configs for experiments, forcing, optimization, and model structure (e.g. `WROASTED_HB`, `LUE`, `LUE_NN`). Referenced by the scripts above. |

Follow the `.md` task descriptions and run the `.jl` scripts or `.ipynb` notebooks as indicated in each tutorial folder.

---

## License and citation

See the repository license. When using these tutorials or SINDBAD in research, please cite the relevant papers and the [Sindbad](https://github.com/LandEcosystems/Sindbad.jl) package.
