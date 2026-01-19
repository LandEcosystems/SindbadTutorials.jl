# SINDBAD-Tutorials
A central repository to develop SINDBAD tutorials

## Getting the repo

The following command will install the tutorial files and SINDBAD itself.

```bash
git clone https://github.com/LandEcosystems/SindbadTutorials.jl
```

Note the root directory of where the repo is, for convenience, we'll call it `repo_root` from now on.

The tutorial files are stored in the `tutorials` subdirectory, and organised by topic or summer school or other event. The current tutorials are:
- ai4pex_2025
- ellis_jena_2025

From now on, we'll refer to the current `tutorial` directory as `which_tutorial`.


## Install Julia

Use Juliaup to install `Julia`. See instructions here:

https://github.com/JuliaLang/juliaup

The, install the VS Code Julia extension: 

https://marketplace.visualstudio.com/items?itemName=julialang.language-server


# Install Tutorial Environment
Open a terminal at the root of this repo (`repo_root`)

Go to the `which_tutorial` tutorial folder:
```bash
cd tutorials/which_tutorial
```

Start up `Julia`, e.g., in Terminal:

```bash
julia
```

or in VSCode, open the `tutorials/which_tutorial` folder and start the REPL (Ctrl+Shift+J)

Activate an environment in the folder with:
```julia
using Pkg
Pkg.activate("./")
```

The prompt should change to `(which_tutorial) pkg>`.

Instantiate/install the packages in the environment with:
```julia
Pkg.instantiate()
```

# Set REPL environment
In VS code, set the `which_tutorial` as the active project by clicking on the `Julia env:` dropdown and selecting `which_tutorial` as the folder. This should change the default environment for the REPL.


# Download Tutorial Data
Download the data for the tutorial from [here](https://nextcloud.bgc-jena.mpg.de/s/Byj9FKr2mr7QYgZ). 
Unzip the file into `tutorials/data/which_tutorial`. *You may need to create the directory.*

# Tutorials

The tutorials are located under `tutorials/which_tutorial`. Follow the instructions in the `.jl` scripts or `.ipynb` notebooks.