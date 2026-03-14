# Challenge 13: Learning global parameterizations of ecosystem processes using hybrid modelling

Author: Xu Shan, Sujan Koirala, and Nuno Carvalhais

This repo is a forked repo from [SindbadTutorials.jl](https://github.com/LandEcosystems/SindbadTutorials.jl) for the Challenges 13 of the AI4PEX winter school at Athens in 2026. The original repo is developed by the team of Model-Data Integration Group at Max Planck Institute of Biogeochemistry. The tutorials are designed to be run in a `terminal` or in a `Colab` notebook, and they will guide you through the process of building hybrid models to learn global parameterizations of ecosystem processes, particularly focusing on carbon and water fluxes. The tutorials will also support the Challenge 17, by exploring different ML architectures for biogeochemical processes (for example, GPP sensitivity to air temperature) to improve the generalizability of the hybrid models across different locations and time scales.

## Description:
Land carbon and water fluxes shape the feedback between terrestrial ecosystems and climate, yet traditional land models remain hampered by structural error and equifinality. Hybrid models—embedding machine learning (ML) modules inside mechanistic frameworks—address several of these gaps by combining physical consistency with data driven flexibility. So far, pioneering work linking process knowledge and ML has already demonstrated superior realism across scales, while underlining the need for richer observations to resolve coupled C–H₂O dynamics. This is demonstrated by the limitation in learning the spatial and temporal controls of parameters that modulate the responses of ecosystems to weather and climate variability.The challenge lies in the need for intensive and long-term observations that underpin robust and comprehensive representations of ecosystem functioning. Although hundreds of locations with such observations exist worldwide, we still observe significant limitations in parameter generalization, consequently limiting our ability to predict ecosystem function. The challenge here is to overcome the previous generalizability in predicting carbon and water fluxes using a hybrid modelling approach. Based on a global open dataset and the SINDBAD hybrid modelling framework, the project will be open to a wide range of approaches towards generalization, from different ML architectures to the ingestion of foundation models.

---

## Motivation:

The challenge of learning global parameterizations of ecosystem processes using hybrid modelling is motivated by the need to improve our understanding and prediction of how terrestrial ecosystems respond to climate change. Traditional land models have limitations in accurately representing the complex interactions between carbon and water fluxes, which are crucial for predicting ecosystem responses to weather and climate variability. Hybrid models, which combine mechanistic frameworks with machine learning (ML) modules, offer a promising approach to address these limitations by leveraging both physical consistency and data-driven flexibility. 

## Challenge Objectives:

The main objectives of this challenge are to:
1. Develop and implement hybrid models that can learn global parameterizations of ecosystem processes, particularly focusing on carbon and water fluxes. This tutorials might start from a simple case of learning the parameters of light using efficiency model of GPP, and then expand to more complex processes and interactions.
2. Evaluate the performance of these hybrid models in predicting ecosystem responses to weather and climate variability using a global open dataset.
3. Explore different ML architectures for biogeochemical processes (for example, GPP sensitivity to air temperature), to improve the generalizability of the hybrid models across different locations and time scales.
4. Identify the key factors and controls that influence the generalization of parameters in hybrid models, and develop strategies to overcome limitations in parameter generalization.

## Two valid tasks
1. Develop a hybrid model that learns the parameters of a light use efficiency model of GPP, and evaluate its performance in predicting GPP across different locations and time scales using a global open dataset. Explore different combinitions of static covariates (e.g., plant functional types, soil properties) to improve the generalizability of the model. Examples could be found in and [Task_Parameter.ipynb](https://colab.research.google.com/github/AI4PEX/SindbadTutorials.jl/blob/ai4pex_winter_school_clean/tutorials/hybrid_inversion_Parameter/Task_Parameter.ipynb).
2. Develop a hybrid model that learns the sensitivity of GPP to air temperature, and evaluate its performance in predicting GPP across different locations and time scales using a global open dataset. Explore different ML architectures (e.g., feedforward neural networks, recurrent neural networks) to improve the generalizability of the model. Examples could be found in [Step02_implement_ml_model.ipynb](https://colab.research.google.com/github/AI4PEX/SindbadTutorials.jl/blob/ai4pex_winter_school_clean/tutorials/hybrid_inversion_Process/Step02_implement_ml_model.ipynb).

## Getting the repo

The following command will install the tutorial files and SINDBAD itself. First of all, go to `colab` (or use `colab` extension if you are using VSCode), then go to `Tools` -> `Command Palette` -> `show terminal` to open a terminal in `colab`. Then mount your Google Drive by:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
Then, run the following command in the terminal to clone the repo:

or 
```bash
git clone https://github.com/AI4PEX/SindbadTutorials.jl.git
```

Note the root directory of where the repo is (```cd SindbadTutorials.jl```), for convenience, we'll call it `repo_root` from now on.

The tutorial files are stored in the `tutorials` subdirectory, and organised by topic or summer school or other event. The current tutorials are:
- hybrid_parameter_learning
- hybrid_process_learning
- insitu_inversion
- global_inversion



## Install Julia

Use Juliaup to install `Julia`. See instructions here:

https://github.com/JuliaLang/juliaup

Or simply running following commands to setup julia 1.11.9:
```bash
curl -fsSL https://install.julialang.org | sh
juliaup status
juliaup add 1.11.9
juliaup default 1.11.9
```

The, install the VS Code Julia extension: 

https://marketplace.visualstudio.com/items?itemName=julialang.language-server


# Install Tutorial Environment
Open a terminal at the root of this repo (`repo_root`)

Start up `Julia`, e.g., in Terminal:

```bash
julia
```

or in VSCode, open the root folder and start the REPL (Ctrl+Shift+J)

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


# Tutorials

The tutorials are located under `./tutorials/hybrid_*Task*`. Follow the instructions in the `.md` files, `.jl` scripts or `.ipynb` notebooks.