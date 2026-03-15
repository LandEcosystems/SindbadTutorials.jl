# Task 02: Build a hybrid model for GPP sensitivity to air temperature

Author: Xu Shan, Reda ElGhawi, Sujan Koirala, and Nuno Carvalhais

In this tutorial, we will build a hybrid model to learn the sensitivity of GPP to air temperature, and evaluate its performance in predicting GPP across different locations and time scales using a global open dataset. We will explore different ML architectures (e.g., feedforward neural networks, recurrent neural networks) to improve the generalizability of the model.

This script is written in the form of md file because of some of the commands cannot be run in the REPL, but need to be run in the terminal. You can copy and paste the commands into the terminal, or run the script directly in the terminal with `julia Step01_add_model.jl`.

## Step 01: Add a new model in Sindbad
In this step, we will add a new model in Sindbad to learn the sensitivity of GPP to air temperature. We will use an external ML model to learn the relationship between GPP and air temperature. The purpose here is to just add the new model ```type``` and the corresponding ```approach``` in Sindbad, without actually implementing the ML model. We will implement the ML model in the next step in an outer script, and link it to the Sindbad model in the step after that.

```julia
Pkg.instantiate()
```

```julia
using Revise
using SindbadTutorials
generateSindbadApproach(:gppAirT, 
                        "Effect of temperature on GPP: 1 indicates no temperature stress, 0 indicates complete stress.", 
                        :externalNN, 
                        "Use external ML model", 
                        1)
```
---
## Step 02: Implement the ML model in an outer script
In this step, we will implement the ML model in an outer script, and link it to the Sindbad model in the step after that. We will use a simple feedforward neural network as an example, but you can explore different ML architectures (e.g., recurrent neural networks) to improve the generalizability of the model.
Please refer to the `Step02_implement_ml_model.jl` script for the implementation of the ML model. The script is written in a way that it can be run in the terminal, and it will save the trained ML model as a file that can be loaded in the Sindbad model in the next step. Alternatively, you can also refer to the notebook `Step02_implement_ml_model.ipynb` for the implementation of the ML model in a Jupyter notebook. The notebook is more interactive and allows you to visualize the training process and the results, but it may not be as convenient for running the script in the terminal. You can choose either way to implement the ML model, depending on your preference and needs.

---

## Step 03: Link the ML model to the Sindbad model
In this step, we will link the trained ML model to the Sindbad model that we created in Step 01. We will load the trained ML model in the Sindbad model, and use it to predict the sensitivity of GPP to air temperature. We will then evaluate the performance of the hybrid model in predicting GPP across different locations and time scales using a global open dataset.

The core idea of the part is to use the predefined model approach (actually a julia `type`) in Sindbad, and ***extend*** the `define`, `precompute`, and `compute` functions for this new model approach. In the `define` function, we will load the trained ML model and define the parameters and variables for the Sindbad model. In the `precompute` function, we will prepare the input data for the ML model, and in the `compute` function, we will use the ML model to predict the sensitivity of GPP to air temperature, and then use it to compute GPP. Please refer to the `Step03_link_ml_model.jl` script for the implementation of linking the ML model to the Sindbad model. The script is written in a way that it can be run in the terminal, and it will evaluate the performance of the hybrid model in predicting GPP across different locations and time scales using a global open dataset.

For step 03, we have two different ways to build the sensitivity of GPP to air temperature in the Sindbad model: 1) use a lookup table to store the predicted sensitivity of GPP to air temperature for different locations and time scales, and then use the lookup table in the `compute` function to get the sensitivity for each location and time step. This is shown in the `Step03_link_ml_model_lut.jl` script; 2) directly call the ML model in the `compute` function to get the sensitivity of GPP to air temperature for each location and time step. The first way is more efficient, but it may not be as accurate as the second way, especially when the input data for the ML model is high-dimensional and complex. This is shown in the `Step03_link_ml_model.jl` script. The second way is more accurate, but it may be computationally expensive, especially when the input data for the ML model is high-dimensional and complex. You can choose either way to link the ML model to the Sindbad model, depending on your preference and needs. 