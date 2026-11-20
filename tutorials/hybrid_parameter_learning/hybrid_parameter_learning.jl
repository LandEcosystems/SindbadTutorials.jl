# ================================== using tools ==================================================
# some of the things that will be using... Julia tools, SINDBAD tools, local codes...
using Revise
using SindbadTutorials
using Sindbad.MachineLearning
using Sindbad.MachineLearning.Random
using SindbadTutorials.Plots
using Flux, PreallocationTools, FiniteDiff, FiniteDifferences, ForwardDiff, Optimisers

## get the sites to run experiment on
selected_site_indices = getSiteIndicesForHybrid();
do_random = 0# set to integer values larger than zero to use random selection of #do_random sites
if do_random > 0
    Random.seed!(1234)
    selected_site_indices = first(shuffle(selected_site_indices), do_random)
end

# ================================== get data / set paths ========================================= 
path_output         = "";

# this one takes a hugh amount of time, leave it here for reference
# ================================== setting up the experiment ====================================
# experiment is all set up according to a (collection of) json file(s)
model_setup = "WROASTED_HB";
# model_setup = "LUE";
path_experiment_json    = joinpath(@__DIR__,"..","setups",model_setup,"experiment_hybrid.json");
path_training_folds     = "";#joinpath(@__DIR__,"..","setups",model_setup,"nfolds_sites_indices.jld2");

replace_info = Dict(
    "forcing.subset.site" => getSiteNames(selected_site_indices),
    "optimization.optimization_cost_threaded" => false,
    "optimization.optimization_parameter_scaling" => nothing,
    "hybrid.ml_training.fold_path" => nothing,
);

# generate the info and other helpers
info            = getExperimentInfo(path_experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
sites_forcing   = forcing.data[1].site;
hybrid_helpers  = prepHybrid(forcing, observations, info, info.hybrid.ml_training.method);

# ================================== train the hybrid model =======================================
trainML(hybrid_helpers, info.hybrid.ml_training.method)
#

# ================================== posterior run diagnostics =====================================
# retrieve the inversed parameters
sites_forcing = forcing.data[1].site;
ml_model = hybrid_helpers.ml_model;
xfeatures = hybrid_helpers.features.data;
loss_functions = hybrid_helpers.loss_functions;
loss_component_functions = hybrid_helpers.loss_component_functions;

params_sites = ml_model(xfeatures)
@info "params_sites: [$(minimum(params_sites)), $(maximum(params_sites))]"

scaled_params_sites = getParamsAct(params_sites, info.optimization.parameter_table)
@info "scaled_params_sites: [$(minimum(scaled_params_sites)), $(maximum(scaled_params_sites))]"

# select a site
# idx = findfirst(x -> x == "DE-Hai", sites_forcing)
site_index = 67
site_name = sites_forcing[site_index]

loc_params = scaled_params_sites(site=site_name).data.data
loss_f_site = loss_functions(site=site_name);
loss_vector_f_site = loss_component_functions(site=site_name);
@time loss_f_site(loc_params)
loss_vector_f_site(loc_params)


# run the model for the site with the default parameters and with the parameters from
# the hybrid model. the run helpers are prepared once and used for all the site runs.
run_helpers     = prepTEM(info.models.forward, forcing, observations, info);
site_default    = runSiteWithParams(run_helpers, info, site_index, info.optimization.parameter_table.default);
site_optimized  = runSiteWithParams(run_helpers, info, site_index, loc_params);
# site_optimized.loss_vector and site_optimized.loss are the same as loss_vector_f_site(loc_params)

# do plots, compute some simple statistics e.g. NSE
plotSiteComparison(info, run_helpers.space_observation[site_index],
    (; default=site_default.output, optimized=site_optimized.output);
    file_prefix="$(model_setup)_$(site_name)");
println("Outputs of plotting will be here: " * info.output.dirs.figure);

# run the model for every site with the default and the hybrid parameters
n_sites = length(sites_forcing);
t_loss_def = Vector{Float32}(undef, n_sites);
t_loss_opt = Vector{Float32}(undef, n_sites);
for site_index in 1:n_sites
    @info "site $(site_index): $(sites_forcing[site_index])"
    params_site = scaled_params_sites(site=sites_forcing[site_index]).data.data
    t_loss_def[site_index] = runSiteWithParams(run_helpers, info, site_index, info.optimization.parameter_table.default).loss
    t_loss_opt[site_index] = runSiteWithParams(run_helpers, info, site_index, params_site).loss
end

# now plot histogram of scalar losses
histogram(
  t_loss_def,
  bins = 50,
  alpha = 0.5,
  label = "default",
  xlabel = "Loss (1-NSE)",
  ylabel = "Count",
  title = "Model Loss: default vs optimized"
)
histogram!(
  t_loss_opt,
  bins = 50,
  alpha = 0.5,
  label = "optimized"
)
savefig(joinpath(info.output.dirs.figure, "loss_histogram_scalar.png"))

## check the docs for output at: http://sindbad-mdi.org/pages/develop/hybrid_modeling.html and http://sindbad-mdi.org/pages/develop/sindbad_outputs.html


# useful code snipets
## access and check the loss values
array_loss = hybrid_helpers.loss_array;
array_loss_components = hybrid_helpers.loss_array_components;


## play around with gradient for sites and batch to understand internal workings
sites_forcing = forcing.data[1].site;
ml_model = hybrid_helpers.ml_model;
xfeatures = hybrid_helpers.features.data;
loss_functions = hybrid_helpers.loss_functions;
loss_component_functions = hybrid_helpers.loss_component_functions;

params_sites = ml_model(xfeatures)
@info "params_sites: [$(minimum(params_sites)), $(maximum(params_sites))]"

scaled_params_sites = getParamsAct(params_sites, info.optimization.parameter_table)
@info "scaled_params_sites: [$(minimum(scaled_params_sites)), $(maximum(scaled_params_sites))]"


## try for a site
site_index = 1
site_name = sites_forcing[site_index]

loc_params = scaled_params_sites(site=site_name).data.data
loss_f_site = loss_functions(site=site_name);
loss_vector_f_site = loss_component_functions(site=site_name);
@time loss_f_site(loc_params)
loss_vector_f_site(loc_params)

@time g_site = gradientSite(info.hybrid.ml_gradient.method, loc_params, info.hybrid.ml_gradient.options, loss_functions(site=site_name))

## try for a batch
sites_batch = hybrid_helpers.sites.training[1:info.hybrid.ml_training.options.batch_size]
scaled_params_batch = scaled_params_sites(; site=sites_batch)
grads_batch = zeros(Float32, size(scaled_params_batch, 1), length(sites_batch));

g_batch = gradientBatch!(info.hybrid.ml_gradient.method, grads_batch, info.hybrid.ml_gradient.options, loss_functions, scaled_params_batch, sites_batch; showprog=true)
#
