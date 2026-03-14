using Pkg; Pkg.add(["Flux", "ProgressMeter", "Zarr", "Statistics", "JLD2", "ForwardDiff", "CairoMakie"])
using Flux, Statistics, ProgressMeter
using Zarr
using JLD2
using ForwardDiff
using CairoMakie

# ── Configuration ─────────────────────────────────────────────────────────────
data_path = "https://s3.bgc-jena.mpg.de:9000/sindbad/FLUXNET_v2023_12_1D_REPLACED_Noise003_v1.zarr"

input_list       = ["atmCO2_SCRIPPS_global", "SW_IN_ERAIv2_gfld", "P_ERAIv2_gfld",
                    "SW_IN_POT_ONEFlux",     "NETRAD_ERAIv2_gfld", "TA_DayTime_ERAIv2_gfld",# "TA_ERAIv2_gfld",
                    "VPD_DayTime_ERAIv2_gfld",
                    # "VPD_ERAIv2_gfld"
                    ]
output_list      = ["GPP_NT"]
output_mask_list = ["GPP_QC_NT_merged"]
tair_idx         = findfirst(==("TA_DayTime_ERAIv2_gfld"), input_list)  # index of Tair in input_list

# ── Site splits (1-based, out of 205 total sites) ─────────────────────────────
# Edit these lists to choose which sites to train/validate on.
training_sites   = collect(1:164)    # ← change as needed
validation_sites = collect(165:205)  # ← change as needed

# ── Load raw arrays from Zarr ─────────────────────────────────────────────────
println("Opening remote Zarr store...")
sindbad_data = zopen(data_path);

println("Loading input variables $(input_list)...")
# Each variable is T × S → stack along a new feature dimension → T × S × F
X_raw = cat([Float32.(sindbad_data[v][:, :]) for v in input_list]...; dims=3);  # T × S × F

println("Loading target: $(output_list[1])...")
Y_raw  = Float32.(sindbad_data[output_list[1]][:, :]);            # T × S

println("Loading QC mask: $(output_mask_list[1])...")
QC_raw = Float32.(sindbad_data[output_mask_list[1]][:, :]);        # T × S

T, S, F = size(X_raw);
println("Data shape: T=$T timesteps × S=$S sites × F=$F features")

# ── Helper: collect valid (input, target) pairs for a set of sites ────────────
# A sample is kept only when:
#   • QC > 0.85  (good-quality measurement)
#   • no NaN in the output
#   • no NaN in any input feature
function collect_samples(sites)
    xs = Vector{Vector{Float32}}()
    ys = Vector{Float32}()
    for s in sites
        for t in 1:T
            y  = Y_raw[t, s]
            qc = QC_raw[t, s]
            x  = X_raw[t, s, :]
            isnan(y)      && continue
            isnan(qc)     && continue
            qc < 0.85f0   && continue          # keep only QC > 0.85
            any(isnan, x) && continue
            x[tair_idx] < 0f0 && continue      # mask sub-zero air temperature
            push!(xs, x)
            push!(ys, y)
        end
    end
    X = reduce(hcat, xs)   # F × N
    Y = reshape(ys, 1, :)  # 1 × N
    return X, Y
end

println("Collecting training samples ($(length(training_sites)) sites)...")
X_train, Y_train = collect_samples(training_sites);

println("Collecting validation samples ($(length(validation_sites)) sites)...");
X_val, Y_val = collect_samples(validation_sites);

println("Training   samples : $(size(X_train, 2))")
println("Validation samples : $(size(X_val,   2))")

# ── Normalise inputs (z-score, statistics from training data only) ────────────
μ_x = mean(X_train; dims=2);                     # F × 1;
σ_x = std(X_train;  dims=2) .+ Float32(1e-6);    # F × 1

X_train_n = (X_train .- μ_x) ./ σ_x
X_val_n   = (X_val   .- μ_x) ./ σ_x

# Normalise output
μ_y = mean(Y_train)
σ_y = std(Y_train) + Float32(1e-6)

Y_train_n = (Y_train .- μ_y) ./ σ_y
Y_val_n   = (Y_val   .- μ_y) ./ σ_y

# ── DataLoader ────────────────────────────────────────────────────────────────
batchsize = 512
loader = Flux.DataLoader((X_train_n, Y_train_n); batchsize=batchsize, shuffle=true)

# ── Model definition ──────────────────────────────────────────────────────────
model = Chain(
    Dense(F => 64, relu),
    Dense(64 => 32, relu),
    Dense(32 => 1),
)
println("\nModel:")
println(model)

opt_state = Flux.setup(Flux.Adam(1e-3), model)

# ── Training loop ─────────────────────────────────────────────────────────────
n_epochs   = 50
train_loss = Float32[]
val_loss   = Float32[]

println("\nTraining for $n_epochs epochs...")
@showprogress "Epoch " for epoch in 1:n_epochs
    epoch_loss = 0f0
    n_batches  = 0
    for (x_b, y_b) in loader
        loss, grads = Flux.withgradient(model) do m
            Flux.mse(m(x_b), y_b)
        end
        Flux.update!(opt_state, model, grads[1])
        epoch_loss += loss
        n_batches  += 1
    end
    push!(train_loss, epoch_loss / n_batches)
    push!(val_loss,   Flux.mse(model(X_val_n), Y_val_n))
end

# ── Evaluation ────────────────────────────────────────────────────────────────
Y_pred = model(X_val_n) .* σ_y .+ μ_y   # back to original units

ss_res = sum((Y_val .- Y_pred) .^ 2)
ss_tot = sum((Y_val .- mean(Y_val)) .^ 2)
r2     = 1f0 - ss_res / ss_tot

println("\n── Results ─────────────────────────────────────────")
println("Final train loss (normalised MSE) : $(round(train_loss[end]; digits=4))")
println("Final val   loss (normalised MSE) : $(round(val_loss[end];   digits=4))")
println("Validation R²                     : $(round(r2; digits=4))")

# ── Tair stress function via ForwardDiff ──────────────────────────────────────
# Strategy:
#   1. Sweep Tair across its observed training range, holding all other
#      inputs fixed at their training mean (= 0 in normalised space).
#   2. Use ForwardDiff.derivative to get ∂GPP/∂Tair at every sweep point.
#   3. Take abs(∂GPP/∂Tair): around optimal T the slope ≈ 0, at cold/hot
#      extremes the magnitude is large.
#   4. Normalise |∂GPP/∂Tair| to [0,1] → abs_sens_norm
#   5. stress_function = 1 - abs_sens_norm
#        → 1 at optimal T (zero slope, no stress)
#        → 0 at worst T   (max |slope|, full stress)

println("\nComputing Tair stress function (feature $tair_idx: \"TA_DayTime_ERAIv2_gfld\")...")

# In normalised feature space, training mean of every variable = 0
x_mean_n = zeros(Float32, F)

# Sweep raw Tair over the range observed in training data
tair_obs_flat  = filter(!isnan, vec(X_raw[:, training_sites, tair_idx]))
tair_range_raw = collect(Float32, range(max(0f0, minimum(tair_obs_flat)),
                                        maximum(tair_obs_flat); length=200))
tair_range_n   = (tair_range_raw .- μ_x[tair_idx]) ./ σ_x[tair_idx]

# GPP as a function of normalised Tair only.
# Written with a parametric type so ForwardDiff can pass Dual numbers through.
function gpp_vs_tair(tair_n::T) where T<:Real
    x = Vector{T}(x_mean_n)     # promote to Dual when T = Dual
    x[tair_idx] = tair_n
    return only(model(reshape(x, F, 1)))  # scalar, normalised GPP units
end

# ∂GPP_n/∂Tair_n at every sweep point (can be positive or negative)
sens_sweep = [ForwardDiff.derivative(gpp_vs_tair, t) for t in tair_range_n]

# Absolute sensitivity: large at cold/hot extremes, ~0 at optimal T
abs_sens = abs.(sens_sweep)

# Normalise |sensitivity| to [0,1]
abs_sens_lo, abs_sens_hi = extrema(abs_sens)
abs_sens_norm = (abs_sens .- abs_sens_lo) ./ (abs_sens_hi - abs_sens_lo + Float32(1e-6))

# Stress function:  1 = no stress (optimal T),  0 = full stress (worst T)
stress_function = 1f0 .- abs_sens_norm

t_opt = tair_range_raw[argmax(stress_function)]   # where slope == 0
t_lo  = tair_range_raw[argmin(stress_function)]   # where |slope| is max
println("Tair range   : [$(round(minimum(tair_range_raw);digits=1)), "
        * "$(round(maximum(tair_range_raw);digits=1))] °C")
println("Optimal T    : $(round(t_opt; digits=1)) °C  (stress = 1.0)")
println("Most-stressed: $(round(t_lo;  digits=1)) °C  (stress = 0.0)")

# ── Save model + normalisation statistics + stress function ───────────────────
save_path = "./gpp_model.jld2"
jldsave(save_path;
    model_state     = Flux.state(model),  # serialisable weight snapshot
    μ_x, σ_x,                             # input  normalisation (F×1)
    μ_y, σ_y,                             # output normalisation (scalar)
    input_list,                            # feature order needed at inference
    train_loss, val_loss,
    # ── stress function ───────────────────────────────────────────────────────
    tair_range_raw,   # raw Tair sweep values (°C),          length=200
    sens_sweep,       # raw ∂GPP_n/∂Tair_n (signed),         length=200
    abs_sens_norm,    # |∂GPP_n/∂Tair_n| normalised to [0,1],length=200
    stress_function,  # 1 - abs_sens_norm ∈ [0,1],           length=200
    abs_sens_lo = abs_sens_lo,
    abs_sens_hi = abs_sens_hi,
)
println("\nModel + stress function saved to: $save_path")

# ── Figures ───────────────────────────────────────────────────────────────────
println("\nGenerating figures...")

# Convenience: raw Tair for validation samples (same masking as collect_samples)
function collect_samples_with_tair(sites)
    xs  = Vector{Vector{Float32}}()
    ys  = Vector{Float32}()
    ts  = Vector{Float32}()          # raw Tair per sample
    tidx = Vector{Int}()             # timestep index per sample
    for s in sites
        for t in 1:T
            y  = Y_raw[t, s]
            qc = QC_raw[t, s]
            x  = X_raw[t, s, :]
            isnan(y) || isnan(qc) || qc < 0.85f0 || any(isnan, x) && continue
            x[tair_idx] < 0f0 && continue      # mask sub-zero air temperature
            push!(xs,  x)
            push!(ys,  y)
            push!(ts,  x[tair_idx])
            push!(tidx, t)
        end
    end
    X    = reduce(hcat, xs)
    Y    = reshape(ys,  1, :)
    Tair = ts
    Ti   = tidx
    return X, Y, Tair, Ti
end

X_val2, Y_val2, Tair_val, Ti_val = collect_samples_with_tair(validation_sites)
X_val2_n = (X_val2 .- μ_x) ./ σ_x
Y_pred2  = vec(model(X_val2_n) .* σ_y .+ μ_y)
Y_obs2   = vec(Y_val2)

# ── Fig 1 : Tair stress / f_Tair response curve ───────────────────────────────
fig1 = Figure(size=(700, 400))
ax1  = Axis(fig1[1,1];
    xlabel = "Air Temperature (°C)",
    ylabel = "f_Tair  (stress function)",
    title  = "Tair temperature response (ForwardDiff sensitivity)")
lines!(ax1, tair_range_raw, stress_function; color=:firebrick, linewidth=2)
vlines!(ax1, [t_opt]; color=:gray, linestyle=:dash, label="T_opt = $(round(t_opt;digits=1)) °C")
axislegend(ax1; position=:lb)
save("./fig1_stress_function.png", fig1)
println("  saved fig1_stress_function.png")

# ── Fig 2 : training / validation loss curves ─────────────────────────────────
fig2 = Figure(size=(700, 400))
ax2  = Axis(fig2[1,1];
    xlabel = "Epoch",
    ylabel = "MSE loss (normalised)",
    title  = "Training and validation loss")
lines!(ax2, 1:n_epochs, train_loss; color=:steelblue,  linewidth=2, label="train")
lines!(ax2, 1:n_epochs, val_loss;   color=:darkorange, linewidth=2, label="validation")
axislegend(ax2; position=:rt)
save("./fig2_loss_curves.png", fig2)
println("  saved fig2_loss_curves.png")

# ── Fig 3 : time-series comparison (validation sites) ────────────────────────
# Sort by timestep so the x-axis is chronological
ordv   = sortperm(Ti_val)
ti_srt = Ti_val[ordv]
yo_srt = Y_obs2[ordv]
yp_srt = Y_pred2[ordv]

fig3 = Figure(size=(900, 400))
ax3  = Axis(fig3[1,1];
    xlabel = "Sample index (time-sorted)",
    ylabel = "GPP  (gC m⁻² d⁻¹)",
    title  = "Time-series: observed vs estimated GPP  [validation sites]")
lines!(ax3, 1:length(yo_srt), yo_srt; color=(:steelblue, 0.7),  linewidth=1, label="Observed")
lines!(ax3, 1:length(yp_srt), yp_srt; color=(:firebrick, 0.7), linewidth=1, label="Estimated")
axislegend(ax3; position=:lt)
save("./fig3_timeseries.png", fig3)
println("  saved fig3_timeseries.png")

# ── Fig 4 : scatter  Tair vs GPP ─────────────────────────────────────────────
# Bin Tair into 60 bins and show mean ± std for clarity
nbins  = 60
edges  = range(minimum(Tair_val), maximum(Tair_val); length=nbins+1)
bin_obs_mean  = Float32[]
bin_pred_mean = Float32[]
bin_centers   = Float32[]
for i in 1:nbins
    lo, hi = edges[i], edges[i+1]
    mask = (Tair_val .>= lo) .& (Tair_val .< hi)
    sum(mask) < 3 && continue
    push!(bin_centers,   (lo + hi) / 2)
    push!(bin_obs_mean,  mean(Y_obs2[mask]))
    push!(bin_pred_mean, mean(Y_pred2[mask]))
end

fig4 = Figure(size=(700, 500))
ax4  = Axis(fig4[1,1];
    xlabel = "Air Temperature (°C)",
    ylabel = "GPP  (gC m⁻² d⁻¹)",
    title  = "GPP vs Tair  [validation sites, binned means]")
scatter!(ax4, Tair_val, Y_obs2;  color=(:steelblue, 0.15), markersize=3, label="Observed (raw)")
scatter!(ax4, Tair_val, Y_pred2; color=(:firebrick, 0.15), markersize=3, label="Estimated (raw)")
lines!(ax4, bin_centers, bin_obs_mean;  color=:steelblue,  linewidth=2.5, label="Observed (binned mean)")
lines!(ax4, bin_centers, bin_pred_mean; color=:firebrick,  linewidth=2.5, label="Estimated (binned mean)")
axislegend(ax4; position=:lt, merge=true)
save("./fig4_scatter_tair_gpp.png", fig4)
println("  saved fig4_scatter_tair_gpp.png")

# ── How to load in any other script ───────────────────────────────────────────
# using Flux, JLD2
#
# checkpoint  = load("./gpp_model.jld2")
#
# # Reconstruct the same architecture (must match the saved one)
# F_loaded = length(checkpoint["input_list"])
# model = Chain(Dense(F_loaded => 64, relu), Dense(64 => 32, relu), Dense(32 => 1))
# Flux.loadmodel!(model, checkpoint["model_state"])
#
# # Recover normalisation stats
# μ_x, σ_x = checkpoint["μ_x"], checkpoint["σ_x"]
# μ_y, σ_y = checkpoint["μ_y"], checkpoint["σ_y"]
#
# # Run inference on new data (F × N matrix)
# X_new_n = (X_new .- μ_x) ./ σ_x
# Y_pred  = model(X_new_n) .* σ_y .+ μ_y