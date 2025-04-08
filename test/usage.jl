using NLlsq
using Plots

# Generate some synthetic data
function true_model(x, p_true)
    return p_true[1] * exp(-p_true[2] * x) + p_true[3]
end

# True parameters
p_true = [10.0, 0.1, 2.0]

# Generate data with noise
x_data = collect(0:0.5:20)
y_true = [true_model(xi, p_true) for xi in x_data]
y_data = y_true + randn(length(x_data)) * 0.5  # Reduced noise for better fitting

# Define the model function
function model_func(x, p)
    return p[1] * exp(-p[2] * x) + p[3]
end

# Initial parameter guess - try different starting points if needed
p0 = [8.0, 0.08, 1.5]

# Debug the fit to see what's happening
debug_params, debug_cost = debug_fit(model_func, x_data, y_data, p0)

# Fit with improved algorithm
result = nlsq_fit(model_func, x_data, y_data, p0; 
                 algorithm=:levenberg_marquardt, 
                 scale_parameters=true,
                 max_iterations=2000,
                 tolerance=1e-15)

result_gauss_newton = nlsq_fit(model_func, x_data, y_data, p0; 
                 algorithm=:gauss_newton, 
                 scale_parameters=true,
                 max_iterations=2000,
                 tolerance=1e-15)

# Print results
print_fit_summary(result)
print_fit_summary(result_gauss_newton)

# Plot the results
plot(x_data, y_data, label="Noisy Data", marker=:o, markersize=4, legend=:topright)
plot!(x_data, y_true, label="True Model", lw=2, color=:green)
plot!(x_data, model_func.(x_data, Ref(result.parameters)), label="Fitted Model (LM)", lw=2, color=:red)
plot!(x_data, model_func.(x_data, Ref(result_gauss_newton.parameters)), label="Fitted Model (GN)", lw=2, color=:blue)