module NLlsq

using LinearAlgebra
using Statistics

"""
    nlsq_fit(f, x, y, p0; algorithm=:levenberg_marquardt, kwargs...)

Fits a non-linear model f to data points (x, y) starting from initial parameters p0.

# Arguments
- `f`: Model function of the form f(x, p) where x is the independent variable 
       and p is the parameter vector
- `x`: Independent variable values (vector)
- `y`: Observed/dependent variable values (vector)
- `p0`: Initial parameter guess (vector)
- `algorithm`: Algorithm to use (:levenberg_marquardt or :gauss_newton)
- `kwargs...`: Additional options passed to the algorithm

# Returns
A named tuple containing fit results and statistics
"""
function nlsq_fit(f, x, y, p0; algorithm=:levenberg_marquardt, kwargs...)
    # Select and apply the chosen algorithm
    if algorithm == :levenberg_marquardt
        p, converged, iter, cost = levenberg_marquardt(f, x, y, p0; kwargs...)
    elseif algorithm == :gauss_newton
        p, converged, iter, cost = gauss_newton(f, x, y, p0; kwargs...)
    else
        error("Unknown algorithm: $algorithm")
    end
    
    # Compute statistics about the fit
    stats = compute_statistics(f, x, y, p)
    
    return (
        parameters = p,
        converged = converged,
        iterations = iter,
        cost = cost,
        statistics = stats
    )
end

"""
    levenberg_marquardt(f, x, y, p0; λ=0.01, λ_factor=10.0, max_iterations=100, 
                         tolerance=1e-8, scale_parameters=true)

Implements the Levenberg-Marquardt algorithm with parameter scaling.
"""
function levenberg_marquardt(f, x, y, p0; λ=0.01, λ_factor=10.0, max_iterations=100, 
                             tolerance=1e-8, scale_parameters=true)
    p = copy(p0)  # Current parameter vector
    n = length(p)  # Number of parameters
    m = length(y)  # Number of data points
    
    # Parameter scaling factors (to improve conditioning)
    scales = ones(n)
    if scale_parameters
        scales = [max(abs(p[j]), 1.0) for j in 1:n]
    end
    
    # Compute initial residuals and cost
    residuals = compute_residuals(f, x, y, p)
    cost = sum(residuals.^2)
    
    # Initialize variables for the iteration
    iter = 0
    converged = false
    
    while !converged && iter < max_iterations
        # Compute Jacobian using central differences
        J = compute_jacobian(f, x, p; method=:central)
        
        # Apply parameter scaling to Jacobian
        if scale_parameters
            for j in 1:n
                J[:, j] *= scales[j]
            end
        end
        
        # Compute gradient: g = J'*residuals
        g = J' * residuals
        
        # Compute approximate Hessian: H = J'*J
        H = J' * J
        
        # Add damping to the Hessian diagonal with proper scaling
        H_lm = H + λ * Diagonal(diag(H) .+ 1e-8)
        
        # Compute the step: Δp = -H_lm \ g (with scaling)
        Δp = -H_lm \ g
        
        # Unscale the step if parameter scaling is used
        if scale_parameters
            Δp ./= scales
        end
        
        # Try the step with line search if needed
        α = 1.0
        p_new = p + α * Δp
        residuals_new = compute_residuals(f, x, y, p_new)
        cost_new = sum(residuals_new.^2)
        
        # Simple backtracking line search to guarantee improvement
        line_search_iters = 0
        while cost_new >= cost && α > 1e-4 && line_search_iters < 10
            α *= 0.5
            p_new = p + α * Δp
            residuals_new = compute_residuals(f, x, y, p_new)
            cost_new = sum(residuals_new.^2)
            line_search_iters += 1
        end
        
        # Accept or reject the step
        if cost_new < cost
            # Step decreased the cost, accept it
            p .= p_new
            residuals .= residuals_new
            cost = cost_new
            
            # Decrease λ (makes algorithm more like Gauss-Newton)
            λ = max(λ / λ_factor, 1e-10)
            
            # Check convergence (scaled by parameter magnitudes)
            if norm(α * Δp ./ max.(abs.(p), 1e-8)) < tolerance
                converged = true
            end
        else
            # Step increased cost, reject it
            # Increase λ (makes algorithm more like gradient descent)
            λ = min(λ * λ_factor, 1e10)
        end
        
        iter += 1
    end
    
    return p, converged, iter, cost
end

"""
    gauss_newton(f, x, y, p0; max_iterations=100, tolerance=1e-8, scale_parameters=true)

Implements the Gauss-Newton algorithm for non-linear least squares with parameter scaling.
"""
function gauss_newton(f, x, y, p0; max_iterations=100, tolerance=1e-8, scale_parameters=true)
    p = copy(p0)
    n = length(p)
    m = length(y)
    
    # Parameter scaling factors (to improve conditioning)
    scales = ones(n)
    if scale_parameters
        scales = [max(abs(p[j]), 1.0) for j in 1:n]
    end
    
    # Compute initial residuals and cost
    residuals = compute_residuals(f, x, y, p)
    cost = sum(residuals.^2)
    
    # Initialize variables for the iteration
    iter = 0
    converged = false
    
    while !converged && iter < max_iterations
        # Compute Jacobian using central differences
        J = compute_jacobian(f, x, p; method=:central)
        
        # Apply parameter scaling to Jacobian
        if scale_parameters
            for j in 1:n
                J[:, j] *= scales[j]
            end
        end
        
        # Compute Gauss-Newton step: Δp = -(J'J)^{-1}J'r
        Δp = -(J' * J) \ (J' * residuals)
        
        # Unscale the step if parameter scaling is used
        if scale_parameters
            Δp ./= scales
        end
        
        # Update parameters
        p_new = p + Δp
        
        # Compute new residuals and cost
        residuals_new = compute_residuals(f, x, y, p_new)
        cost_new = sum(residuals_new.^2)
        
        # Simple line search to ensure cost reduction
        α = 1.0
        line_search_iters = 0
        while cost_new >= cost && α > 1e-4 && line_search_iters < 10
            α *= 0.5
            p_new = p + α * Δp
            residuals_new = compute_residuals(f, x, y, p_new)
            cost_new = sum(residuals_new.^2)
            line_search_iters += 1
        end
        
        # Update state
        if cost_new < cost
            p .= p_new
            residuals .= residuals_new
            cost = cost_new
            
            # Check convergence (scaled by parameter magnitudes)
            if norm(α * Δp ./ max.(abs.(p), 1e-8)) < tolerance
                converged = true
            end
        else
            # Line search failed
            converged = true  # Exit the loop
        end
        
        iter += 1
    end
    
    return p, converged, iter, cost
end

# Helper functions

"""
    compute_residuals(f, x, y, p)

Compute the residuals (differences between model predictions and observations).
"""
function compute_residuals(f, x, y, p)
    return [y[i] - f(x[i], p) for i in 1:length(x)]
end

"""
    compute_jacobian(f, x, p; method=:central, epsilon=1e-8)

Compute the Jacobian matrix using finite differences.
Supports both forward (:forward) and central (:central) difference methods.
"""
function compute_jacobian(f, x, p; method=:central, epsilon=1e-8)
    n = length(p)  # Number of parameters
    m = length(x)  # Number of data points
    
    J = zeros(m, n)  # Jacobian matrix
    
    # Scale epsilon for each parameter to improve numerical stability
    eps_vec = [max(abs(p[j]) * epsilon, epsilon) for j in 1:n]
    
    if method == :forward
        for j in 1:n
            # Compute parameter perturbation
            dp = zeros(n)
            dp[j] = eps_vec[j]
            
            # Forward difference approximation
            f_plus = [f(x[i], p + dp) for i in 1:m]
            f_current = [f(x[i], p) for i in 1:m]
            
            J[:, j] = (f_plus .- f_current) ./ eps_vec[j]
        end
    elseif method == :central
        for j in 1:n
            # Compute parameter perturbations for central difference
            dp_plus = zeros(n)
            dp_minus = zeros(n)
            dp_plus[j] = eps_vec[j]
            dp_minus[j] = -eps_vec[j]
            
            # Central difference approximation (more accurate)
            f_plus = [f(x[i], p + dp_plus) for i in 1:m]
            f_minus = [f(x[i], p + dp_minus) for i in 1:m]
            
            J[:, j] = (f_plus .- f_minus) ./ (2 * eps_vec[j])
        end
    else
        error("Unknown differentiation method: $method")
    end
    
    return J
end

"""
    compute_statistics(f, x, y, p)

Compute statistics about the fit (R-squared, parameter uncertainties, etc.).
"""
function compute_statistics(f, x, y, p)
    residuals = compute_residuals(f, x, y, p)
    m = length(y)
    n = length(p)
    
    # Degrees of freedom
    dof = m - n
    
    # Residual sum of squares
    rss = sum(residuals.^2)
    
    # Total sum of squares
    y_mean = mean(y)
    tss = sum((y .- y_mean).^2)
    
    # R-squared
    r_squared = 1 - rss / tss
    
    # Adjusted R-squared
    r_squared_adj = 1 - (rss / dof) / (tss / (m - 1))
    
    # Standard error of the regression
    se = sqrt(rss / dof)
    
    # Compute Jacobian for parameter uncertainty estimation
    J = compute_jacobian(f, x, p)
    
    # Covariance matrix of parameters
    cov_p = try
        inv(J' * J) * se^2
    catch
        fill(NaN, (n, n))
    end
    
    # Standard errors of parameters
    se_p = sqrt.(diag(cov_p))
    
    return (
        residuals = residuals,
        rss = rss,
        r_squared = r_squared,
        r_squared_adj = r_squared_adj,
        se = se,
        cov_p = cov_p,
        se_p = se_p
    )
end

"""
    print_fit_summary(result)

Prints a summary of non-linear least squares fit results.
"""
function print_fit_summary(result)
    println("Fit Summary:")
    println("------------")
    println("Converged: $(result.converged)")
    println("Iterations: $(result.iterations)")
    println("Final cost: $(result.cost)")
    println("Parameters:")
    for (i, p) in enumerate(result.parameters)
        se = result.statistics.se_p[i]
        println("  p$i = $p ± $se")
    end
    println("R² = $(result.statistics.r_squared)")
    println("Adjusted R² = $(result.statistics.r_squared_adj)")
    println("Standard error of regression = $(result.statistics.se)")
end

"""
    predict(f, p)

Creates a prediction function from the fitted parameters.
"""
function predict(f, p)
    return x -> f(x, p)
end

"""
    debug_fit(f, x, y, p0; steps=5)

Runs a few steps of the optimization process with detailed output for debugging.
"""
function debug_fit(f, x, y, p0; steps=5)
    p = copy(p0)
    residuals = compute_residuals(f, x, y, p)
    cost = sum(residuals.^2)
    
    println("Initial parameters: $p")
    println("Initial cost: $cost")
    
    for i in 1:steps
        println("\n--- Step $i ---")
        
        # Compute Jacobian
        J = compute_jacobian(f, x, p; method=:central)
        
        # Show the condition number of J'J to diagnose ill-conditioning
        H = J' * J
        cond_num = try
            cond(H)
        catch
            Inf
        end
        println("Condition number of J'J: $cond_num")
        
        # Show gradient norm
        g = J' * residuals
        println("Gradient norm: $(norm(g))")
        
        # Compute a regularized step
        λ = 0.01 * (i == 1 ? 1.0 : 10.0^(i-2))
        H_lm = H + λ * Diagonal(diag(H) .+ 1e-8)
        
        Δp = try
            -H_lm \ g
        catch
            println("Matrix inversion failed!")
            zeros(length(p))
        end
        
        println("Step with λ=$λ: $Δp")
        
        # Try the step
        p_new = p + Δp
        residuals_new = compute_residuals(f, x, y, p_new)
        cost_new = sum(residuals_new.^2)
        
        println("New cost: $cost_new ($(cost_new < cost ? "improved" : "worsened"))")
        
        if cost_new < cost
            p = p_new
            residuals = residuals_new
            cost = cost_new
        end
    end
    
    return p, cost
end

export nlsq_fit, levenberg_marquardt, gauss_newton
export compute_residuals, compute_jacobian, compute_statistics
export print_fit_summary, predict, debug_fit

end # module