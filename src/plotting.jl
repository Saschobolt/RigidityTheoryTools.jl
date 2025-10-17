using GraphMakie
using GLMakie

"""
    plot_framework(F::Framework; kwargs...)

Plots a 2D or 3D Framework, optionally visualizing self-stresses and infinitesimal motions.

# Keyword Arguments
- `resolution=(800, 600)`: The size of the output figure.
- `plot_labels=false`: If `true`, plots vertex labels *inside* the nodes. If `false`, plots them *next to* the nodes.
- `self_stress=nothing`: A vector of stress values, one for each edge in `edges(graph(F))`.
- `inf_motion=nothing`: A vector of `d*n` velocity components for the `n` vertices.
- `stress_color=:orange`: Color for the self-stress edge labels.
- `motion_color=:red`: Color for the infinitesimal motion arrows.
- `motion_scale=0.3`: A scaling factor for the length of motion arrows to improve visibility.
- Other `kwargs...`: Are passed directly to `GraphMakie.graphplot`.
"""
function plot_framework(F::Framework;
    resolution=(800, 600),
    # New argument for label style
    show_labels=false,
    self_stress=nothing,
    inf_motion=nothing,
    stress_color=:orange,
    motion_color=:red,
    motion_scale=0.8,
    kwargs...)

    # Extract data from the framework
    g = graph(F)
    pos = realization(F)
    labs = string.(labels(F))
    dim = d(F)
    n = nv(g)

    # --- Prepare arguments for graphplot ---
    plot_args = Dict{Symbol,Any}(:layout => pos)

    # 1. Handle vertex labels (internal vs. external)
    if show_labels
        plot_args[:ilabels] = labs
    end

    # 2. Handle self-stress labels on edges
    if self_stress !== nothing
        if length(self_stress) != ne(g)
            throw(ArgumentError("`self_stress` must have one entry for each edge ($(ne(g)))"))
        end
        rounded_stresses = [string(round(s, digits=2)) for s in self_stress]
        plot_args[:elabels] = rounded_stresses
        plot_args[:elabels_color] = stress_color
        plot_args[:elabels_textsize] = 15
    end

    # Create figure and axis/scene
    fig = Figure(size=resolution)
    ax = (dim == 2) ? Axis(fig[1, 1], aspect=DataAspect()) : LScene(fig[1, 1], show_axis=false)
    if dim == 2
        hidedecorations!(ax)
    end

    # Plot the framework
    if dim == 2
        graphplot!(ax, g;
            plot_args...,
            kwargs...)
    elseif dim == 3
        graphplot!(ax, g;
            arrow_shift=0.0,
            plot_args...,
            kwargs...)
    end

    # Plot infinitesimal motions if provided
    if inf_motion !== nothing
        if length(inf_motion) != dim * n
            throw(ArgumentError("`inf_motion` vector must have d*n = $(dim*n) entries"))
        end
        motion_vectors = reshape(inf_motion, dim, n)
        directions = [Vec{dim,Float32}(motion_vectors[:, i]) for i in 1:n]

        arrows!(ax, pos, directions,
            lengthscale=motion_scale,
            color=motion_color,
            linewidth=dim == 2 ? 1.5 : 0.05,
            arrowsize=dim == 2 ? 10 : 0.15)
    end

    if d(F) == 2
        hidedecorations!(ax)
    end

    return fig
end