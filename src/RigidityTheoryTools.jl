module RigidityTheoryTools

using Graphs
using AbstractAlgebra
using BracketAlgebras
using Combinatorics

export Framework, graph, realization, labels, d
export rigidity_matrix, basis_inf_motions, is_infrigid, basis_inf_flex, index, is_genrigid, is_isostatic
export tiedown, condition, pure_condition

# abstract supertype for all embedded graphs. Every subtype has to have Fields
# - `G::SimpleGraph`: graph with `n` vertices and `m` edges
# - `realization::Union{MatrixElem,Matrix}`: `d` x `n` matrix with `d` the dimension of the underlying space. Columns represent vertex positions.
# - `labels::Vector`: vector of vertex labels
abstract type AbstractEmbeddedGraph end

mutable struct Framework <: AbstractEmbeddedGraph
    """
        Framework(G::Graph, realization::Union{MatrixElem,Matrix}; labels::AbstractVector=collect(1:nv(G)))

    A framework is the realization of a graph in a d-dimensional space. It is represented by a graph `G`, a realization `realization` and a vector of vertex labels `labels`.

    # Fields
    - `G::SimpleGraph`: graph with `n` vertices and `m` edges
    - `realization::Union{MatrixElem,Matrix}`: `d` x `n` matrix with `d` the dimension of the underlying space. Columns represent vertex positions.
    - `labels::Vector`: vector of vertex labels
    """
    G::SimpleGraph # graph with n vertices and m edges
    realization::Union{MatrixElem,Matrix} # d x n matrix with d the dimension of the space representing vertex positions
    labels::Vector # vector of vertex labels
end

"""
    Framework(G::Graphs.AbstractSimpleGraph, realization::Union{MatrixElem,Matrix}; labels::AbstractVector=collect(1:nv(G)))

Return the Framework with graph `G`, realization `realization` and labels `labels`. If `labels` is not provided, it is set to `1:nv(G)`.

# Examples
```jldoctest
julia> G = complete_graph(4);

julia> F = Framework(G, [0 1 0 0; 0 0 1 0; 0 0 0 1])
Framework with 4 vertices and 6 edges, vertex labels [1, 2, 3, 4] and 3-realization
[0 1 0 0; 0 0 1 0; 0 0 0 1]
```

```jldoctest
julia> G = complete_graph(4);

julia> F = Framework(G, [0 1 0 0; 0 0 1 0; 0 0 0 1], labels = ["a", "b", "c", "d"])
Framework with 4 vertices and 6 edges, vertex labels ["a", "b", "c", "d"] and 3-realization
[0 1 0 0; 0 0 1 0; 0 0 0 1]
```
"""
function Framework(G::Graphs.AbstractSimpleGraph, realization::Union{MatrixElem,Matrix}; labels::AbstractVector=collect(1:nv(G)))
    n = nv(G)
    if size(realization, 2) != n
        throw(ArgumentError("The number of columns of the realization matrix must be equal to the number of vertices of the graph"))
    end
    if length(labels) != n
        throw(ArgumentError("The number of labels must be equal to the number of vertices of the graph"))
    end
    if eltype(labels) <: Integer && labels != collect(1:n)
        throw(ArgumentError("If the labels are a vector of integers it has to be equal to 1:n"))
    end
    return Framework(G, realization, labels)
end

"""
    Framework(G::Graphs.AbstractSimpleGraph, d::Int=2; labels::Vector=collect(1:nv(G)))

Return the Framework with graph `G`, a random realization `d`-dimensional realization and labels `labels`. If `labels` is not provided, it is set to `1:nv(G)`.
"""
function Framework(G::Graphs.AbstractSimpleGraph, d::Int=2; labels::Vector=collect(1:nv(G)))
    n = nv(G)
    realization = rand(d, n)
    return Framework(G, realization, labels)
end

"""
    graph(f::AbstractEmbeddedGraph)

Return the underlying graph of the Framework `f`.
"""
graph(f::AbstractEmbeddedGraph) = deepcopy(f.G)

"""
    realization(f::AbstractEmbeddedGraph)

Return the realization of the framework `f`.
"""
realization(f::AbstractEmbeddedGraph) = deepcopy(f.realization)

"""
    realization!(f::AbstractEmbeddedGraph, realization::Union{MatrixElem,Matrix})

Set the realization of the Framework `f` to `realization` and return the updated Framework.
"""
function realization!(f::AbstractEmbeddedGraph, realization::Union{MatrixElem,Matrix})
    if size(realization, 2) != nv(f.G)
        throw(ArgumentError("The number of columns of the realization matrix must be equal to the number of vertices of the graph"))
    end
    f.realization = realization

    return f
end

"""
    labels(f::AbstractEmbeddedGraph)

Return the vertex labels of the Framework `f`.
"""
labels(f::AbstractEmbeddedGraph) = deepcopy(f.labels)

"""
    labels!(f::AbstractEmbeddedGraph, labels::AbstractVector)

Set the vertex labels of the Framework `f` to `labels` and return the updated Framework.
"""
function labels!(f::AbstractEmbeddedGraph, labels::AbstractVector)
    if length(labels) != nv(f.G)
        throw(ArgumentError("The number of labels must be equal to the number of vertices of the graph"))
    end
    if eltype(labels) <: Integer && labels != collect(1:nv(f.G))
        throw(ArgumentError("If the labels are a vector of integers it has to be equal to 1:n"))
    end
    f.labels = labels

    return f
end

"""
    d(f::AbstractEmbeddedGraph)

Return the dimension of the underlying space of the framework `f`.
"""
d(f::AbstractEmbeddedGraph) = size(f.realization, 1)

# displaying Frameworks
function Base.show(io::IO, f::AbstractEmbeddedGraph)
    println(io, "Framework with $(nv(graph(f))) vertices and $(ne(graph(f))) edges, vertex labels $(labels(f)) and $(d(f))-realization")
    println(io, f.realization)
end

include("infinitesimal.jl")
include("henneberg.jl")
include("pure_condition.jl")

end
