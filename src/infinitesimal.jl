"""
    rigidity_matrix(f::AbstractEmbeddedGraph)

Construct the rigidity matrix of the embedded graph `f`.

# Examples
```jldoctest
julia> G = complete_graph(4);

julia> f = Framework(G, [0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.]);

julia> rigidity_matrix(f)
6×12 Matrix{Float64}:
 -1.0   0.0   0.0  1.0   0.0   0.0   0.0  0.0   0.0   0.0   0.0  0.0
  0.0  -1.0   0.0  0.0   0.0   0.0   0.0  1.0   0.0   0.0   0.0  0.0
  0.0   0.0  -1.0  0.0   0.0   0.0   0.0  0.0   0.0   0.0   0.0  1.0
  0.0   0.0   0.0  1.0  -1.0   0.0  -1.0  1.0   0.0   0.0   0.0  0.0
  0.0   0.0   0.0  1.0   0.0  -1.0   0.0  0.0   0.0  -1.0   0.0  1.0
  0.0   0.0   0.0  0.0   0.0   0.0   0.0  1.0  -1.0   0.0  -1.0  1.0
```
"""
function rigidity_matrix(f::AbstractEmbeddedGraph)
    coords = coordinate_matrix(f)
    d, n = size(coords)
    r = zeros(eltype(coords), ne(graph(f)), d * n)
    for (i, e) in enumerate(edges(graph(f)))
        v = src(e)
        w = dst(e)
        r[i, (v-1)*d+1:v*d] = coords[:, v] - coords[:, w]
        r[i, (w-1)*d+1:w*d] = coords[:, w] - coords[:, v]
    end

    return r
end

"""
    basis_inf_motions(f::AbstractEmbeddedGraph)

Calculate a basis for the space of infinitesimal motions of the embedded graph f.
"""
function basis_inf_motions(f::AbstractEmbeddedGraph)
    return nullspace(rigidity_matrix(f))
end

"""
    is_infrigid(f::AbstractEmbeddedGraph)

Return whether the embedded graph `f` is infinitesimally rigid or not.
"""
function is_infrigid(f::AbstractEmbeddedGraph)
    dim = d(f)
    return size(basis_inf_motions(f))[2] == binomial(dim + 1, 2)
end

"""
    basis_inf_flex(f::AbstractEmbeddedGraph)

Calculate a basis for a complement of the space of infinitesimal rigid motions of the embedded graph f.
"""
function basis_inf_flex(f::AbstractEmbeddedGraph)
    # TODO!
end

"""
    index(g::Graphs.AbstractSimpleGraph, d::Integer)

Calculate the index of the graph `g` in `d`-space. It es equal to `ne(g) - d * nv(g) + binomial(d + 1, 2)`.

# Examples
```jldoctest

"""
index(g::Graphs.AbstractSimpleGraph, d::Integer) = ne(g) - d * nv(g) + binomial(d + 1, 2)

"""
    index(f::AbstractEmbeddedGraph)

Calculate the index of the embedded graph `f`.

# Examples
```jldoctest
julia> G = complete_graph(4)
{4, 6} undirected simple Int64 graph

julia> F = Framework(G,2);

julia> index(F)
1
```
"""
index(f::AbstractEmbeddedGraph) = index(graph(f), d(f))

"""
    is_genrigid(g::Graphs.AbstractSimpleGraph, d::Integer = 2)

Return whether the graph `g` is generically rigid in `d`-space.

# Examples
```jldoctest
julia> G = complete_graph(3);

julia> is_genrigid(G, 2)
true
```

```jldoctest
julia> G = Graph([0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0])
{4, 4} undirected simple Int64 graph

julia> is_genrigid(G, 2)
false
"""
is_genrigid(g::Graphs.AbstractSimpleGraph, d::Integer=2) = d == 2 ? index(g, d) >= 0 : (index(g, d) >= 0 ? is_infrigid(Framework(g, d)) : false)

is_genrigid(f::AbstractEmbeddedGraph, d::Integer=2) = is_genrigid(SimpleGraph(f), d)

is_isostatic(g::Graphs.AbstractSimpleGraph, d::Integer=2) = (index(g, d) == 0)

# TODO: self stresses

# TODO: generic (symbolic) rigidity matrix