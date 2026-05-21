"""
    rigidity_matrix(f::AbstractEmbeddedGraph)

Construct the rigidity matrix ``R(G, p)`` of the embedded graph (or framework) `f`.

The rigidity matrix is an ``|E| \\times d|V|`` matrix.
For each edge ``e = \\{v, w\\} \\in E``, the corresponding row has non-zero block entries at the columns for vertices ``v`` and ``w``.
Specifically, the entry for vertex ``v`` is `p(v) - p(w)` and for vertex ``w`` is `p(w) - p(v)`.

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

Calculate a basis for the space of infinitesimal motions of the framework `f`.

An infinitesimal motion is an assignment of velocities ``v: V \\to \\mathbb{R}^d`` to the vertices such that the length of every edge is infinitesimally preserved, meaning ``\\langle p(v) - p(w), v(v) - v(w) \\rangle = 0`` for all edges ``\\{v, w\\} \\in E``. These correspond precisely to the kernel (nullspace) of the rigidity matrix ``R(G, p)``.
"""
function basis_inf_motions(f::AbstractEmbeddedGraph)
    return nullspace(rigidity_matrix(f))
end

"""
    is_infrigid(f::AbstractEmbeddedGraph)

Return whether the framework `f` is infinitesimally rigid.

A framework in ``\\mathbb{R}^d`` is infinitesimally rigid if its space of infinitesimal motions only contains the trivial motions (translations and rotations). This is equivalent to the kernel of the rigidity matrix having dimension ``\\binom{d+1}{2}`` (for ``|V| \\ge d``).
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

Calculate the index of the graph `g` in `d`-space. It is equal to ``|E| - d |V| + \\binom{d + 1}{2}``.

The index relates to the number of generic redundancies or degrees of freedom in the graph when considered as an isostatic framework.


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
is_genrigid(g::Graphs.AbstractSimpleGraph, d::Integer=2)

Return whether the graph `g` is generically rigid in `d`-space.

A generic realization of a graph `g` is rigid if and only if it satisfies the rigidity criteria. For ``d = 2``, this is related to Laman's theorem, checked by the sparse condition index. For general ``d``, this checks an arbitrary generic realization.

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
```
"""
is_genrigid(g::Graphs.AbstractSimpleGraph, d::Integer=2) = d == 2 ? index(g, d) >= 0 : (index(g, d) >= 0 ? is_infrigid(Framework(g, d)) : false)

"""
    is_genrigid(f::AbstractEmbeddedGraph, d::Integer=2)

Return whether the graph associated with framework `f` is generically rigid in `d`-space.
"""
is_genrigid(f::AbstractEmbeddedGraph, d::Integer=2) = is_genrigid(SimpleGraph(f), d)

"""
    is_isostatic(g::Graphs.AbstractSimpleGraph, d::Integer=2)

Return whether the graph `g` is isostatic (minimally generically rigid) in `d`-space. This happens when the index is exactly zero.
"""

# TODO: self stresses

# TODO: generic (symbolic) rigidity matrix