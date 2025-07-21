function BracketAlgebras.BracketAlgebra(g::Graphs.AbstractSimpleGraph, d::Integer=2; point_labels::Vector=collect(1:nv(g)))
    return BracketAlgebras.BracketAlgebra(nv(g), d; point_labels=point_labels)
end

"""
    tiedown(g::Graphs.AbstractSimpleGraph, d::Integer; tiedown_verts::Union{Nothing,AbstractVector{<:Integer}})

Preprocessing step to calculate pure condition of `g`. Transform `g` into a directed graph and add standard tiedown bars directed from vertices of the graph
to tiedown verts. If `tiedown_verts` are given, 1 bar is added to the `d`-th vertex, 2 to the `(d-1)`-st, ... `d` to the 1st. If `tiedown_verts` is `nothing`, the first `d` vertices are chosen.

A directed graph is returned. This is because every edge from a vertex `v` to a vertex `w` corresponds to a non zero `1 × d` submatrix of the rigidity matrix (exactly the submatrix indexed by the edge `{v,w}` and the vertex `v`).
This is, why we only add an edge  from a vertex v ̲~towards~ a tiedown vertex w as they add a 1 × d submatrix indexed with the edge {v,w} and the vertex v."""
function tiedown(g::Graphs.AbstractSimpleGraph, d::Integer=2; tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    @assert Graphs.nv(g) >= d "Number of vertices needs to be larger than dimension, but got $(Graphs.nv(g)) vertices and dimension $d."
    if isnothing(tiedown_verts)
        tiedown_verts = collect(1:d)
    else
        @assert all(tiedown_verts .<= Graphs.nv(g)) "tiedown_verts need to consist of verts of g (1..$(nv(g)))"
        @assert all(tiedown_verts .>= 1) "tiedown_verts need to consist of verts of g (1..$(Graphs.nv(g)))"
        @assert length(tiedown_verts) == d "tiedown_verts needs to be vector of length d ($d), but got $(length(tiedown_verts))."
    end

    g_di = Graphs.SimpleDiGraph(g)

    # add tiedown bars as directed edges to graph
    for i in 1:d
        for j in 1:(d+1)-i
            Graphs.add_vertex!(g_di)
            Graphs.add_edge!(g_di, tiedown_verts[i], Graphs.nv(g_di))
        end
    end

    return g_di
end

# reduce the graph g by removing all edges in edges and their reverse edges and removing all outward edges from v
# this corresponds to deleting the rows of the rigidity matrix indexed by edges and the columns indexed by v as during Laplace expansion.
function reduction!(g::Graphs.SimpleDiGraph, v::Integer, edges::Vector{<:Graphs.AbstractEdge}, d::Integer)
    @assert all(map(e -> Graphs.has_edge(g, Graphs.src(e), Graphs.dst(e)), edges)) "edges needs to be subset of edges of g."
    @assert length(edges) == d "Number of edges needs to be equal to dimension, but got $(length(edges)) edges and dimension $d."
    @assert all(Graphs.src.(edges) .== v) "All edges need to start at vertex $v, but edge sources are $(src.(edges))."

    remove = union(edges, reverse.(edges), [Graphs.Edge(v, w) for w in Graphs.outneighbors(g, v)])
    for e in remove
        Graphs.rem_edge!(g, e)
    end

    return g
end

function reduction(g::Graphs.SimpleDiGraph, v::Integer, edges::Vector{<:Graphs.AbstractEdge}, d::Integer)
    return reduction!(deepcopy(g), v, edges, d)
end

# g represents a submatrix of the rigidity matrix of some original graph with vertices 1:nv(g) : 
# - If a vertex v in g still has outedges, this means that the columns corresponding to v have not been eliminated via Laplace expansion in a previous step.
# - If an edge (v,w) in g still exists, this means that the row corresponding to {v,w} has not been eliminated via Laplace expansion in a previous step.
# This function computes the sign of the bracket expression that occurs when deleting the rows indexed by edges and the columns indexed by v
# from the submatrix of the original rigidity matrix represented by the graph g.
function Base.sign(g::Graphs.SimpleDiGraph, v::Integer, edges::Vector{<:Graphs.AbstractEdge}, d::Integer)
    @assert all(map(e -> Graphs.has_edge(g, Graphs.src(e), Graphs.dst(e)), edges)) "edges needs to be subset of edges of g."
    @assert length(edges) == d "Number of edges needs to be equal to dimension, but got $(length(e)) edges and dimension $d."
    @assert all(Graphs.src.(edges) .== v) "All edges need to start at vertex $v, but edge sources are $(src.(e))."

    # vertices that still have columns in the rigidity matrix are those that have outgoing edges
    relevant_verts = filter(v -> length(Graphs.outneighbors(g, v)) > 0, Graphs.vertices(g))
    i = indexin(v, relevant_verts)[1]
    # indices of the columns indexed by v in the remaining rigidity matrix
    col_indices = d*(i-1)+1:d*(i-1)+d

    undirected_edges = unique(filter(e -> Graphs.src(e) < Graphs.dst(e), union(Graphs.edges(g), reverse.(Graphs.edges(g)))))

    # indices of the rows indexed by edges in the remaining rigidity matrix. Wlog we assume that the rows of the rigidity matrix are sorted lexicographically.
    sort!(undirected_edges, by=e -> (Graphs.src(e), Graphs.dst(e)))
    row_indices = indexin(map(e -> (Graphs.src(e) > Graphs.dst(e)) ? Graphs.reverse(e) : e, edges), undirected_edges)

    return (-1)^(sum(row_indices) + sum(col_indices))
end

# recursively calculate the condition for the directed graph g to be infinitesimally flexible as an expression in the bracket algebra B.
# This is done by iterating Laplace expansion of the rigidity matrix of g by expanding aling the columns corresponding to the vertices.
# The expansion can be read off the graph without actually constructing the rigidity matrix.
# For reference see https://omni.wikiwand.com/en/articles/Laplace_expansion#General_statement 
function condition(g::Graphs.SimpleDiGraph, B::BracketAlgebra; remaining_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    d = B.d

    if isnothing(remaining_verts)
        remaining_verts = filter(v -> Graphs.outdegree(g, v)[1] > 0, 1:Graphs.nv(g))
    end

    if length(remaining_verts) == 0
        return one(B)
    end

    # if any vertex v has outdegree less than d, every d×d submatrix containing the columns corresponding to v has determinant zero.
    if any(map(v -> length(Graphs.outneighbors(g, v)) < d, remaining_verts))
        return zero(B)
    end

    # select the vertex with smallest defect: d - (outdegree - indegree).
    (defect, i) = findmin(v -> d - (length(Graphs.outneighbors(g, v)) - length(Graphs.inneighbors(g, v))), remaining_verts)
    v = remaining_verts[i]

    if defect == 0
        # if defect == 0 that means the vertex with minimum defect satisfies (outdegree - indegree) = d. 
        # This means the rigidity matrix can be rearranged as a upper left triangular block matrix with a d×d block in the upper left corner. 
        # Thus, we get the determinant of the matrix by multiplying the determinant of this block with the determinant of the matrix after deleting rows and columns of the block.
        # The determinant of the block is a bracket expression with v and d outwards neighbors of v that are not inwards neighbors of v.

        # edges from v that don't have a reverse edge 
        edges = setdiff([Graphs.Edge(v, w) for w in Graphs.outneighbors(g, v)], [Graphs.Edge(v, w) for w in Graphs.inneighbors(g, v)])[1:d]

        # recursive call. The determinant of the rigiditymatrix is via Laplace: ± [v, e1_2, …, ed_2]  * (determinant of matrix after deleting rows corresponding to edges and columns corresponding to v)
        return sign(g, v, edges, d) * B(pushfirst!([Graphs.dst(e) for e in edges], v)) * condition(reduction(g, v, edges, d), B; remaining_verts=setdiff(remaining_verts, [v]))
    elseif defect > 0
        # If defect > 0 the determinant has to be calculated using Laplace expansion that involve more than one nonzero summand. 
        # See Wikipedia article for Laplace expansion by multiple columns.

        edges_onlyout = setdiff([Graphs.Edge(v, w) for w in Graphs.outneighbors(g, v)], [Graphs.Edge(v, w) for w in Graphs.inneighbors(g, v)])
        edges_both = [Graphs.Edge(v, w) for w in intersect(Graphs.outneighbors(g, v), Graphs.inneighbors(g, v))]

        # recursive call for Laplace expansion
        sum_index = map(edges -> union(edges, edges_onlyout), combinations(edges_both, defect))
        return sum(edges -> sign(g, v, edges, d) * B(pushfirst!([Graphs.dst(e) for e in edges], v)) * condition(reduction(g, v, edges, d), B; remaining_verts=setdiff(remaining_verts, [v])), sum_index)
    elseif defect < 0
        # If defect < 0 the determinant is zero as after Laplace expansion along the columns of v and d outedges of v, the matrix has a zero row.
        return zero(B)
    end
end

function condition(g::Graphs.SimpleDiGraph, d::Integer=2; B::BracketAlgebra=BracketAlgebra(g, d))
    if B.d != d
        throw(ArgumentError("BracketAlgebra B and d don't match: B.d=$(B.d), d=$d."))
    end
    if B.n != Graphs.nv(g)
        throw(ArgumentError("BracketAlgebra and graph don't match: B.n=$(B.n), nv(g)=$(Graphs.nv(g))."))
    end

    return condition(g, B)
end

"""
    condition(g::Graphs.AbstractSimpleGraph, d::Integer=2; tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing, vertex_labels::Vector=collect(1:nv(g)), tiedown_label_prefix::String="tiedown")

Calculate the condition for the graph `g` to be infinitesimally flexible in P^`d` as an element of its bracket algebra with tiedown.
# Arguments:
- g::Graphs.AbstractSimpleGraph                                     Graph of which condition is computed
- d::Integer=2                                                      Dimension of underlying projective space
- tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing   Vertex indices of vertices to tie down.
- vertex_labels::Vector=collect(1:nv(g))                            Labels for vertices in bracket algebra expression.
- tiedown_label_prefix::String="tiedown"                            Prefix for tiedown vertex label in bracket algebra expression.

# Examples
```jldoctest
julia> m = [0 1 1 1 1;
       1 0 1 1 1;
       1 1 0 1 1;
       1 1 1 0 0;
       1 1 1 0 0;];

julia> g = Graph(m);

julia> condition(g, 3)
Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 2, "tiedown_4", "tiedown_5"]Any[1, 2, 3, "tiedown_6"]Any[1, 2, 3, 5]Any[1, 2, 3, 4]

julia> condition(g, 3, tiedown_verts=[1,4,5])
Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 2, "tiedown_4", "tiedown_5"]Any[1, 2, 3, "tiedown_6"]Any[1, 2, 3, 5]Any[1, 2, 3, 4]

julia> condition(g, 3, tiedown_verts=[1,4,5])
Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 4, "tiedown_4", "tiedown_5"]Any[1, 3, 4, 5]Any[1, 2, 5, "tiedown_6"]Any[1, 2, 3, 4] + Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 4, "tiedown_4", "tiedown_5"]Any[1, 3, 5, "tiedown_6"]Any[1, 2, 4, 5]Any[1, 2, 3, 4]
```
"""
function condition(g::Graphs.AbstractSimpleGraph, d::Integer=2; tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing, vertex_labels::Vector=collect(1:nv(g)), tiedown_label_prefix::String="tiedown")
    if isnothing(tiedown_verts)
        tiedown_verts = collect(1:d)
    end

    g_tiedown = tiedown(g, d; tiedown_verts=tiedown_verts)

    # Bracket Algebra of which condition is an elment of. It has nv(g) + d(d+1)/2 points. Dimension of underlying space is d. New tiedown vertex labels are tiedown_label_prefix_id.
    labels = vertex_labels
    tiedown_labels = [tiedown_label_prefix * "_$i" for i in collect(1:Integer(d * (d + 1) / 2))]
    labels = vcat(labels, tiedown_labels)
    B = BracketAlgebra(g_tiedown, d, point_labels=labels)

    return condition(tiedown(g, d, tiedown_verts=tiedown_verts), d; B=B)
end

"""
    pure_condition(g::Graphs.AbstractSimpleGraph, d::Integer=2; tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing, vertex_labels::Vector=collect(1:nv(g)))

Calculate the pure condition for the graph `g` to be infinitesimally flexible in P^`d` as an element of its bracket algebra with tiedown.
# Arguments:
- g::Graphs.AbstractSimpleGraph                                     Graph of which condition is computed
- d::Integer=2                                                      Dimension of underlying projective space
- tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing   Vertex indices of vertices to tie down. Has to form an isostatic subgraph
- vertex_labels::Vector=collect(1:nv(g))                            Labels for vertices in bracket algebra expression.
- tiedown_label_prefix::String="tiedown"                            Prefix for tiedown vertex label in bracket algebra expression.

# Examples
```jldoctest
julia> m = [0 1 1 1 1;
       1 0 1 1 1;
       1 1 0 1 1;
       1 1 1 0 0;
       1 1 1 0 0;];

julia> g = Graph(m);

julia> condition(g, 3)
Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 2, "tiedown_4", "tiedown_5"]Any[1, 2, 3, "tiedown_6"]Any[1, 2, 3, 5]Any[1, 2, 3, 4]

julia> condition(g, 3, tiedown_verts=[1,4,5])
Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 2, "tiedown_4", "tiedown_5"]Any[1, 2, 3, "tiedown_6"]Any[1, 2, 3, 5]Any[1, 2, 3, 4]

julia> condition(g, 3, tiedown_verts=[1,4,5])
Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 4, "tiedown_4", "tiedown_5"]Any[1, 3, 4, 5]Any[1, 2, 5, "tiedown_6"]Any[1, 2, 3, 4] + Any[1, "tiedown_1", "tiedown_2", "tiedown_3"]Any[1, 4, "tiedown_4", "tiedown_5"]Any[1, 3, 5, "tiedown_6"]Any[1, 2, 4, 5]Any[1, 2, 3, 4]
```
"""
function pure_condition(g::Graphs.AbstractSimpleGraph, d::Integer=2; tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing, vertex_labels::Vector=collect(1:nv(g)))
    if isnothing(tiedown_verts)
        tiedown_verts = collect(1:d)
    end

    if !is_isostatic(g[tiedown_verts], d)
        error("Subgraph spanned by tiedown vertices has to be $d-isostatic, but it has $(Graphs.nv(g[tiedown_verts])) vertices and $(Graphs.ne(g[tiedown_verts])) edges and thus index $(index(g[tiedown_verts], d)).")
    end

    g_tiedown = tiedown(g, d; tiedown_verts=tiedown_verts)

    # perform reduction of tiedown verts: first vertex has d new edges attached to it, second has d-1, and so on.
    v = tiedown_verts[1]
    onlyout_neis = setdiff(Graphs.outneighbors(g_tiedown, v), Graphs.inneighbors(g_tiedown, v))
    edges = [Graphs.Edge(v, w) for w in onlyout_neis]
    reduction!(g_tiedown, v, edges, d)

    for (i, v) in enumerate(tiedown_verts[2:end])
        onlyout_neis = setdiff(Graphs.outneighbors(g_tiedown, v), Graphs.inneighbors(g_tiedown, v))
        edges = union([Graphs.Edge(v, w) for w in onlyout_neis], [Graphs.Edge(v, w) for w in tiedown_verts[1:i]])
        reduction!(g_tiedown, v, edges, d)
    end

    remaining_verts = setdiff(1:Graphs.nv(g), tiedown_verts)

    return condition(g_tiedown, BracketAlgebra(g, d, point_labels=vertex_labels); remaining_verts=remaining_verts)
end

function pure_condition(f::Framework; tiedown_verts::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    g = graph(f)
    return pure_condition(g, d(f), tiedown_verts=tiedown_verts, vertex_labels=labels(f))
end