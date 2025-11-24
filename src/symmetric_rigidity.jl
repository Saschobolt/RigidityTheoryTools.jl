using Graphs
using Symbolics
import Oscar: GAPGroup, GroupElem, orbits
import AbstractAlgebra: Group
import Oscar
import GAP

# Import all the functions we need to extend from Graphs.jl
import Graphs: AbstractGraph, edgetype, nv, ne, vertices, edges,
    src, dst, has_edge, has_vertex, inneighbors, outneighbors,
    is_directed


#################################################################
#
# Quotient Gain Graph Structure
#
#################################################################
"""
An edge type for the Quotient Gain Graph.
It represents an edge `((i, j); γ)` as defined in the paper,
where `i` and `j` are indices of representative vertices and
`γ` is the group label (gain).
"""
struct QuotientGainEdge{G<:GroupElem,VertexType<:Integer} <: AbstractEdge{VertexType}
    src::VertexType  # Index of the source representative vertex (1 to nv_0)
    dst::VertexType  # Index of the destination representative vertex (1 to nv_0)
    gain::G   # The group element γ
end

# Implement basic Graphs.jl functions for our custom edge
Graphs.src(e::QuotientGainEdge) = e.src
Graphs.dst(e::QuotientGainEdge) = e.dst

function Base.show(io::IO, e::QuotientGainEdge)
    # Try to print a nice string for the gain, check if `one` is defined
    gain_str = try
        (e.gain == one(parent(e.gain))) ? "id" : string(e.gain)
    catch
        string(e.gain) # Fallback if `one` or `parent` fails
    end
    print(io, "QuotientGainEdge (($(src(e)), $(dst(e))); $(gain_str))")
end

"""
A struct representing a Quotient Γ-Gain Graph.

This is a directed multigraph where vertices are representative
vertices from the orbits of a larger graph, and edges
are representative edges with an associated group "gain".

- `G` is the type of the group elements (e.g., `Perm{Int}` from Oscar.jl).
- `V_orig` is the integer type of the vertices in the *original* graph.
"""
mutable struct QuotientGainGraph{G<:GroupElem,VertexType<:Integer} <: AbstractGraph{VertexType}
    # The symmetry group Γ
    group::Group

    # --- Vertex mapping ---
    # v_reps: Maps quotient vertex index (1:nv) -> original graph vertex
    v_reps::Vector{VertexType}
    # v_map: Maps original graph vertex -> quotient vertex index (1:nv)
    v_map::Dict{VertexType,Int}

    # --- Edge storage (Adjacency Lists) ---
    # We use adjacency lists for efficient neighbor lookups,
    # which is required by the AbstractGraph interface.

    # out_gains[i] = Vector of (neighbor_j, gain_γ) for all edges i -> j
    out_gains::Vector{Vector{Tuple{Int,G}}}

    # in_gains[i] = Vector of (neighbor_j, gain_γ) for all edges j -> i
    in_gains::Vector{Vector{Tuple{Int,G}}}

    num_edges::Int
end

function Base.show(io::IO, g::QuotientGainGraph)
    print(io, "QuotientGainGraph ($(nv(g)) vertices, $(ne(g)) edges) with gains in $(g.group)")
end

# --- Constructor Functions ---
# constructor for an empty QuotientGainGraph given group and vertex representatives
function QuotientGainGraph(
    group::Group,
    v_representatives::Vector{V_orig}
) where {V_orig<:Integer}

    G = eltype(group)
    nv_0 = length(v_representatives)

    # Create the vertex mappings
    v_reps = v_representatives
    v_map = Dict{V_orig,Int}(rep => i for (i, rep) in enumerate(v_reps))

    # Initialize empty adjacency lists
    out_gains = [Vector{Tuple{Int,G}}() for _ in 1:nv_0]
    in_gains = [Vector{Tuple{Int,G}}() for _ in 1:nv_0]

    return QuotientGainGraph{G,V_orig}(
        group, v_reps, v_map, out_gains, in_gains, 0
    )
end

# constructor from group and vertex orbits
function QuotientGainGraph(
    group::Group,
    v_orbits::Vector{Vector{V_orig}}
) where {V_orig<:Integer}

    G_El = eltype(group)

    # Representatives are the first (canonical) element of each orbit
    v_reps = [first(orbit) for orbit in v_orbits]
    nv_0 = length(v_reps)

    # Create the vertex mappings
    # v_map: Maps *any* original vertex -> quotient vertex index (1:nv_0)
    v_map = Dict{V_orig,Int}()
    for (i, orbit) in enumerate(v_orbits)
        # `i` is the index for this orbit (1, 2, ...)
        for v_orig in orbit
            v_map[v_orig] = i
        end
    end

    # Initialize empty adjacency lists
    out_gains = [Vector{Tuple{Int,G_El}}() for _ in 1:nv_0]
    in_gains = [Vector{Tuple{Int,G_El}}() for _ in 1:nv_0]

    return QuotientGainGraph{G_El,V_orig}(
        group, v_reps, v_map, out_gains, in_gains, 0
    )
end

# --- AbstractGraph Interface Implementation ---

# The vertex type is Int (indices 1 to nv_0)
Base.eltype(g::QuotientGainGraph{G,V}) where {G,V} = V

# The edge type is our custom struct
Graphs.edgetype(g::QuotientGainGraph{G,V}) where {G,V} = QuotientGainEdge{G,V}

# Number of vertices is the number of representatives 
Graphs.nv(g::QuotientGainGraph) = length(g.v_reps)

# Number of edges is the number of representative edges 
Graphs.ne(g::QuotientGainGraph) = g.num_edges

# Vertices are the indices from 1 to nv(g)
Graphs.vertices(g::QuotientGainGraph) = 1:nv(g)

# The graph is directed 
Graphs.is_directed(::Type{<:QuotientGainGraph}) = true

Graphs.has_vertex(g::QuotientGainGraph, v::Integer) = 1 <= v <= nv(g)

"""
has_edge(g, s, d)

Returns `true` if *at least one* edge exists from `s` to `d`.
Note: This is lossy. It does not tell you *how many* edges
or *what their gains* are.
"""
function Graphs.has_edge(g::QuotientGainGraph, s::Int, d::Int)
    if !has_vertex(g, s) || !has_vertex(g, d)
        return false
    end
    # Check the out-list of s for d
    for (neighbor, gain) in g.out_gains[s]
        if neighbor == d
            return true
        end
    end
    return false
end

"""
edges(g)

Returns an iterator of all edges in the graph as `QuotientGainEdge` objects.
"""
function Graphs.edges(g::QuotientGainGraph)
    # Use a generator for an efficient iterator
    return (QuotientGainEdge(s, d, gain)
            for s in 1:nv(g)
            for (d, gain) in g.out_gains[s])
end

"""
outneighbors(g, v)

Returns a list of unique vertices `j` such that an edge `(v, j)` exists.
"""
function Graphs.outneighbors(g::QuotientGainGraph, v::Int)
    if !has_vertex(g, v)
        return Int[]
    end
    # Map (neighbor, gain) -> neighbor, and get unique values
    return unique(first(pair) for pair in g.out_gains[v])
end

"""
inneighbors(g, v)

Returns a list of unique vertices `j` such that an edge `(j, v)` exists.
"""
function Graphs.inneighbors(g::QuotientGainGraph, v::Int)
    if !has_vertex(g, v)
        return Int[]
    end
    return unique(first(pair) for pair in g.in_gains[v])
end

"""
add_gain_edge!(g, s, d, gain)

Adds a new directed edge from representative vertex `s` to
representative vertex `d` with a specific `gain`.
"""
function add_gain_edge!(
    g::QuotientGainGraph{G,V},
    s::Int,
    d::Int,
    gain::G
) where {G,V}

    if !has_vertex(g, s) || !has_vertex(g, d)
        error("Attempted to add edge to non-existent vertex.")
    end

    push!(g.out_gains[s], (d, gain))
    push!(g.in_gains[d], (s, gain))
    g.num_edges += 1
    return true
end

"""
edge_gains(g, s, d)

Returns a `Vector` of all gains for edges from `s` to `d`.
"""
function edge_gains(g::QuotientGainGraph, s::Int, d::Int)
    gains = Vector{eltype(g.group)}()
    if has_vertex(g, s) && has_vertex(g, d)
        for (neighbor, gain) in g.out_gains[s]
            if neighbor == d
                push!(gains, gain)
            end
        end
    end
    return gains
end

"""
original_vertex(g, v_quotient)

Returns the vertex label in the *original* graph corresponding
to the quotient graph's vertex index `v_quotient`.
"""
original_vertex(g::QuotientGainGraph, v_quotient::Int) = g.v_reps[v_quotient]

"""
quotient_vertex(g, v_original)

Returns the quotient graph's vertex index (1:nv) corresponding
to the *original* graph's vertex label `v_original`.
"""
quotient_vertex(g::QuotientGainGraph, v_original) = g.v_map[v_original]

"""
Returns a list of `(neighbor_idx, gain)` tuples for edges
leaving vertex `s`.
"""
out_edges_with_gains(g::QuotientGainGraph, s::Int) = [QuotientGainEdge(s, dst, group_elem) for (dst, group_elem) in g.out_gains[s]]

# --- Construct Quotient Graph from simple graph and group ---

"""
    on_edges(edge::AbstractEdge, g::GroupElem) -> AbstractEdge

Applies a group element `g` (like a `Perm`) to an `AbstractEdge` by
acting on its source and destination vertices.

Returns a new edge of the same type as the input.
"""
function on_edges(edge::AbstractEdge, g::GroupElem)
    # Get the original source and destination
    s = src(edge)
    d = dst(edge)

    # Apply the group action to the vertices
    res_action = Oscar.on_sets([s, d], g)
    s_new = res_action[1]
    d_new = res_action[2]

    # Get the concrete type of the edge (e.g., SimpleEdge{Int})
    EdgeType = typeof(edge)

    # Construct and return a new edge of the same type
    # This relies on the edge type having a (src, dst) constructor
    return EdgeType(s_new, d_new)
end


"""
    quotient_graph(g::AbstractGraph{V}, G::Group) where {V<:Integer}

Compute the quotient graph of the graph `g` under the action of the group `G`.
This is the graph with vertices corresponding to the orbits of the vertices of `g` under the action of `G`, and edges corresponding to the orbits of the edges of `g`, with gains given by the group elements mapping representative vertices.

```jldoctest
julia> g = SimpleGraph(6);

julia> add_edge!(g, 1, 2);

julia> add_edge!(g, 2, 3);

julia> add_edge!(g, 1, 3);

julia> add_edge!(g, 1, 4);

julia> add_edge!(g, 2, 5);

julia> add_edge!(g, 3, 6);

julia> add_edge!(g, 4, 5);

julia> add_edge!(g, 5, 6);

julia> add_edge!(g, 4, 6);

julia> G = Oscar.symmetric_group(6);

julia> H, _ = Oscar.sub(G, [Oscar.perm([2,3,1,5,6,4])]);

julia> qg = quotient_graph(g, H)
QuotientGainGraph (2 vertices, 3 edges) with gains in Permutation group of degree 6
```
"""
function quotient_graph(g::AbstractGraph{V}, G::Group) where {V<:Integer}
    # 1. Compute Vertex Orbits and create the empty quotient graph
    v_set = collect(vertices(g))
    v_gset = Oscar.gset(G, v_set)
    v_orbits = Oscar.orbits(v_gset)

    qg = QuotientGainGraph(G, collect.(v_orbits))

    # 2. Compute Edge Orbits
    e_set = collect(edges(g))
    e_gset = Oscar.gset(G, on_edges, e_set)
    e_orbits = Oscar.orbits(e_gset)

    # 3. Map representative edges to quotient edges
    for e_orbit in e_orbits
        # Get the canonical representative edge for this orbit
        e_rep = first(e_orbit)
        u_orig = src(e_rep)
        v_orig = dst(e_rep)

        # Find the quotient index of the source vertex `u`
        i_idx = quotient_vertex(qg, u_orig)
        # Get the *representative vertex* for that index
        i_rep = original_vertex(qg, i_idx)

        # To get the canonical edge form `(i, γ(j))` from the paper,
        # we first find the group element `g` that maps `i_rep` to `u_orig`.
        # Then `g_inv` maps `u_orig` back to its representative `i_rep`.
        g = Oscar.is_conjugate_with_data(v_gset, i_rep, u_orig)[2]
        g_inv = inv(g)

        # Apply `g_inv` to the *entire* edge to get its canonical form,
        # which is guaranteed to start at the representative `i_rep`.
        e_canonical = on_edges(e_rep, g_inv)

        # We know src(e_canonical) == i_rep. We need the destination.
        v_canonical = dst(e_canonical)

        # Find the quotient information for this canonical destination
        j_idx = quotient_vertex(qg, v_canonical)
        j_rep = original_vertex(qg, j_idx)

        # The "gain" (γ) is the group element that maps the
        # representative `j_rep` to the canonical destination `v_canonical`.
        gamma = Oscar.is_conjugate_with_data(v_gset, j_rep, v_canonical)[2]

        # Add this directed, gained edge to the quotient graph
        add_gain_edge!(qg, i_idx, j_idx, gamma)
    end

    return qg
end

#################################################################
#
# Orbit Rigidity Matrix
#
#################################################################

"""
    orbit_rigidity_matrix(qg::QuotientGainGraph, p_reps::AbstractMatrix{T}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where T

Computes the Orbit Rigidity Matrix (Definition 8) for a
quotient gain graph `qg` at a specific configuration `p_reps`.

# Arguments
- `qg::QuotientGainGraph`: The quotient graph.
- `p_reps::AbstractMatrix{T}`: A `d x nv_0` matrix where `d` is the dimension and `nv_0` is the number of representative vertices.
  `p_reps[:, i]` is the coordinate vector for the i-th representative.
- `phi::GroupHomomorphism`: An Oscar group homomorphism from `qg.group` to an orthogonal matrix group (e.g., `O(d, R)`).
"""
function orbit_rigidity_matrix(qg::QuotientGainGraph, p_reps::AbstractMatrix{T}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where T

    G = qg.group
    H = Oscar.codomain(phi)

    # --- 1. Validate Inputs ---
    d = size(p_reps, 1) # Spatial dimension

    # Check p_reps dimensions
    if size(p_reps) != (d, nv(qg))
        error("Position matrix p_reps has wrong size. Expected ($(d), $(nv(qg))), got $(size(p_reps)).")
    end

    # Check homomorphism domain
    if Oscar.domain(phi) != G
        error("Homomorphism `phi` must have `qg.group` as its domain.")
    end

    # Check homomorphism codomain (matrices) and orthogonality
    if size(one(H)) != (d, d)
        error("Representation codomain has to be a subgroup of O($(d), R), but got matrices of size $(size(one(H))).")
    end

    # Check all generators
    for g in gens(G)
        M = Matrix(T.(matrix(phi(g))))
        if !isapprox(M' * M, I(d); atol=1e-8)
            error("Representation matrix for $(g) is not orthogonal.")
        end
    end


    # --- 2. Initialize and Fill Orbit Matrix O ---
    nv_0 = nv(qg)
    ne_0 = ne(qg)
    O = zeros(T, ne_0, d * nv_0)

    for (row_idx, e) in enumerate(edges(qg))
        i = src(e)
        j = dst(e)
        gamma = e.gain

        # Call the homomorphism to get the matrix
        phi_g = Matrix(T.(matrix(phi(gamma))))
        # Use M' for inverse of orthogonal matrix
        phi_g_inv = phi_g'

        p_i = p_reps[:, i]
        p_j = p_reps[:, j]

        # Get column ranges for vertices i and j
        cols_i = ((i-1)*d+1):(i*d)
        cols_j = ((j-1)*d+1):(j*d)

        if i == j # Loop edge: ((i, i); γ)
            entry = 2 * p_i - phi_g * p_i - phi_g_inv * p_i
            O[row_idx, cols_i] = entry' # Transpose to get row vector
        else     # Non-loop edge: ((i, j); γ)
            entry_i = p_i - phi_g * p_j
            entry_j = p_j - phi_g_inv * p_i

            O[row_idx, cols_i] = entry_i'
            O[row_idx, cols_j] = entry_j'
        end
    end

    return O
end

function orbit_rigidity_matrix(g::SimpleGraph, emb::AbstractMatrix{T}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where T
    # test that (g, emb) is symmetric for phi
    if !is_symmetric(g, emb, phi)
        error("The graph embedding is not symmetric with respect to the given group homomorphism.")
    end

    # construct the quotient gain graph
    G = Oscar.domain(phi)
    qg = quotient_graph(g, G)
    # get vertex orbits to extract representative positions
    v_reps = [original_vertex(qg, v) for v in vertices(qg)]
    # extract representative positions
    p_reps = hcat([emb[:, v_rep] for v_rep in v_reps]...)
    # compute and return the orbit rigidity matrix
    return orbit_rigidity_matrix(qg, p_reps, phi)
end

"""
    orbit_rigidity_matrix(f::Framework, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup})

Return the orbit rigidity matrix of the framework `f` with respect to the orthogonal representation `phi`.
"""
function orbit_rigidity_matrix(f::Framework, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup})
    return orbit_rigidity_matrix(graph(f), coordinate_matrix(f), phi)
end

function basis_sym_inf_motions(f::Framework, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup})
    return nullspace(orbit_rigidity_matrix(f, phi))
end

########################## symbolic orbit rigidity matrix ##########################
"""
    symbolic_orbit_rigidity_matrix(qg::QuotientGainGraph; representation_name=:tau, coordinate_prefix=:p)

Construct the orbit rigidity matrix with symbolic entries for the quotient Γ-gain
graph `qg`. Each column corresponds to a representative vertex and carries one
symbol `p_i`, where `i` is the original vertex chosen as that representative.
The image of each gain `γ` under the orthogonal representation is denoted by
`τ_γ` (or whatever name is given via `representation_name`), so a typical entry
looks like `p_1 - τ_((1,2)(3,4)) * p_4`.
"""
function symbolic_orbit_rigidity_matrix(
    qg::QuotientGainGraph;
    representation_name::Union{Symbol,String}=:τ,
    coordinate_prefix::Union{Symbol,String}=:p,
)
    repr = Symbolics.variable(representation_name; T=Symbolics.FnType)
    points = Symbolics.variables(coordinate_prefix, [original_vertex(qg, v) for v in vertices(qg)])

    nv_0 = nv(qg)
    ne_0 = ne(qg)
    zero_symbol = Num(0)

    tau_cache = Dict{GroupElem,Num}()

    O = fill(zero_symbol, ne_0, nv_0)
    for (row_idx, e) in enumerate(edges(qg))
        i = src(e)
        j = dst(e)
        gamma = e.gain
        gamma_inv = inv(gamma)

        if i == j
            O[row_idx, i] =
                Symbolics.value(Num(2) * points[i] - repr(gamma) * points[i] - repr(gamma_inv) * points[i])
        else
            O[row_idx, i] = points[i] - repr(gamma) * points[j]
            O[row_idx, j] = points[j] - repr(gamma_inv) * points[i]
        end
    end

    return O
end

"""
    symbolic_orbit_rigidity_matrix(g::AbstractGraph, G::Group; representation_name=:tau, coordinate_prefix=:p)

Compute the symbolic orbit rigidity matrix by first forming the quotient gain
graph of `g` with respect to the group action of `G`.
"""
function symbolic_orbit_rigidity_matrix(
    g::AbstractGraph{V},
    G::Group;
    kwargs...,
) where {V<:Integer}
    qg = quotient_graph(g, G)
    return symbolic_orbit_rigidity_matrix(qg; kwargs...)
end

"""
    symbolic_orbit_rigidity_matrix(g::SimpleGraph, phi::Oscar.GAPGroupHomomorphism; representation_name=:tau, coordinate_prefix=:p)

Convenience method that uses the domain of `phi` to build the quotient gain graph
before producing the symbolic orbit rigidity matrix.
"""
function symbolic_orbit_rigidity_matrix(
    g::SimpleGraph,
    phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup};
    kwargs...,
)
    return symbolic_orbit_rigidity_matrix(g, Oscar.domain(phi); kwargs...)
end

"""
    symbolic_orbit_rigidity_matrix(f::Framework, phi::Oscar.GAPGroupHomomorphism; representation_name=:tau, coordinate_prefix=:p)

Return the symbolic orbit rigidity matrix for the framework `f`.
"""
function symbolic_orbit_rigidity_matrix(
    f::Framework,
    phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup};
    kwargs...,
)
    return symbolic_orbit_rigidity_matrix(graph(f), phi; kwargs...)
end



##################################################################
#
# Symmetric realizations of frameworks
#
##################################################################
"""
    symmetric_framework(g::Graphs.AbstractSimpleGraph, p_reps::Dict{<:Integer,Point{EmbeddingDim,EmbeddingType}}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where {EmbeddingDim,EmbeddingType}

Compute a symmetric embedding of the graph `g` with respect to the orthogonal representation `phi` of a permutation group acting on the vertices of `g`, given the positions of representative vertices `p_reps`.`

```
julia> g = SimpleGraph(6);

julia> add_edge!(g, 1, 2);

julia> add_edge!(g, 2, 3);

julia> add_edge!(g, 1, 3);

julia> add_edge!(g, 1, 4);

julia> add_edge!(g, 2, 5);

julia> add_edge!(g, 3, 6);

julia> add_edge!(g, 4, 5);

julia> add_edge!(g, 5, 6);

julia> add_edge!(g, 4, 6);

julia> G = Oscar.symmetric_group(6);

julia> H, _ = Oscar.sub(G, [Oscar.perm([2,3,1,5,6,4])]);

julia> mat_group = RigidityTheoryTools.rotation_group_around_origin_2d(1 // 3); # rotation group around origin generated by rotation of angle 1/3*2π

julia> repr = Oscar.hom(H, mat_group, Oscar.gens(H), Oscar.gens(mat_group)); # representation of H as matrix group

julia> f = symmetric_framework(g, Dict(1 => Point(1., 0.), 4 => Point(0.5, 0.5)), repr)
Framework with 6 vertices and 9 edges, vertex labels [1, 2, 3, 4, 5, 6] and 2-realization
Point{2, Float64}[[1.0, 0.0], [-0.5, 0.8660254037844386], [-0.5, -0.8660254037844386], [0.5, 0.5], [-0.6830127018922193, 0.1830127018922193], [0.1830127018922193, -0.6830127018922193]]
```
"""
function symmetric_framework(g::Graphs.AbstractSimpleGraph, p_reps::Dict{<:Integer,Point{EmbeddingDim,EmbeddingType}}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where {EmbeddingDim,EmbeddingType}
    G = Oscar.domain(phi)
    n = nv(g)

    if !Oscar.is_subgroup(G, Oscar.automorphism_group(oscar_graph(g)))[1]
        error("The group G is not a subgroup of the automorphism group of the graph.")
    end

    if size(one(Oscar.codomain(phi))) != (EmbeddingDim, EmbeddingDim)
        error("Representation codomain has to be a subgroup of O($(EmbeddingDim), R), but got matrices of size $(size(one(Oscar.codomain(phi)))).")
    end

    if !is_orthogonal(phi)
        error("The representation phi is not orthogonal.")
    end

    # 1. Compute vertex orbits
    v_set = collect(vertices(g))
    v_gset = Oscar.gset(G, v_set)
    v_orbits = collect.(Oscar.orbits(v_gset))

    # 2. Build full embedding from representatives
    emb = zeros(EmbeddingType, EmbeddingDim, n)
    for orbit in v_orbits
        v_rep = first(orbit)
        p_rep = p_reps[v_rep]
        for v in orbit
            g_elem = Oscar.is_conjugate_with_data(v_gset, v_rep, v)[2]
            M = julia_matrix(EmbeddingType, phi(g_elem))
            emb[:, v] = M * collect(p_rep)
        end
    end

    return Framework(g, emb)
end

function symmetric_framework(g::Graphs.AbstractSimpleGraph, p_reps::Dict{<:Integer,<:AbstractVector{EmbeddingType}}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where {EmbeddingType}
    EmbeddingDim = length(first(values(p_reps)))
    return symmetric_framework(g, Dict(k => Point{EmbeddingDim,EmbeddingType}(v) for (k, v) in p_reps), phi)
end

##################################################################
#
# Auxiliary Functions for Symmetric Frameworks
#
##################################################################
"""
    rotation_group_around_origin_2d(θ::Union{Rational, Oscar.QQFieldElem})

Return the 2d rotation group around the origin generated by a rotation of angle `θ`*2π.
"""
function rotation_group_around_origin_2d(θ::Union{Rational,Oscar.QQFieldElem})
    F = Oscar.algebraic_closure(Oscar.QQ)
    s = sinpi(one(F) * 2 * θ)
    c = cospi(one(F) * 2 * θ)
    G = Oscar.matrix_group([matrix(F, [c -s; s c])])
    return G
end

oscar_graph(g::Graphs.SimpleGraph) = Oscar.graph_from_adjacency_matrix(Oscar.Undirected, Matrix(Graphs.adjacency_matrix(g)))

julia_matrix(::Type{T}, g::Oscar.MatrixGroupElem) where T = Matrix(T.(matrix(g)))

"""
    is_orthogonal(phi::Oscar.GAPGroupHomomorphism{<:Group, <:Oscar.MatrixGroup})

Return whether the image of the group homomorphism `phi` is a subgroup of an orthogonal matrix group, thus making `phi` an orthogonal representation.
"""
function is_orthogonal(phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup})
    G = Oscar.domain(phi)
    d = size(Oscar.matrix(phi(one(G))))[1]
    for g in gens(G)
        M = julia_matrix(Float64, phi(g))
        if size(M) != (d, d) || !isapprox(M' * M, I(d); atol=1e-8)
            return false
        end
    end
    return true
end

# check that embedding of a graph is symmetric wrt the representation phi
function is_symmetric(g::SimpleGraph, emb::AbstractMatrix{T}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}; atol::Real=1e-8) where T
    G = Oscar.domain(phi)
    n = nv(g)
    embedding_dim = size(emb, 1)

    # 0. Check embedding dimensions
    if size(emb, 2) != n
        return error("Embedding matrix has wrong number of columns. Expected $(n), got $(size(emb, 2)).")
    end
    # Check that the representation codomain matrices have correct size
    H = Oscar.codomain(phi)
    if size(one(H)) != (embedding_dim, embedding_dim)
        return error("Representation codomain has to be a subgroup of O($(embedding_dim), R), but got matrices of size $(size(one(H))).")
    end

    # 1. Check that G is a subgroup of Aut(g)
    g_oscar = oscar_graph(g)
    aut = Oscar.automorphism_group(g_oscar)
    if !Oscar.is_subgroup(G, aut)[1]
        display("The group G is not a subgroup of the automorphism group of the graph.")
        return false
    end

    # 2. Check that H is an orthogonal matrix group
    if !is_orthogonal(phi)
        display("The representation phi is not orthogonal.")
        return false
    end

    # 3. Check the symmetry condition for all generators
    for g in gens(G)
        if !isapprox(emb[:, collect(1:n)^g], julia_matrix(T, phi(g)) * emb; atol=atol)
            display("The embedding is not symmetric with respect to generator $(g). We have emb[:, ...] = $(emb[:, collect(1:n)^g]) but expected $(julia_matrix(T, phi(g)) * emb).")
            return false
        end
    end

    return true
end

"""
    is_symmetric(f::Framework{PositionDim,PositionType}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where {PositionDim,PositionType}

Return whether the framework `f` is `phi`-symmetric, i.e. with `g = graph(f)` and `emb = coordinate_matrix(f)` we have
- the domain of `phi` is a subgroup of the automorphism group of `g`
- the codomain of `phi` is an orthogonal matrix group
- for all generators `g` of the codomain of `phi` we have `isapprox(emb[:, collect(1:n)^g], julia_matrix(T, g) * emb; atol=atol)``

```jldoctest
julia> g = SimpleGraph(6);

julia> add_edge!(g, 1, 2);

julia> add_edge!(g, 2, 3);

julia> add_edge!(g, 1, 3);

julia> add_edge!(g, 1, 4);

julia> add_edge!(g, 2, 5);

julia> add_edge!(g, 3, 6);

julia> add_edge!(g, 4, 5);

julia> add_edge!(g, 5, 6);

julia> add_edge!(g, 4, 6);

julia> G = Oscar.symmetric_group(6);

julia> H, _ = Oscar.sub(G, [Oscar.perm([2,3,1,5,6,4])]);

julia> p = [
                       cosd(120) sind(120);
                       cosd(240) sind(240);
                       cosd(0) sind(0);
                       1.5*cosd(120) 1.5*sind(120);
                       1.5*cosd(240) 1.5*sind(240);
                       1.5*cosd(0) 1.5*sind(0)
                   ]';

julia> f = Framework(g, p)
Framework with 6 vertices and 9 edges, vertex labels [1, 2, 3, 4, 5, 6] and 2-realization
GeometryBasics.Point{2, Float64}[[-0.5, 0.8660254037844386], [-0.5, -0.8660254037844386], [1.0, 0.0], [-0.75, 1.299038105676658], [-0.75, -1.299038105676658], [1.5, 0.0]]

julia> mat_group = RigidityTheoryTools.rotation_group_around_origin_2d(1 // 3) # rotation group around origin generated by rotation of angle 1/3*2π
Matrix group of degree 2
  over algebraic closure of rational field

julia> repr = Oscar.hom(H, mat_group, Oscar.gens(H), Oscar.gens(mat_group)) # representation of H as matrix group
Group homomorphism
  from permutation group of degree 6
  to matrix group of degree 2 over QQBar

julia> is_symmetric(f, repr)
true
``` 
"""
is_symmetric(f::Framework{PositionDim,PositionType}, phi::Oscar.GAPGroupHomomorphism{<:Group,<:Oscar.MatrixGroup}) where {PositionDim,PositionType} = is_symmetric(graph(f), coordinate_matrix(f), phi)
