using Graphs
import Oscar: Group, GroupElem
import Oscar
import GAP

# Import all the functions we need to extend from Graphs.jl
import Graphs: AbstractGraph, edgetype, nv, ne, vertices, edges,
    src, dst, has_edge, has_vertex, inneighbors, outneighbors,
    is_directed

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

# --- Helper Functions (Not in AbstractGraph, but necessary) ---

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
out_edges_with_gains(g::QuotientGainGraph, s::Int) = g.out_gains[s]

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
    # In Oscar.jl, `v^g` is the standard notation
    s_new = s^g
    d_new = d^g

    # Get the concrete type of the edge (e.g., SimpleEdge{Int})
    EdgeType = typeof(edge)

    # Construct and return a new edge of the same type
    # This relies on the edge type having a (src, dst) constructor
    return EdgeType(s_new, d_new)
end

using Oscar

"""
    representative_action(G::Oscar.Group, x, y, fun::Function = (x, g) -> x^g)

Finds a group element `g` in `G` such that `fun(x, g) == y`.

This function implements a bi-directional Breadth-First Search (a "meet-in-
the-middle" algorithm). It builds two search trees, one starting from `x`
and one from `y`, and searches for a common element (a "collision").

The action `fun` must be a valid right group action, meaning that
`fun(el, g1 * g2) == fun(fun(el, g1), g2)` for all `g1, g2` in `G`
and `el` in the orbit.

# Arguments
- `G::Oscar.Group`: The group containing the element we are looking for.
- `x`: The starting element of the orbit.
- `y`: The target element of the orbit.
- `fun::Function`: (Optional) A function `(el, g::Oscar.GroupElem) -> el_new`
  that describes the right action. Defaults to Oscar's `(x, g) -> x^g`.

# Returns
- An `Oscar.GroupElem` `g` such that `fun(x, g) == y`.
- `nothing` if `x` and `y` are not in the same orbit (i.e., no such
  element `g` exists).
"""
function representative_action(G::Group, x, y, fun::Function=(x, g) -> x^g)
    # --- 1. Handle Trivial Case ---
    if x == y
        return one(G)
    end

    # --- 2. Initialize Generators ---
    # We need both generators and their inverses to traverse the
    # entire Schreier graph.
    gens_list = Oscar.gens(G)
    # S is our set of "steps"
    S = [gens_list; inv.(gens_list)]

    # --- 3. Initialize Data Structures ---
    # Get types for type-stable dictionaries and arrays
    T_orbit = typeof(x)
    T_group = eltype(G)

    # from_x: Dict(orbit_el => g) such that fun(x, g) == orbit_el
    from_x = Dict{T_orbit,T_group}(x => one(G))
    # from_y: Dict(orbit_el => h) such that fun(y, h) == orbit_el
    from_y = Dict{T_orbit,T_group}(y => one(G))

    # Queues for the BFS
    queue_x = T_orbit[x]
    queue_y = T_orbit[y]

    # --- 4. Bi-directional Search Loop ---
    # We continue as long as either search has nodes to expand.
    while !isempty(queue_x) || !isempty(queue_y)

        # --- 4a. Expand one layer from x ---
        if !isempty(queue_x)
            next_queue_x = T_orbit[]
            for current_x_el in queue_x
                # g_current is the element that maps x -> current_x_el
                g_current = from_x[current_x_el]

                for s in S
                    # new_g = g_current * s
                    # By the right action property:
                    # fun(x, new_g) = fun(x, g_current * s) 
                    #                = fun(fun(x, g_current), s)
                    #                = fun(current_x_el, s)
                    new_g = g_current * s
                    next_x_el = fun(current_x_el, s)

                    # Check for collision: Has the y-search already found this node?
                    if haskey(from_y, next_x_el)
                        # COLLISION! We found a path.
                        g = new_g
                        h = from_y[next_x_el]

                        # We have:
                        #   fun(x, g) = next_x_el  (from the x-search)
                        #   fun(y, h) = next_x_el  (from the y-search)
                        #
                        # We want `res` such that fun(x, res) = y.
                        #
                        # From fun(y, h) = next_x_el, apply inv(h) to both sides:
                        #   fun(fun(y, h), inv(h)) = fun(next_x_el, inv(h))
                        #   fun(y, h * inv(h))     = fun(next_x_el, inv(h))
                        #   y                      = fun(next_x_el, inv(h))
                        #
                        # Now substitute next_x_el with fun(x, g):
                        #   y = fun(fun(x, g), inv(h))
                        #   y = fun(x, g * inv(h))
                        #
                        # So, our result is g * inv(h)
                        return g * inv(h)
                    end

                    # If it's a new element for the x-search, add it
                    if !haskey(from_x, next_x_el)
                        from_x[next_x_el] = new_g
                        push!(next_queue_x, next_x_el)
                    end
                end
            end
            queue_x = next_queue_x
        end # end expanding from x

        # --- 4b. Expand one layer from y ---
        if !isempty(queue_y)
            next_queue_y = T_orbit[]
            for current_y_el in queue_y
                # h_current is the element that maps y -> current_y_el
                h_current = from_y[current_y_el]

                for s in S
                    new_h = h_current * s
                    next_y_el = fun(current_y_el, s)

                    # Check for collision: Has the x-search already found this node?
                    if haskey(from_x, next_y_el)
                        # COLLISION! We found a path.
                        g = from_x[next_y_el]
                        h = new_h

                        # We have:
                        #   fun(x, g) = next_y_el
                        #   fun(y, h) = next_y_el
                        #
                        # The logic is identical to the first collision check:
                        return g * inv(h)
                    end

                    # If it's a new element for the y-search, add it
                    if !haskey(from_y, next_y_el)
                        from_y[next_y_el] = new_h
                        push!(next_queue_y, next_y_el)
                    end
                end
            end
            queue_y = next_queue_y
        end # end expanding from y

    end # end while loop

    # --- 5. No Path Found ---
    # If we exit the loop, the queues are empty and no collision
    # was found. x and y are not in the same orbit.
    return nothing
end

function QuotientGainGraph(g::Graphs.AbstractSimpleGraph, group::Group)

end