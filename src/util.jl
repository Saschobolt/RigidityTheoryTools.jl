_point_matrix(points::AbstractVector{<:Point}) = Matrix(hcat(points...))

function _translation_vector(translation::AbstractVector, dim::Integer)
    if length(translation) != dim
        throw(DimensionMismatch("translation vector has length $(length(translation)), expected $dim"))
    end

    return collect(translation)
end

function _check_transformation_dimensions(matrix::AbstractMatrix, translation::AbstractVector, dim::Integer)
    if size(matrix) != (dim, dim)
        throw(DimensionMismatch("transformation matrix has size $(size(matrix)), expected ($dim, $dim)"))
    end

    return _translation_vector(translation, dim)
end

function _is_isometry(matrix::AbstractMatrix)
    gram = transpose(matrix) * matrix
    identity = Matrix{eltype(gram)}(I, size(matrix, 2), size(matrix, 2))

    return isapprox(gram, identity)
end

"""
    transform!(f::AbstractEmbeddedGraph, matrix::AbstractMatrix, translation::AbstractVector; check_iso::Bool=false)

Apply the Euclidean transformation ``p \\mapsto A p + t`` to every vertex of `f`
in place and return `f`.

If `check_iso=true`, the linear part `matrix` is checked to be orthogonal.

# Examples
```jldoctest
julia> using Graphs

julia> G = path_graph(2);

julia> f = Framework(G, [0.0 1.0; 0.0 0.0]);

julia> transform!(f, [0.0 -1.0; 1.0 0.0], [1.0, 2.0]; check_iso=true);

julia> coordinate_matrix(f)
2×2 Matrix{Float64}:
 1.0  1.0
 2.0  3.0
```
"""
function transform!(f::AbstractEmbeddedGraph, matrix::AbstractMatrix, translation::AbstractVector; check_iso::Bool=false)
    coords = coordinate_matrix(f)
    dim = size(coords, 1)
    translation = _check_transformation_dimensions(matrix, translation, dim)

    if check_iso && !_is_isometry(matrix)
        throw(ArgumentError("The transformation matrix is not orthogonal"))
    end

    transformed_coords = matrix * coords .+ reshape(translation, :, 1)
    coordinate_matrix!(f, transformed_coords)

    return f
end

"""
    transform(f::AbstractEmbeddedGraph, matrix::AbstractMatrix, translation::AbstractVector; check_iso::Bool=false)

Return a transformed copy of `f`, with each vertex position mapped by
``p \\mapsto A p + t``.

If `check_iso=true`, the linear part `matrix` is checked to be orthogonal.

# Examples
```jldoctest
julia> using Graphs

julia> G = path_graph(2);

julia> f = Framework(G, [0.0 1.0; 0.0 0.0]);

julia> g = transform(f, [0.0 -1.0; 1.0 0.0], [1.0, 2.0]; check_iso=true);

julia> coordinate_matrix(f)
2×2 Matrix{Float64}:
 0.0  1.0
 0.0  0.0

julia> coordinate_matrix(g)
2×2 Matrix{Float64}:
 1.0  1.0
 2.0  3.0
```
"""
function transform(f::AbstractEmbeddedGraph, matrix::AbstractMatrix, translation::AbstractVector; check_iso::Bool=false)
    return transform!(deepcopy(f), matrix, translation; check_iso=check_iso)
end

"""
    kabsch(points::AbstractMatrix, target::AbstractMatrix)
    kabsch(points::AbstractVector{<:Point}, target::AbstractVector{<:Point})

Compute a least-squares optimal Euclidean isometry aligning `points` with
`target`.

The columns of the two matrices are interpreted as point clouds. The returned
tuple `(matrix, translation, rmsd)` minimizes
``\\|A \\cdot points + t - target\\|`` over orthogonal matrices `A` and
translation vectors `t`. The `rmsd` is the root mean squared point distance
after alignment.

# Examples
```jldoctest
julia> points = [0.0 1.0 0.0; 0.0 0.0 1.0];

julia> target = [0.0 -1.0; 1.0 0.0] * points .+ reshape([1.0, 2.0], :, 1);

julia> A, t, rmsd = kabsch(points, target);

julia> isapprox(A, [0.0 -1.0; 1.0 0.0]; atol=1e-12)
true

julia> isapprox(t, [1.0, 2.0]; atol=1e-12)
true

julia> isapprox(rmsd, 0.0; atol=1e-12)
true
```

```jldoctest
julia> using GeometryBasics

julia> points = [Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)];

julia> target = [Point(1.0, 2.0), Point(1.0, 3.0), Point(0.0, 2.0)];

julia> A, t, rmsd = kabsch(points, target);

julia> isapprox(A, [0.0 -1.0; 1.0 0.0]; atol=1e-12)
true

julia> isapprox(t, [1.0, 2.0]; atol=1e-12)
true
```
"""
function kabsch(points::AbstractMatrix, target::AbstractMatrix)
    if size(points) != size(target)
        throw(DimensionMismatch("points has size $(size(points)), while target has size $(size(target))"))
    end
    if size(points, 2) == 0
        throw(ArgumentError("point clouds must contain at least one point"))
    end

    dim, num_points = size(points)
    points_centroid = vec(sum(points; dims=2) ./ num_points)
    target_centroid = vec(sum(target; dims=2) ./ num_points)
    centered_points = points .- reshape(points_centroid, dim, 1)
    centered_target = target .- reshape(target_centroid, dim, 1)

    factorization = svd(centered_target * transpose(centered_points))
    matrix = factorization.U * factorization.Vt
    translation = target_centroid - matrix * points_centroid
    residual = matrix * points .+ reshape(translation, dim, 1) - target
    rmsd = sqrt(sum(abs2, residual) / num_points)

    return matrix, translation, rmsd
end

function kabsch(points::AbstractVector{<:Point}, target::AbstractVector{<:Point})
    if length(points) != length(target)
        throw(DimensionMismatch("points has length $(length(points)), while target has length $(length(target))"))
    end
    if isempty(points)
        throw(ArgumentError("point clouds must contain at least one point"))
    end

    return kabsch(_point_matrix(points), _point_matrix(target))
end

function _check_sphere_intersection_data(points::AbstractMatrix, distances::AbstractVector)
    dim, num_centers = size(points)
    if dim != num_centers
        throw(DimensionMismatch("expected $dim centers in $dim-dimensional space, got $num_centers centers"))
    end
    if length(distances) != dim
        throw(DimensionMismatch("distances has length $(length(distances)), expected $dim"))
    end
    if dim == 0
        throw(ArgumentError("at least one sphere is required"))
    end

    return collect(distances)
end

function _sphere_intersection_tolerance(height_squared, points::AbstractMatrix, distances::AbstractVector)
    scale = max(one(float(abs(height_squared))), maximum(abs, points)^2, maximum(abs, distances)^2)

    return 100 * eps(typeof(scale)) * scale
end

function _oriented_normal(basis_matrix::AbstractMatrix)
    normal = vec(nullspace(transpose(basis_matrix)))
    index = findfirst(x -> !iszero(x), normal)
    if index !== nothing && normal[index] < 0
        normal = -normal
    end

    return normal
end

"""
    sphere_intersection(points::AbstractMatrix, distances::AbstractVector; sign=1)
    sphere_intersection(points::AbstractVector{<:Point}, distances::AbstractVector; sign=1)

Return one of the two intersection points of `d` spheres in `d`-dimensional
space.

The centers are the columns of `points`, or the entries of a vector of
`Point`s, and `distances` contains the corresponding radii. The keyword `sign`
chooses the halfspace determined by the affine span of the centers. With the
deterministic orientation used here, `sign=1` chooses the side whose normal has
first nonzero coordinate positive, and `sign=-1` chooses the opposite side.

# Examples
```jldoctest
julia> using GeometryBasics

julia> p = sphere_intersection([Point(0.0, 0.0), Point(1.0, 0.0)], [1.0, 1.0]; sign=1);

julia> isapprox(collect(p), [0.5, sqrt(3) / 2])
true

julia> q = sphere_intersection([Point(0.0, 0.0), Point(1.0, 0.0)], [1.0, 1.0]; sign=-1);

julia> isapprox(collect(q), [0.5, -sqrt(3) / 2])
true
```
"""
function sphere_intersection(points::AbstractMatrix, distances::AbstractVector; sign::Real=1)
    distances = _check_sphere_intersection_data(points, distances)
    dim = size(points, 1)
    side = Base.sign(sign)
    if side == 0
        throw(ArgumentError("sign must be nonzero"))
    end

    if dim == 1
        return points[:, 1] .+ side .* distances[1]
    end

    base = points[:, 1]
    basis_matrix = points[:, 2:end] .- reshape(base, :, 1)
    if rank(basis_matrix) != dim - 1
        throw(ArgumentError("sphere centers must be affinely independent"))
    end

    right_hand_side = (vec(sum(abs2, basis_matrix; dims=1)) .+ distances[1]^2 .- distances[2:end] .^ 2) ./ 2
    projection_coefficients = (transpose(basis_matrix) * basis_matrix) \ right_hand_side
    projection = base + basis_matrix * projection_coefficients
    height_squared = distances[1]^2 - sum(abs2, projection - base)
    tolerance = _sphere_intersection_tolerance(height_squared, points, distances)

    if height_squared < -tolerance
        throw(DomainError(height_squared, "the spheres have no real common intersection"))
    end

    normal = _oriented_normal(basis_matrix)

    return projection + side * sqrt(max(height_squared, zero(height_squared))) * normal
end

function sphere_intersection(points::AbstractVector{<:Point}, distances::AbstractVector; sign::Real=1)
    if isempty(points)
        throw(ArgumentError("at least one sphere is required"))
    end

    return Point(sphere_intersection(_point_matrix(points), distances; sign=sign)...)
end
