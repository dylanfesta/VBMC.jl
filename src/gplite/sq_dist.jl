
"""
    sq_dist(a::AbstractMatrix,b::AbstractMatrix)

computes a matrix of all pairwise squared distances
between two sets of vectors, stored in the columns of the two matrices, `a`
(of size `D` by `n`) and `b` (of size `D` by `m`). If only a single argument is given,
the missing matrix is taken to be identical
to the first.

*output* :  matrix `C` of size (`n` by `m`) , where `C[i,j]` is the squared distance
between column `i` of `a` and column `j` of `b`
"""
function sq_dist(a::AbstractMatrix,b::AbstractMatrix)
  (D,n) = size(a)
  (d,m) = size(b)
  @assert d == D "Error: column lengths must agree."
  # subtract the mean
  mu = (m/(n+m))*mean(b;dims=2) .+ (n/(n+m))*mean(a; dims=2)
  a .-= mu
  b .-= mu
  sa = mapreduce(x->x*x,+, a; dims = 1)
  sb = mapreduce(x->x*x,+, b; dims = 1)
  C = broadcast(+, sa', sb, -2 .* (a'*b))
  @.  C = max(C,0.0)
end
sq_dist(a::AbstractMatrix) = sq_dist(a,a)

# small test
# a = randn(5,4)
# b =  randn(5,22)
# C = sq_dist(a,b)
# i,j = (1,18)
# sum( @. (a[:,i]-b[:,j])^2 ) - C[i,j]
