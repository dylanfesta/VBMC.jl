#=
Small sandbox to visualize Gaussian Processes
=#

using Random
using Statistics, StatsBase, LinearAlgebra
using Distributions
using Base.Iterators
using Plots ; clibrary(:cmocean)
##

kerfun(x1::Real,x2::Real) = exp(-0.5*(x1-x2)^2)
kerfun(tup::Tuple{T,T}) where T<:Real = kerfun(tup...)
function regularize!(mat,regu=1E-9)
  for k in 1:min(size(mat)...)
    mat[k,k] += regu
  end
  mat
end

function kerfun(v1::AbstractVector{T},v2::AbstractVector{T}; regu=1E-9) where T<:Real
  out = map(xx->kerfun(xx[1],xx[2]) , product(v1,v2) )
  regularize!(out, regu)
end
kerfun(v1::AbstractVector) = kerfun(v1,v1)

##
# first test, just randomly generated

pointstest= range(-10.,10.; length=300)

covmat = kerfun(pointstest)
@assert isposdef(covmat)

gp_distr = MultivariateNormal(covmat)
nsampl = 20
samplesy = rand(gp_distr,nsampl)

plot(pointstest, samplesy , leg=false , linewidth=3)

##
# second test, some points are known
x_known = [ -7., 0., 1 , 5 ,-3., -2 ]
y_known = [ 1. , 0. , 2., -0.5, 3, 1]

pointstest= range(-10.,10.; length=300)
mean_full = ( kerfun(pointstest,x_known)/kerfun(x_known) ) * y_known
cov_full = kerfun(pointstest) -
        (kerfun(pointstest,x_known)/kerfun(x_known))*kerfun(x_known,pointstest) |> Symmetric
regularize!(cov_full)
@assert isposdef(cov_full)

gp_distr = MultivariateNormal(mean_full,cov_full)
nsampl = 200
samplesy = rand(gp_distr,nsampl)


plot(pointstest, samplesy , leg=false , linewidth=3, color=:black,opacity=0.05)
scatter!(x_known,y_known, marker=:star, color=:red)
