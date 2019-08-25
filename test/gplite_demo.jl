using Revise
using VBMC ; const V = VBMC
using LinearAlgebra, Random , Distributions , StatsBase, Statistics

##

VBMC.MeanFConst

x = range(-5,5. ; length=11)
y = sin.(x)

n_samples = 8
meanfun = V.MeanFConst(y)

##
whatevs = V.GaussianProcessLite(n_samples,x,y,meanfun)

Distribution.Normal