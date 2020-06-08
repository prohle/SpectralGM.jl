module SpectralGM

  function __init__()
  end

  using Revise
  using SuiteSparse
  using DataStructures
  using SparseArrays
  using Random
  using LinearAlgebra
  using Statistics
  using Printf

  using DelimitedFiles

  using Plots

  include("cmg_precondition.jl")
  export cmg_precondition, steiner_group

  include("Calculator.jl")
  export test, greet

end
