"""
%% 
% CMG, Copyright (c) 2008-2020  Ioannis Koutis, Gary Miller               
% 
% The CMG solver is distributed under the terms of the GNU General Public  
% Lincense Version 3.0 of the Free Software Foundation.                    
% CMG is also available under other licenses; contact authors for details. 


%% List of functions in this file:
%
% cmg_precondition:    main function
% 
% validate_input
% make_preconditioner
%
% Combinatorial Functions
% steiner_group
% forest_components_
% split_forest_
%
% Numerical Functions
% ldl_
% preconditioner_           
%sd

"""

## Prerequisite
# import Pkg
# Pkg.add("LinearAlgebra", "SparseArrays", "Laplacians","Statistics")
# using LinearAlgebra, SparseArrays, Laplacians, Statistics

"""
   cmg_precondition(A::SparseMatrixCSC)

For an abritrary b-side in the null space of A, the
system Ax = b can be solved by:
   x = pcg(A, b, tol, iter, pfun)

# Arguments
- `A::SparseMatrixCSC`: SDD matrix (positive off-diagonals not supported currently)

# Outputs
- `pfun`: Preconditioning function for A
- `H`: Preconditioning hierarchy used internally in pfun
- `flag`: failure status variable

# Example
include("cmg_precondition.jl")
A = lap(grid2(100,100))
@time pfun, H, flag = cmg_precondition(A); 
"""

mutable struct Chol
    ld::SparseMatrixCSC
    ldT::SparseMatrixCSC
    d::Array{Float64,1}
    p::Array{Int64,1}    
    invp::Array{Int64,1}
end

Chol() = Chol(sparse([]), sparse([]), sparse([]), [], [])

mutable struct HierarchyLevel
    sd::Bool
    islast::Bool
    iterative::Bool
    A::SparseMatrixCSC
    invD::Array{Float64}
    cI::Array{Int}
    nc::Int
    R::SparseMatrixCSC
    repeat::Int
    chol::Chol
end

HierarchyLevel() = HierarchyLevel(true, 0, 1, sparse([]), [], [], 0, sparse([]), 0, Chol())


function cmg_precondition(A::SparseMatrixCSC)

    msg = undef
    flag = undef
   
    if size(A, 1) < 500 # handle small input
        msg = "Input matrix is small. Solve Ax=B with A\\b."
    else # validate input
        (A_, flag) = validate_input(A)

        if flag == 1
            msg = "The input matrix must be symmetric."
        elseif flag == 2
            msg = "The current version of CMG does not support positive off-diagonals."
        end
    end

    if typeof(msg) != UndefInitializer
        println("\n", msg, "\n")
        pfun = undef;
        H = undef;
        return pfun, H, flag
    end

   # initialize H
    H = HierarchyLevel[]

   # initialize a hierarchy level

    push!(H, HierarchyLevel())

   # A_ is Laplacian
    if size(A_, 1) > size(A, 1)
        H[1].sd = true # original matrix is strongly dominant
    else
        H[1].sd = false
    end

   # construct hierarchy
    loop = true
    j = 1
    h_nnz = 0
    n = 0 # initialize n
    # build up H
    while loop
        n = size(A_, 1)

      # direct method for small size
        if (n < 500)
            H[j].iterative = 0
            break
        end

        dA_ = diag(A_)
        (cI, ~) = steiner_group(A_, dA_)
        nc = maximum(cI)
        H[j].islast = 0
        H[j].A = A_ # !
        H[j].invD = 1 ./ (2 * dA_) # !
        H[j].cI = cI
        H[j].nc = nc
        H[j].R = sparse(cI, 1:n, ones(n), nc, n) # added for efficiency

        # check for full contraction
        if nc == 1
            H[j].islast = 1
            H[j].iterative = 1
        end

        # check for hierarchy stagnation for potentially bad reasons
        h_nnz = h_nnz + nnz(A_)
        if (nc >= n - 1) || (h_nnz > 5 * nnz(H[1].A))
            H[j].islast = 1
            H[j].iterative = 1
            flag = 3 # indicates stagnation
            @warn "CMG convergence may be slow due to matrix density. Future versions of CMG will eliminate this problem."
            break
        end

        Rt = sparse(cI, 1:n, 1, nc, n) # ! take out double
        A_ = Rt * (H[j].A) * Rt'

        j += 1
        push!(H, HierarchyLevel())
    end

    
    # code for last hierarchy level
    if flag == 0
        H[j].islast = 1
        H[j].iterative = 0
        H[j].A = A_[1:n - 1,1:n - 1]
      # L, D, p = ldl(H[j].A, 'vector')
        L, D, p = ldl_(H[j].A)
        H[j].chol.ld = L
        H[j].chol.ldT = L'
        H[j].chol.d = (1 / diag(D))' # ! 1.?
        H[j].chol.p = p # x = A*y => y(p) = LT\(H{j}.d.*(L\x(p))); 
    end

    # determine number of recursive calls between levels
    for k in 1:(j - 2)
        H[k].repeat = max(floor(nnz(H[k].A) / nnz(H[k + 1].A) - 1), 1)
    end

    H[j - 1].repeat = max(floor(nnz(H[j - 1].A) / nnz(H[j].chol.ld) - 1), 1)

   # H = cell2mat(H);

   # create precondition function
    pfun = make_preconditioner(H)
    return pfun, H, flag
end

"""
   function (A0, flag) = validate_input(A::SparseMatrixCSC)
Input validation. 

# Arguments
-`A`: Laplacian, possibly extended by one coordinate when A is SD

# Outputs
-`A0`
-`flag`: failure flag code indicating issue with input validity
"""

function validate_input(A::SparseMatrixCSC)
    flag = 0

   # check symmetry
    if !issymmetric(A)
        flag = 1
    end

   # detect strict dominance
    n = size(A, 1)
    dA = diag(A)
    sA = sum(A, dims = 2)
    sAp = (sA + abs.(sA)) / 2
    sd = sAp ./ dA .> 1e-13 # if sd(i) = 1 then row i is numerically strictly dominant

   # A is a sparse matrix
    (i, j, v) = findnz(A)

   # check for positive off-diagonals that are not currently supported
    if maximum(v[i .!= j]) > 0
        flag = 2
    end

   # augment by extra coordinate if strictly dominant
    if maximum(sd)
        ex_v = -sAp[sd]
        exd = length(ex_v)
        exd_c = nonzeros(sd) # get coordinates
        i = Int[i;(n + 1) * ones(exd); exd_c; n + 1]
        j = Int[j;exd_c; (n + 1) * ones(exd);n + 1]
        v = [v; ex_v; ex_v; -sum(ex_v)]
        A0 = sparse(i, j, v, n + 1, n + 1)
    else
        A0 = A
    end

    return A0, flag
end

"""
   function(cI, nc) = steiner_group(A, dA_)
Steiner groups
# Arguments
-`A`: Laplacian
"""
function steiner_group(A, dA_)

    (M, c1) = findmin(A, 1) # c1 represents a unimodal forest #! dims = 1

   # adjust c1 to array of indices
    C = Array{Int,1}(undef, length(c1))
    
    for i in 1:length(c1)
        C[i] = c1[i][1]
    end

    split_forest_!(C)

    efd = abs.(M' ./ dA_)

    if minimum(efd) < 1 / 8 # low effective degree nodes found
        C = update_groups_(A, C, dA_)
    end
    cI, nc, ~ = forest_components_(C)
end

"""
   function C = split_forest_(C1::Array{Int})
decompose unimodal forest into low conductance components
"""
function split_forest_!(C::Array{Int64})
    n = length(C)
    ancestors = zeros(Int, n)
    indegree = zeros(Int, n + 2)
    visited = falses(n) # logical sparse array

    walkbuffer = zeros(Int, 20);
    newancestorbuff = zeros(Int, 20)

   # compute indegrees
    for i in 1:n
        indegree[C[i]] = indegree[C[i]] + 1
    end

   # partition into clusters of small diameter

    for j in 1:n
        jwalk = j
        startwalk = true
      
        while (startwalk && (indegree[jwalk] == 0 ) && !visited[jwalk])
            startwalk = false
            ancestors_in_path = 0 # change over C-CMG
            k = 1
            walkbuffer[k] = jwalk
            newancestorbuff[k] = 0
            while (k <= 6 || visited[jwalk]) # +1 for c-indexing adjust
                jwalk = C[jwalk]
                walkterminated = (jwalk == walkbuffer[k]) || ((k > 1) && (jwalk == walkbuffer[k - 1]))
                if walkterminated
                    break; # while
                end
                k = k + 1
                walkbuffer[k] = jwalk
                if visited[jwalk]
                    newancestorbuff[k] = ancestors_in_path
                else
                    ancestors_in_path = ancestors_in_path + 1
                    newancestorbuff[k] = ancestors_in_path
                end
            end

            if k > 6 # large diameter - cut 
                middlek = Int(ceil(k / 2))
                C[walkbuffer[middlek]] = walkbuffer[middlek] # cut middle edge
                indegree[walkbuffer[middlek + 1]] = indegree[walkbuffer[middlek + 1]] - 1 # update indegree

                for ik in (middlek + 1):k
                    ancestors[walkbuffer[ik]] = ancestors[walkbuffer[ik]] - ancestors[walkbuffer[middlek]]
                end
            
            # update ancestors and visited flag
                for ik in 1:middlek
                    visited[walkbuffer[ik]] = true
                    ancestors[walkbuffer[ik]] = ancestors[walkbuffer[ik]] + newancestorbuff[ik]
                end

            # set first vertex in new walk
                jwalk = walkbuffer[middlek + 1]
                startwalk = true
            end # end cut procedure

         # commit walk changes
            if !startwalk
                for ik in 1:k
                    ancestors[walkbuffer[ik]] = ancestors[walkbuffer[ik]] + newancestorbuff[ik]
                    visited[walkbuffer[ik]] = true
                end
            end
        end # outer while
    end
    
   # tree partition into clusters of high conductance
    for j in 1:n
        jwalk = j
        startwalk = true

        while startwalk && (indegree[jwalk] == 0)
            startwalk = false
            jwalkb = jwalk
            cut_mode = false
            # initialize new_front
            new_front = 0
            removed_ancestors = []

            while true
                jwalka = C[jwalk]
                walkterminated = (jwalka == jwalk) || (jwalka == jwalkb)
                if walkterminated
                    break; # while
                end

                if (!cut_mode && (ancestors[jwalk] > 2) && (ancestors[jwalka] - ancestors[jwalk] > 2)) # possibly low conductance - make cut
                    C[jwalk] = jwalk # cut edge
                    indegree[jwalka] = indegree[jwalka] - 1
                    removed_ancestors = ancestors[jwalk]
                    new_front = jwalka
                    cut_mode = true
                end # end making cut
            
                jwalkb = jwalk
                jwalk = jwalka
                if cut_mode
                    ancestors[jwalk] = ancestors[jwalk] - removed_ancestors
                end
            end
            if cut_mode
                startwalk = true
                jwalk = new_front
            end
        end
    end
end # split_forest_

"""
   function C1 = update_groups_(A, C, dA_)
update groups based on nodes with low effective degree
# Arguments
-`A`: Laplacian
"""
function update_groups_(A, C, dA_)
    n = size(C, 1)
    B = zeros(n)
   
   # B[j] is the total tree weight incident to node j
    for i in 1:n
        if C[i] != i
            B[i] = A[i,C[i]] + B[i]
            B[C[i]] = A[i,C[i]] + B[C[i]]
        end
    end

    ndx = findall(x->x > 0.125, B ./ dA_)
    C[ndx] = Array{Int32}(ndx)

    C1 = C

    return C1
end # update_groups_


"""
   function[cI, nc, csizes] = forest_components_(C)
forest components, connected components in unimodal forest
# Arguments
-`C`: unimodal tree
"""

function forest_components_(C)
    n = size(C, 1)
    cI = zeros(Int, n)
    cSizes = zeros(Int, n)
    buffer = zeros(Int, 100)

    ccI = 1
    for j in 1:n
        bufferI = 1
        jwalk = j

      # tree walk
        while cI[jwalk] == 0
            cI[jwalk] = ccI
            buffer[bufferI] = jwalk
            bufferI = bufferI + 1

            if bufferI == size(buffer, 1) # for memory efficiency
                buffer_ = zeros(Int, min(2 * size(buffer, 1), n))
                buffer_[1:size(buffer, 1)] = buffer
                buffer = buffer_
            end

            jwalk = C[jwalk]
        end # while

        bufferI = bufferI - 1
        en = C[jwalk] # end node
        if cI[en] != ccI
            cI[buffer[1:bufferI]] .= cI[en] # ! .=
        else
            ccI = ccI + 1
        end
        cSizes[en] = cSizes[en] + bufferI
    end # for
    if cSizes[ccI] == 0
        ccI = ccI - 1
    end

    nc = ccI
    cSizes = cSizes[1:ccI]

    return cI, nc, cSizes
end

"""
   function pfun = make_preconditioner(H)
Make preconditioner
"""
function make_preconditioner(H::Array{HierarchyLevel,1})

    if !H[1].sd 
        pfun = b->preconditioner_(b, H, 1)
        return pfun
    end

    if H[1].sd 
        pfun = b->preconditioner_sd(b, H)
        return pfun
    end

end

"""
   function x = preconditioner_(b, H::Array{HierarchyLevel,1}, level)
preconditioner
"""
function preconditioner_(b::Array{Float64}, H::Array{HierarchyLevel,1}, level::Int64)
    n = size(H[level].A, 1)

   # base case I: last level direct
    if (H[level].islast && !H[level].iterative)
        b = b[1:n]
        x = zeros(Float64, n + 1)

        L = H[level].chol.ld
        LT = H[level].chol.ldT
        di = H[level].chol.d
        p = H[level].chol.p

      # back substitution needed here
        x[p] = LT \ (di .* (L \ b[p])) # ! .*
        return x
    end

   # base case II: last level iterative
    if (H[level].islast && H[level].iterative)
        invD = H[level].invD
        x = invD .* b
        return x
    end

   # main cycle
   # unpack variables for efficiency
    invD = H[level].invD
    A = H[level].A
    repeat = H[level].repeat
    cI = H[level].cI
    R = H[level].R

    r = Array{Float64}(undef, size(A, 1))
    b_small = Array{Float64}(undef, size(R, 1))
    z = Array{Float64}(undef, size(R, 1))


    for j in 1:repeat
      # jacobi pre-smooth
        if j == 1
            x = invD .* b # previous x = 0
        else
            x = x + invD .* (b - A * x)
        end

      # residual
        r = b - A * x

      # projection of residual onto smaller level
      # b_small = spzeros(nc)
      # for k in 1:n
      #     b_small[cI[k]] = b_small[cI[k]] + r[k]
      # end
        b_small = R * r

        z = preconditioner_(b_small, H, level + 1)

      # interpolation
        x = x + z[cI]    #z[cI] does interpolation

      # Jacobi post-smooth
        x = x + invD .* (b - A * x)
    end
    return x
end


"""
   function x = preconditioner_sd(H::Array{HierarchyLevel}, b)
preconditioner sd
"""
function preconditioner_sd(b::Array{Float64}, H::Array{HierarchyLevel})
    n = length(b)
    push!(b, -sum(b))

    x = b->preconditioner_(b, H, 1)

    x = x[1:n] + x[n + 1]
    return x
end

"""
    function[L, D, p] = ldl_(A)
ldl Factorization with min degree heuristic
"""
function ldl_(A::SparseMatrixCSC)
    n = size(A, 1)
    D = zeros(n)

    # trivial min degree heuristic: sorting degrees
    (iA, jA, ~) = findnz(A)    
    Ao = sparse(iA, jA, ones(length(iA)), n, n)
    dA = sum(Ao, dims = 1)
    p = sortperm(dA[:])

    A = A[p,p]
    L = sparse(1:n, 1:n, ones(n)) # identity matrix
    for i in 1:n
        vi = A[i + 1:n,i]
        di = A[i,i]
        D[i] = di
        L[i + 1:n,i] = vi / di
        Z = vi * vi' / di
        (iz, jz, vz) = findnz(Z)
        Z = sparse(iz .+ i, jz .+ i, vz, n, n)
        A = A - Z
    end

    D = sparse(1:n, 1:n, D, n, n)
    return L, D, p
end

