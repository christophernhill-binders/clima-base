#!/usr/bin/env julia
# coding: utf-8

# In[1]:


using Pkg


# In[2]:


Pkg.add(PackageSpec(url="https://github.com/climate-machine/CLIMA"))


# In[3]:


Pkg.add("MPI")
Pkg.add("StaticArrays")
using MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.VTK
using LinearAlgebra
using Logging
using Dates
using Printf
using StaticArrays


MPI.Init()


# In[4]:


const uvec = (1, 2, 3)

function advectionflux!(F, state, _...)
  DFloat = eltype(state) # get the floating point type we are using
  @inbounds begin
    q = state[1]
    F[:, 1] = SVector{3, DFloat}(uvec) * q
  end
end

function upwindflux!(fs, nM, stateM, viscM, auxM, stateP, viscP, auxP, t)
  DFloat = eltype(fs)
  @inbounds begin
    ## determine the advection speed and direction
    un = dot(nM, DFloat.(uvec))
    qM = stateM[1]
    qP = stateP[1]
    ## Determine which state is "upwind" of the minus side
    fs[1] = un ≥ 0 ? un * qM : un * qP
  end
end

function initialcondition!(Q, x_1, x_2, x_3, _...)
  @inbounds Q[1] = exp(sin(2π * x_1)) * exp(sin(2π * x_2)) * exp(sin(2π * x_3))
end

function exactsolution!(dim, Q, t, x_1, x_2, x_3, _...)
  @inbounds begin
    DFloat = eltype(Q)

    y_1 = mod(x_1 - DFloat(uvec[1]) * t, 1)
    y_2 = mod(x_2 - DFloat(uvec[2]) * t, 1)

    y_3 = dim == 3 ? mod(x_3 - DFloat(uvec[3]) * t, 1) : x_3

    initialcondition!(Q, y_1, y_2, y_3)
  end
end

function setupDG(mpicomm, dim, Ne, polynomialorder, DFloat=Float64,
                 ArrayType=Array)

  @assert ArrayType === Array

  brickrange = (range(DFloat(0); length=Ne+1, stop=1), # x_1 corner locations
                range(DFloat(0); length=Ne+1, stop=1), # x_2 corner locations
                range(DFloat(0); length=Ne+1, stop=1)) # x_3 corner locations
  periodicity = (true, true, true)
  topology = BrickTopology(mpicomm, brickrange[1:dim];
                           periodicity=periodicity[1:dim])

  grid = DiscontinuousSpectralElementGrid(topology; polynomialorder =
                                          polynomialorder, FloatType = DFloat,
                                          DeviceArray = ArrayType,)
  spatialdiscretization = DGBalanceLaw(grid = grid, length_state_vector = 1,
                                       flux! = advectionflux!,
                                       numerical_flux! = upwindflux!)

end


# In[ ]:


let
  mpicomm = MPI.COMM_WORLD

  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)

  dim = 2
  Ne = 20
  polynomialorder = 4
  spatialdiscretization = setupDG(mpicomm, dim, Ne, polynomialorder)
  Q = MPIStateArray(spatialdiscretization, initialcondition!)
  filename = @sprintf("initialcondition_mpirank%04d", MPI.Comm_rank(mpicomm))
  writevtk(filename, Q, spatialdiscretization,
                                       ("q",))
  h = 1 / Ne                           # element size
  CFL = h / maximum(abs.(uvec[1:dim])) # time to cross the element once
  dt = CFL / polynomialorder^2         # DG time step scaling (for this

  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

  finaltime = 1.0
  if (parse(Bool, lowercase(get(ENV,"TRAVIS","false")))       #src
      && "Test" == get(ENV,"TRAVIS_BUILD_STAGE_NAME","")) ||  #src
    parse(Bool, lowercase(get(ENV,"APPVEYOR","false")))       #src
    finaltime = 2dt                                           #src
  end                                                         #src
  solve!(Q, lsrk; timeend = finaltime)

  filename = @sprintf("finalsolution_mpirank%04d", MPI.Comm_rank(mpicomm))
  writevtk(filename, Q, spatialdiscretization,
                                       ("q",))

  Qe = MPIStateArray(spatialdiscretization) do Qin, x, y, z, aux
    exactsolution!(dim, Qin, finaltime, x, y, z)
  end
  error = euclidean_distance(Q, Qe)
  with_logger(mpi_logger) do
    @info @sprintf("""Run with
                   dim              = %d
                   Ne               = %d
                   polynomial order = %d
                   error            = %e
                   """, dim, Ne, polynomialorder, error)
  end
end


# In[ ]:
