# Wave2D
The problem is to parallelize a sequential version of a two-dimensional wave diffusion program, using MPI and OpenMP. The water surface is taken as a two dimensional square surface. This square surface is then partitioned into N by N cells. Each cell (i,j) will compute it’s surface height from its previous neighboring cells. Zt_i,j, Zt-1_i,j, and Zt-2_i,j are the surface height of cell(i, j) at time t, time t-1, and time t-2 respectively.


Schroedinger’s wave formula for Zt_i,j (where t >= 2 ):
Zt_i,j = 2.0 * Zt-1_i,j – Zt-2_i,j + c2 * (dt/dd) 2 * (Zt-1_i+1,j + Zt-1_i-1,j + Zt-1_i,j+1 + Zt-1_i,j-1 – 4.0 * Zt-1_i,j)
where
c is the wave speed and should be set to 1.0,
dt is a time quantum for simulation, and should be set to 0.1, and
dd is a change of the surface, and should be set to 2.0

Schroedinger’s wave formula for Zt_i,j (at t == 1)
Zt_i,j = Zt-1_i,j + c2 / 2 * (dt/dd) 2 * (Zt-1_i+1,j + Zt-1_i-1,j + Zt-1_i,j+1 + Zt-1_i,j-1 – 4.0 * Zt-1_i,j)
