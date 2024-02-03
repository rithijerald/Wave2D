#include <iostream>
#include "Timer.h"
#include <stdlib.h>   // atoi
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <omp.h>

int default_size = 100;  // the default system size
int defaultCellWidth = 8;
double c = 1.0;      // wave speed
double dt = 0.1;     // time quantum
double dd = 2.0;     // change in system

using namespace std;

int main(int argc, char *argv[]) {
    int my_rank = 0;            // used by MPI
    
    // verify arguments
    if (argc != 5) {
        cerr << "usage: Wave2D size max_time interval n_thread" << endl;
        return -1;
    }
    int size = atoi(argv[1]);
    int max_time = atoi(argv[2]);
    int interval = atoi(argv[3]);
    int nThreads = atoi(argv[4]);
    int mpi_size;
    
    if (size < 100 || max_time < 3 || interval < 0 || nThreads <= 0) {
        cerr << "usage: Wave2D size max_time interval" << endl;
        cerr << "       where size >= 100 && time >= 3 && interval >= 0 && mpi_size > 0" << endl;
        return -1;
    }
    
    // start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // change # of threads
    omp_set_num_threads(nThreads);
    
    // create a simulation space
    double z[3][size][size];
    for (int p = 0; p < 3; p++)
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                z[p][i][j] = 0.0; // no wave
    
    // start a timer
    Timer time;
    time.start();
    
    // time = 0;
    // initialize the simulation space: calculate z[0][][]
    int weight = size / default_size;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i > 40 * weight && i < 60 * weight &&
                    j > 40 * weight && j < 60 * weight) {
                z[0][i][j] = 20.0;
            } else {
                z[0][i][j] = 0.0;
            }
        }
    }
    
// Calculation of Schroedinger's equation at time = 1 and parallelizing it with OpenMP.
#pragma omp parallel for
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++) {
            z[1][i][j] = z[0][i][j] + (pow(c, 2) / 2) * pow(dt / dd, 2) * (z[0][i + 1][j] + z[0][i - 1][j] + z[0][i][j + 1] + z[0][i][j - 1] - (4.0 * z[0][i][j]));
        }
    }
    
    //Calculation of stripe size, stripe_ends and stripe_begins for each rank.
    int stripe = size / mpi_size;     
    int stripe_begins[mpi_size];
    int stripe_ends[mpi_size];
    stripe_begins[my_rank] = stripe * my_rank;
    stripe_ends[my_rank] = (stripe * (my_rank + 1))-1;

    //Calculation of Schroedinger's equation at time = 2  and using MPI to exchange data across boundary.
    for (int t = 2; t < max_time; t++) {
        int p, q, r;
        p = t % 3;
        q = (t + 2) % 3;
        r = (t + 1) % 3;
            
        /*if(mpi_size == 1){
            continue;
        }*/
        if (my_rank == 0 ) {
            MPI_Send((z[q][stripe_ends[my_rank]]), size, MPI_DOUBLE, my_rank +
                    1, 0, MPI_COMM_WORLD);
            MPI_Status status;
            MPI_Recv(z[q][stripe], size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD,
                    &status);

        }
        else if (my_rank == mpi_size - 1) {
            MPI_Send(z[q][stripe_begins[my_rank]], size, MPI_DOUBLE, my_rank -
                    1, 0, MPI_COMM_WORLD);
            MPI_Status status;
            MPI_Recv((z[q][stripe_begins[my_rank]-1]), size, MPI_DOUBLE, my_rank - 1, 0,
                    MPI_COMM_WORLD, &status);
        } 
        else if(my_rank % 2 == 0)
            {
            MPI_Send(z[q][stripe_begins[my_rank]], size, MPI_DOUBLE, my_rank - 1, 0,
                    MPI_COMM_WORLD);
            MPI_Send((z[q][stripe_ends[my_rank]-1]), size, MPI_DOUBLE, my_rank + 1,
                    0, MPI_COMM_WORLD);
        
            MPI_Status status;
            MPI_Recv((z[q][stripe_begins[my_rank]-1]), size, MPI_DOUBLE, my_rank - 1, 0,
                    MPI_COMM_WORLD, &status);
            MPI_Recv(z[q][stripe_ends[my_rank]], size, MPI_DOUBLE, my_rank + 1, 0,
                    MPI_COMM_WORLD, &status);
            }
            else{
                MPI_Status status;
            MPI_Recv((z[q][stripe_begins[my_rank]-1]), size, MPI_DOUBLE, my_rank - 1, 0,
                    MPI_COMM_WORLD, &status);
            MPI_Recv(z[q][stripe_ends[my_rank]], size, MPI_DOUBLE, my_rank + 1, 0,
                    MPI_COMM_WORLD, &status);
            MPI_Send(z[q][stripe_begins[my_rank]], size, MPI_DOUBLE, my_rank - 1, 0,
                    MPI_COMM_WORLD);
            MPI_Send((z[q][stripe_ends[my_rank]]), size, MPI_DOUBLE, my_rank + 1,
                    0, MPI_COMM_WORLD);

            }
            
        
        
        //Parallelization for the Schroedinger's formula
#pragma omp parallel for
        for (int i = stripe_begins[my_rank]; i < stripe_ends[my_rank] ; i++) {
            if (i == 0 || i == size - 1) {
                continue;
            }
            for (int j = 1; j < size - 1; j++) {
                
                z[p][i][j] =2.0 * z[q][i][j] - z[r][i][j] + (pow(c, 2) * pow(dt / dd, 2)* (z[q][i + 1][j] + z[q][i - 1][j] + z[q][i][j + 1] +z[q][i][j - 1] - (4.0 * z[q][i][j])));
            }
        }
        
        //output if it's interval
        if (interval != 0 && t % interval == 0) {
            //Aggregate all results from all ranks
            if (my_rank == 0) {
                for (int rank = 1; rank < mpi_size; ++rank) {
                    MPI_Status status;
                    MPI_Recv(z[q][stripe_begins[my_rank]], stripe * size, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);
                }
                
                cout << t << endl;
                for (int j = 0; j < size; j++) {
                    for (int i = 0; i < size; i++) {
                        cout << z[p][i][j] << " ";
                    }
                    cout << endl;
                }
                cout << endl;
                
            } else {
                MPI_Send(z[q][stripe_begins[my_rank]], stripe * size, MPI_DOUBLE, 0, 0,
                        MPI_COMM_WORLD);
            }
        }
    } // end of simulation
    
    MPI_Finalize(); // shut down MPI
    
    // finish the timer
    if (my_rank == 0) {
        cerr << "Elapsed time = " << time.lap() << endl;
    }
    
    return 0;
}
