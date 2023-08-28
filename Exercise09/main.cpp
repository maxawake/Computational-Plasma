
#include <mpi.h>
#include <stdio.h>
#include <blitz/array.h>

#include <string>
#include <fstream>

#include <iostream>
#include <vector>
#define DIM 1

// Define constants for directions
#define LEFT 0

#define RIGHT 1
#define B 2

// Namespaces
using namespace blitz;
using namespace std;

// Some constants
const int N = 1000;
const int tend = 10000;

const int outputstep = 50;
const double dx = 1.;

const double dt = 0.1;


class Duck {
    public:
        float x;
        float v;
        int current;

        Duck(float x_0, float v_0, int current_0) {
            printf("Construct Duck...\n");
            x = x_0;
            v = v_0;
            current = current_0;
        }

        ~Duck() {
            printf("Destroy Ducks...\n");
        }

        void advance(blitz::Array<double, 1> &u) {
            this->x = this->x + dt * (this->v + u((int) this->x));
        }
};


int main(int argc, char *argv[])
{
    blitz::Array<double, 1> un, unp;

    int ducksToTransfer = 0;
    int ducksToReceive = 0;
    int hasDuck = 0;
    std::vector<Duck> ducklist; 

    // Buffers for data transfer

    double in[B], out[B], induck[B], outduck[B];

    int localsize, localindex, localmin, localmax;
    // MPIv ariables

    int size, rank, dims[DIM];
    int periods[DIM], coords[DIM];

    int neighbours[2];
    MPI_Comm cartcomm;

    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Find out how many processes are available
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cerr << "CPU " << rank << " of " << size << endl;

    // Create a cartesian communicator
    // Initialize the dim array
    dims[0] = 0;

    // Choose periodic boundaries for the first dimension
    periods[0] = 1;

    // Find a suitable decomposition
    MPI_Dims_create(size, 1, dims);

    // Acutally create the cartcomm
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, true, &cartcomm);

    // Find rank (this will be the same as for MPI_COMM_WORLD)
    MPI_Comm_rank(cartcomm, &rank);

    // Find coordinates in cartesian grid (pretty lame in 1D...)
    MPI_Cart_coords(cartcomm, rank, 1, coords);

    // Find neighbours
    MPI_Cart_shift(cartcomm, 0, 1, &neighbours[LEFT], &neighbours[RIGHT]);
    // This is a bit dirty, but it works. Somehow.

    // We divide to global coordinates of the computational grid
    int n_per_cell = N / dims[0];
    int remainder = N % dims[0];

    if (rank < remainder)
    {
        localmin = rank * (n_per_cell + 1);

        localmax = (rank + 1) * (n_per_cell + 1);
        localsize = n_per_cell + 1 + 2 * B;

        localindex = localmin + B;
    }
    else
    {

        localmin = rank * n_per_cell + remainder;
        localmax = (rank + 1) * n_per_cell + remainder;

        localsize = n_per_cell + 2 * B;
        localindex = localmin - B;
    }
    // local size includes the boundaries on both sides

    // localmin <= x < localmax
    // this is the local set of global coordinates

    printf("rank=%d coords=%d localmin/max/size=%d %d %d\n", rank, coords[0], localmin, localmax, localsize);

    // set the right size for all arrays
    // Blitz arrays can have arbitrary indices
    // so we don't have to introduce a different
    // set of indices

    un.resize(localsize);
    un.reindexSelf(localindex);

    unp.resize(localsize);
    unp.reindexSelf(localindex);

    // init
    // the intialization depends on a global position!

    for (int i = localmin; i < localmax; i++)
        un(i) = 1 + 0.3 * sin(2 * M_PI * 2 * i / N);

    auto send_data = [&]() {
        // copy borders
        // Right border to right CPU
        for (int i = 1; i <= B; i++)
        {
            out[i - 1] = un(localmax + i - 1 - B);
        }
        MPI_Send(out, 2, MPI_DOUBLE, neighbours[RIGHT], 7, cartcomm);
        MPI_Recv(in, 2, MPI_DOUBLE, neighbours[LEFT], 7, cartcomm, &status);
        MPI_Barrier(cartcomm);
        for (int i = 1; i <= B; i++)
        {
            un(localmin + i - 1 - B) = in[i - 1];
        }

        // Left border to left CPU
        for (int i = 1; i <= B; i++)
        {
            out[i - 1] = un(localmin + i - 1 - B);
        }
        MPI_Send(out, 2, MPI_DOUBLE, neighbours[LEFT], 17, cartcomm);
        MPI_Recv(in, 2, MPI_DOUBLE, neighbours[RIGHT], 17, cartcomm, &status);
        MPI_Barrier(cartcomm);
        for (int i = 1; i <= B; i++)
        {
            un(localmax + i - 1) = in[i - 1];
        }
    };


    if (rank==0) {
        ducklist.push_back(Duck(0.0,0.0, rank));
    }

    auto send_duck = [&]() {
        if (ducksToTransfer>0) {
            outduck[0] = ducklist[0].x;
            outduck[1] = ducklist[0].v;
            ducklist.erase(ducklist.begin());
            MPI_Send(outduck, 2, MPI_DOUBLE, neighbours[RIGHT], 6, cartcomm);
            ducksToTransfer = 0;
        } else if (ducksToReceive>0) {
            MPI_Recv(induck, 2, MPI_DOUBLE, neighbours[LEFT], 6, cartcomm, &status);
            ducklist.push_back(Duck(induck[0], induck[1], rank));
            ducksToReceive = 0;
        }
    };

    auto send_duck_number = [&]() {
        MPI_Send(&ducksToTransfer, 1, MPI_INT, neighbours[RIGHT], 37, cartcomm);
        MPI_Recv(&ducksToReceive, 1, MPI_INT, neighbours[LEFT], 37, cartcomm, &status);
        MPI_Barrier(cartcomm);
    };
    
    send_data();

    // Copy un to unp
    unp = un;

    // loop
    for (int timestep = 0; timestep < tend; timestep++)
    {   
        
        send_data();

        if (timestep % outputstep == 0)
        {
            stringstream filename;
            ofstream file;
            filename << "./data/out_" << timestep << "_CPU_" << rank << ".dat";
            file.open(filename.str().c_str());
            for (int i = localmin; i < localmax; i++)
                file << i << '\t' << unp(i) << endl;
            file.close();
            
            if (ducklist.size() > 0) {    
                stringstream filename2;
                ofstream file2;
                filename2 << "./data/duck_" << timestep << ".dat";
                file2.open(filename2.str().c_str());
                file2 << ducklist[0].x << '\t' << ducklist[0].v << endl;
                file2.close();
            }
            //printf("Rank: %i, Transfer: %i, Receive: %i\nMin: %i, Max: %i, Position: %f\n", rank, ducksToTransfer, ducksToReceive, localmin, localmax, duck.x);// duck.current, duck.x, duck.v);

        }

        // Field update
        for (int i = localmin; i < localmax; i++)
            unp(i) = un(i) - dt * ((un(i) * un(i) - un(i - 1) * un(i - 1)) / dx / 2.);

        for (Duck &duck: ducklist) {
            //printf("Rank %i, Duck %f\n", rank, duck.x);
            if (duck.x >= localmax) {
             ducksToTransfer=1; 
            }
        } 
        send_duck_number();

        for (Duck &duck: ducklist) {
            duck.advance(un);
        } 
        // if (duck.x >= localmax) {
        //     ducksToTransfer=1;
        //     //duck.current++;
        // }
        //send_duck_number();

        send_duck(); //Funktioniert leider nicht :(

        // if (rank==duck.current) duck.advance(un);
        //duck.advance(unp);

        //delete duck;
        un = unp;

        MPI_Barrier(cartcomm);
        //(Here we have to exchange borders again \dots)
    }
    MPI_Finalize();
}