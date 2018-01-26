#include <iris/iris.h>

#define M 8
#define N 8
#define P 8

using namespace ORG_NCSA_IRIS;

main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // debugging facility
    // bool ready = false;
    // if(rank == 3) {
    // 	printf("Rank %d is PID %d\n", rank, getpid());
    // 	while (!ready) {
    // 	    sleep(5);
    // 	}
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    iris *x = new iris(MPI_COMM_WORLD);

    x->set_global_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    x->set_mesh_size(M, N, P);
    x->set_taylor_stencil(2);
    x->commit();
    x->rhs_fn(example1_rhs);
    x->solve();

    delete x;
    MPI_Finalize();
}
