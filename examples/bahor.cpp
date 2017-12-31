#include <new>
#include <exception>
#include <stdio.h>
#include <mpi.h>
#include <iris/real.h>
#include <iris/memory.h>
#include <iris/iris.h>
#include <iris/domain.h>
#include <iris/utils.h>

using namespace ORG_NCSA_IRIS;

main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm mycomm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int duty = (rank < size/2)?1:2;
    int iris_size = size - size/2;

    MPI_Comm_split(MPI_COMM_WORLD, duty, rank, &mycomm);
    
    if(duty == 2) {
	// IRIS nodes
	iris *x = new iris(MPI_COMM_WORLD, mycomm, 0);
	x->domain_set_box(0.0, 0.0, 0.0, 100.78128, 100.78128, 100.78128);
	x->apply_conf();
	delete x;
    }else {
	if(rank == 0) {
	    iris_real *local_boxes;

	    iris::recv_local_boxes(MPI_COMM_WORLD, iris_size, local_boxes);
	    
	    for(int i = 0; i < iris_size; i++) {
		printf("%d: [%g %g] [%g %g] [%g %g]\n",
		       i,
		       local_boxes[i*6 + 0], local_boxes[i*6 + 3],
		       local_boxes[i*6 + 1], local_boxes[i*6 + 4],
		       local_boxes[i*6 + 2], local_boxes[i*6 + 5]);
	    }

	    memory::destroy_1d(local_boxes);
	}
    }

    MPI_Finalize();
}
