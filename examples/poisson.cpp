#include <math.h>
#include <iris/iris.h>
#include <iris/mesh.h>

#define M 5
#define N 5
#define P 5

using namespace ORG_NCSA_IRIS;

iris_real example1_rhs(iris *iris, int i, int j, int k)
{
    if(i==0) {
	return 0.75;
    }

    if(i==1) {
	return 7.5;
    }

    if(i==2) {
	return 0.0;
    }

    if(i==3) {
	return -7.5;
    }

    if(i==4) {
	return -0.75;
    }
}

iris_real example2_rhs(iris *iris, int i, int j, int k)
{
    if(i==M/2-1) {
    	return 1.0;
    }else if(i==M/2+1) {
    	return -1.0;
    }else {
    	return 0.0;
    }
}

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
    x->set_rhs(example2_rhs);
    x->m_mesh->dump_rho();
    x->solve();
    x->m_mesh->dump_phi();
    delete x;

    MPI_Finalize();
}
