#include <mpi.h>
#include <iris/iris.h>
#include <iris/haloex.h>
#include <memory.h>

using namespace ORG_NCSA_IRIS;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm cart_comm;
  int pbc[] = { 1, 1, 1 };
  int m_size[] = {2, 1, 1};
  int m_hood[3][2];
  int n[3] = {32,32,32};
  MPI_Cart_create(MPI_COMM_WORLD, 3, m_size, pbc, 0, &cart_comm);

  MPI_Cart_shift(cart_comm, 0, -1, &m_hood[0][0], &m_hood[0][1]);
  MPI_Cart_shift(cart_comm, 1, -1, &m_hood[1][0], &m_hood[1][1]);
  MPI_Cart_shift(cart_comm, 2, -1, &m_hood[2][0], &m_hood[2][1]);

  iris_real*** data;
  memory::create_3d(data,n[0],n[1],n[2]);
  haloex he(MPI_COMM_WORLD,&m_hood[0][0],0,data,&n[0], 2,2,77);
  he.exch_full();
  memory::destroy_3d(data);
  MPI_Finalize();
}
