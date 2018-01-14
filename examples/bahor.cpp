#include <new>
#include <exception>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <iris/real.h>
#include <iris/memory.h>
#include <iris/iris.h>
#include <iris/domain.h>
#include <iris/utils.h>
#include <iris/comm_driver.h>
#include <iris/event_queue.h>
#include <iris/logger.h>

#define NSTEPS 1

using namespace ORG_NCSA_IRIS;

char **split(char *line, int max)
{
    char *pch;
    char **tokens = new char *[max];
    int count = 0;

    pch = strtok(line, " ");
    while(pch != NULL && count < max) {
	tokens[count++] = pch;
	pch = strtok(NULL, " ");
    }

    return tokens;
}

void read_nacl(char *fname, iris_real **&x, iris_real *&q)
{

    FILE *fp = fopen(fname, "r");
    if(fp == NULL) {
	printf("Cannot open NaCl.data!");
	return;
    }

    memory::create_2d(x, 27000, 3);
    memory::create_1d(q, 27000);

    char *line = NULL;
    size_t sz = 0;

    getline(&line, &sz, fp);
    getline(&line, &sz, fp);
    getline(&line, &sz, fp);
    getline(&line, &sz, fp);
    getline(&line, &sz, fp);

    int count = 0;
    while(getline(&line, &sz, fp) != -1) {

	// process the Na+     1 line (type of atom, id of atom)
	iris_real charge = 0.0;
	int atom_id = 0;
	char **tokens = split(line, 2);
	if(!strcmp(tokens[0], "Na+")) {
	    charge = (iris_real)1.0;
	}else if(!strcmp(tokens[0], "Cl-")) {
	    charge = (iris_real)-1.0;
	}
	atom_id = atoi(tokens[1]);
	if(atom_id != 0) {
	    q[atom_id-1] = charge;
	}
	delete [] tokens;

	// read coords
	getline(&line, &sz, fp);
	tokens = split(line, 3);
	x[atom_id-1][0] = (iris_real) atof(tokens[0]);
	x[atom_id-1][1] = (iris_real) atof(tokens[1]);
	x[atom_id-1][2] = (iris_real) atof(tokens[2]);
	delete [] tokens;

	getline(&line, &sz, fp);  // skip next two lines
	getline(&line, &sz, fp);
    }

    free(line);
    fclose(fp);
}

void read_atoms(char *fname, int rank, int pp_size, MPI_Comm local_comm,
		iris_real **&my_x, iris_real *&my_q)
{
    iris_real **x;
    iris_real *q;
    
    memory::create_2d(my_x, 27000/pp_size, 3);
    memory::create_1d(my_q, 27000/pp_size);
    
    void *fromx = NULL;
    void *fromq = NULL;

    if(rank == 0) {
	read_nacl(fname, x, q);
	fromx = &(x[0][0]);
	fromq = &(q[0]);
    }
    
    MPI_Scatter(fromx, 27000*3/pp_size, IRIS_REAL,
		&(my_x[0][0]), 27000*3/pp_size, IRIS_REAL,
		0, local_comm);

    MPI_Scatter(fromq, 27000/pp_size, IRIS_REAL,
		&(my_q[0]), 27000/pp_size, IRIS_REAL,
		0, local_comm);
    
    if(rank == 0) {
	memory::destroy_2d(x);
	memory::destroy_1d(q);
    }
}
/*
void __send_atoms(MPI_Comm local_comm, int rank,
		  iris_real **my_x,
		  iris_real *my_q,
		  size_t natoms,
		  iris_real *local_boxes,
		  int nboxes,
		  int iris_offset)
{
    iris_real ***scratch;
    size_t *counts;

    memory::create_3d(scratch, nboxes, natoms, 4);
    memory::create_1d(counts, nboxes);

    for(int j=0;j<nboxes;j++) {
	counts[j] = 0;
    }
    
    for(int i=0;i<natoms;i++) {
	iris_real x = my_x[i][0];
	iris_real y = my_x[i][1];
	iris_real z = my_x[i][2];
	iris_real q = my_q[i];

	for(int j=0;j<nboxes;j++) {
	    iris_real x0, y0, z0, x1, y1, z1;
	    x0 = local_boxes[j*6 + 0];
	    y0 = local_boxes[j*6 + 1];
	    z0 = local_boxes[j*6 + 2];
	    x1 = local_boxes[j*6 + 3];
	    y1 = local_boxes[j*6 + 4];
	    z1 = local_boxes[j*6 + 5];

	    if(x >= x0 && x < x1 &&
	       y >= y0 && y < y1 &&
	       z >= z0 && z < z1)
	    {
		scratch[j][counts[j]][0] = x;
		scratch[j][counts[j]][1] = y;
		scratch[j][counts[j]][2] = z;
		scratch[j][counts[j]][3] = q;
		counts[j]++;
		break;
	    }
	}
    }

    MPI_Request *reqs1 = new MPI_Request[nboxes];
    MPI_Request *reqs2 = new MPI_Request[nboxes];

    for(int i=0;i<nboxes;i++) {
	MPI_Isend(&(scratch[i][0][0]), counts[i] * 4, IRIS_REAL,
		  i + iris_offset, IRIS_EVENT_ATOMS, MPI_COMM_WORLD,
		  &reqs1[i]);
	MPI_Irecv(NULL, 0, MPI_INT, i + iris_offset, IRIS_EVENT_ATOMS_ACK, MPI_COMM_WORLD, &reqs2[i]);
    }

    MPI_Waitall(nboxes, reqs1, MPI_STATUSES_IGNORE);
    MPI_Waitall(nboxes, reqs2, MPI_STATUSES_IGNORE);
    memory::destroy_3d(scratch);
    memory::destroy_1d(counts);

    MPI_Barrier(local_comm);

    if(rank == 0) {
	int dummy = 0;
	for(int i=0;i<nboxes;i++) {
	    MPI_Send(&dummy, 1, MPI_INT,
		     i + iris_offset, IRIS_EVENT_ATOMS_EOF, MPI_COMM_WORLD);
	}
    }

}
*/
main(int argc, char **argv)
{
    if(argc < 3) {
	printf("Usage: %s <path-to-NaCl.data> <mode>\n", argv[0]);
	printf("  mode = 0 is all nodes are client/server\n");
	printf("  mode = 1 is half nodes are clients, half nodes are server\n");
	exit(-1);
    }

    // handle arguments
    char *fname = argv[1];
    int mode = atoi(argv[2]);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    //This code here facilitates debugging with gdb
    // printf("%d has MPI rank %d\n", getpid(), rank);
    // if(rank == 0) {
    // 	getc(stdin);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    

    int role;
    iris *x;
    MPI_Comm local_comm;
    int client_size;
    int server_size;
    iris_real **my_x;
    iris_real *my_q;


    if(mode == 0) {
	// In mode 0, all nodes are both client and server.
	// Thus client_size  = size and local_comm is just MPI_COMM_WORLD
	client_size = size;
	MPI_Comm_dup(MPI_COMM_WORLD, &local_comm);

	role = IRIS_ROLE_CLIENT | IRIS_ROLE_SERVER;
	x = new iris(MPI_COMM_WORLD);
    }else if(mode == 1) {
	// split the world communicator in two groups:
	// - client group: the one that "uses" IRIS. It provides atom coords
	//                 and charges to IRIS and receives forces, energies,
	//                 etc. back
	// - server group: the processes that IRIS can use to do its
	//                 calculations of the long-range interactions
	//
	// In this example only, we decide to split the world comm in two mostly
	// equal parts. The first part is the client, the second -- the server.
	// If the world comm has odd number of procs, client will receive one
	// less proc than the server. (E.g. if nprocs = 3, we have:
	//   0 - client
	//   1 - server
	//   2 - server
	client_size = size/2;
	server_size = size - client_size;
	role = (rank < client_size)?IRIS_ROLE_CLIENT:IRIS_ROLE_SERVER;
	MPI_Comm_split(MPI_COMM_WORLD, role, rank, &local_comm);


	// figure out the remote leader
	// In this example only:
	// - the client's remote leader is server's rank 0, which = client_size
	// - the server's remote leader is client's rank 0, which is 0
	int remote_leader = (role==IRIS_ROLE_SERVER)?0:client_size;


	x = new iris(role, local_comm, MPI_COMM_WORLD, remote_leader);
    }else {
	printf("Unknown mode. Only 0 and 1 are supported\n");
	exit(-1);
    }


    // Client nodes must have somehow aquired knowledge about atoms. In this
    // example, we read them from a DL_POLY CONFIG file. For demonstration
    // purposes we split the atoms between the client processors in a 
    // straightforward way: first 27000/client_size atoms go to the first
    // client proc, second -- to the second proc, etc. This is obviously not
    // optimal or anything, but this is not IRIS's responsibility -- the client
    // must have already done some kind of domain decomposition and scattering
    // of the atoms in a way that it thinks is optimal for it.
    // So this part would have usually been done already in some other way.
    if(x->is_client()) {
	read_atoms(fname, rank, client_size, local_comm, my_x, my_q);
    }


    // Setup IRIS. Although called by both client and server nodes, only
    // nodes for which this information is meaningful will do something with it
    // others will just noop.
    //
    // At the end of the configuration, call commit in order to apply all
    // the configuration and make the IRIS server nodes perform any preliminary
    // calculations in order to prepare for the calculation proper.
    x->set_global_box(-50.39064, -50.39064, -50.39064,
    		      50.39064,  50.39064,  50.39064);
    x->set_mesh_size(128, 128, 128);
    x->set_order(3);
    x->commit();
    

    // The run() call spawns a new event looping thread on the node. The thread
    // blocks waiting for events, so it won't consume resources when there are
    // no events. The run call returns immediately, so the client nodes can 
    // continue doing whatever they need to do.
    x->run();

    // Sending atoms from client to server
    //------------------------------------
    // The client needs to know the domain decomposition of the server nodes
    // so it can know which client node send which atoms to which server node.
    // So, each client node must ask all server nodes about their local boxes.
    // This must be done once after each commit.
    //
    // Instead of doing this in all-client x all-server fashion, we do it
    // all-servers -> server-leader -> client-leader -> all-clients. This
    // greatly reduces the amount of communications needed.
    iris_real *local_boxes = x->get_local_boxes(server_size);


    // simulate some work
    sleep(1);

    // Cleanup
    delete x;
    MPI_Finalize();
    exit(0);



    // if(role == 2) {
    // 	// IRIS nodes
    // 	iris *x = new iris(MPI_COMM_WORLD, local_comm, 0);
    // 	x->set_global_box(-50.39064, -50.39064, -50.39064,
    // 		   50.39064,  50.39064,  50.39064);
    // 	x->mesh_set_size(128, 128, 128);
    // 	x->apply_conf();
    // 	x->run();
    // 	delete x;
    // }else {
    // 	// PP nodes
    // 	iris_real **my_x;
    // 	iris_real *my_q;
    // 	iris_real *local_boxes;
    // 	size_t natoms = 27000/client_size;

    // 	__read_atoms(argv[1], rank, client_size, local_comm, my_x, my_q);
    // 	int iris_size = size - client_size;
    // 	iris::recv_local_boxes(iris_size, rank, 0, MPI_COMM_WORLD, local_comm,
    // 			       local_boxes);

    // 	for(int i=0;i<NSTEPS;i++) {
    // 	    __send_atoms(local_comm, rank, my_x, my_q, natoms, local_boxes, iris_size, client_size);
    // 	}

    // 	memory::destroy_2d(my_x);
    // 	memory::destroy_1d(my_q);
    // 	memory::destroy_1d(local_boxes);
    // }

    MPI_Finalize();
}
