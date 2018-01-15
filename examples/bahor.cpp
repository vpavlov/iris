#include <new>
#include <vector>
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
#include <iris/comm_rec.h>

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

void read_nacl(char *fname, iris_real **&atoms)
{

    FILE *fp = fopen(fname, "r");
    if(fp == NULL) {
	printf("Cannot open NaCl.data!");
	return;
    }

    memory::create_2d(atoms, 27000, 4);

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
	    atoms[atom_id-1][3] = charge;
	}
	delete [] tokens;

	// read coords
	getline(&line, &sz, fp);
	tokens = split(line, 3);
	atoms[atom_id-1][0] = (iris_real) atof(tokens[0]);
	atoms[atom_id-1][1] = (iris_real) atof(tokens[1]);
	atoms[atom_id-1][2] = (iris_real) atof(tokens[2]);
	delete [] tokens;

	getline(&line, &sz, fp);  // skip next two lines
	getline(&line, &sz, fp);
    }

    free(line);
    fclose(fp);
}

void read_atoms(iris *in_iris, char *fname, int rank, int pp_size,
		MPI_Comm local_comm,
		iris_real **&out_my_atoms, int &out_my_count)
{
    if(rank == 0) {

	// the global box; for simplicity, hardcoded in this example
	box_t<iris_real> gb {
	    -50.39064, -50.39064, -50.39064,
		50.39064,  50.39064,  50.39064,
		100.78128, 100.78128, 100.78128};

	// read the atoms from the input file
	iris_real **atoms;
	read_nacl(fname, atoms);

	// "domain decomposition" in the client. In this example, we use a
	// simple domain decomposition in X direction: each client proc gets a
	// strip in X direction and contains all atoms in the YZ planes that
	// fall into that strip. This is obviously not the proper way to do it,
	// but since this is outside IRIS and in the domain of the client, it
	// is supposed that a proper MD package will do its own DD
	iris_real *xmin;
	iris_real *xmax;
	memory::create_1d(xmin, pp_size);
	memory::create_1d(xmax, pp_size);
	for(int i=0;i<pp_size;i++) {
	    iris_real xsplit1 = i * 1.0 / pp_size;
	    iris_real xsplit2 = (i+1) * 1.0 / pp_size;
	    xmin[i] = gb.xlo + gb.xsize * xsplit1;
	    if(i < pp_size - 1) {
		xmax[i] = gb.xlo + gb.xsize * xsplit2;
	    }else {
		xmax[i] = gb.xhi;
	    }
	}
	
	// Figure out which atoms go to which client processor. Again, this
	// is maybe not the best way to do it, but this is outside IRIS and the
	// client MD code should have already have mechanisms for this in place.
	std::vector<int> *vatoms = new std::vector<int>[pp_size];
	for(int i=0;i<27000;i++) {
	    for(int j=0;j<pp_size;j++) {
		if(atoms[i][0] >= xmin[j] && atoms[i][0] < xmax[j]) {
		    vatoms[j].push_back(i);
		    break;
		}
	    }
	}
	
	memory::destroy_1d(xmin);
	memory::destroy_1d(xmax);
	
	// Package and send the atoms for each target client node
	for(int i=0;i<pp_size;i++) {
	    iris_real **sendbuf;
	    memory::create_2d(sendbuf, vatoms[i].size(), 4);
	    int j = 0;
	    for(auto it = vatoms[i].begin(); it != vatoms[i].end(); it++) {
		sendbuf[j][0] = atoms[*it][0];
		sendbuf[j][1] = atoms[*it][1];
		sendbuf[j][2] = atoms[*it][2];
		sendbuf[j][3] = atoms[*it][3];
		j++;
	    }

	    if(i != 0) {
		MPI_Send(&(sendbuf[0][0]), 4*vatoms[i].size(), IRIS_REAL,
			 i, 1, local_comm);
	    }else {
		memory::create_2d(out_my_atoms, vatoms[i].size(), 4);
		memcpy(&(out_my_atoms[0][0]), &(sendbuf[0][0]),
		       4*vatoms[i].size());
		out_my_count = vatoms[i].size();
	    }
	    memory::destroy_2d(sendbuf);
	}

	delete [] vatoms;
	memory::destroy_2d(atoms);
	    
    }else {
	MPI_Status status;
	MPI_Probe(0, 1, local_comm, &status);
	int nreals;
	MPI_Get_count(&status, IRIS_REAL, &nreals);
	int natoms = nreals / 4;
	memory::create_2d(out_my_atoms, natoms, 4);
	MPI_Recv(&(out_my_atoms[0][0]), nreals, IRIS_REAL, 0, 1, local_comm,
		 MPI_STATUS_IGNORE);
	out_my_count = natoms;
    }
}

void send_atoms(iris *in_iris, iris_real **in_my_atoms, size_t in_my_count,
		box_t<iris_real> *in_local_boxes, int in_server_size)
{
    iris_real ***scratch;
    size_t *counts;

    memory::create_3d(scratch, in_server_size, in_my_count, 4);
    memory::create_1d(counts, in_server_size);

    for(int j=0;j<in_server_size;j++) {
	counts[j] = 0;
    }
    
    for(int i=0;i<in_my_count;i++) {
	iris_real x = in_my_atoms[i][0];
	iris_real y = in_my_atoms[i][1];
	iris_real z = in_my_atoms[i][2];
	iris_real q = in_my_atoms[i][3];

	for(int j=0;j<in_server_size;j++) {
	    int x0 = in_local_boxes[j].xlo;
	    int y0 = in_local_boxes[j].ylo;
	    int z0 = in_local_boxes[j].zlo;
	    int x1 = in_local_boxes[j].xhi;
	    int y1 = in_local_boxes[j].yhi;
	    int z1 = in_local_boxes[j].zhi;

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

    MPI_Request *reqs1 = new MPI_Request[in_server_size];

    for(int i=0;i<in_server_size;i++) {
	in_iris->m_inter_comm->send_event(&(scratch[i][0][0]),
					  counts[i] * 4 * sizeof(iris_real),
					  IRIS_EVENT_ATOMS, i);
    }

    //MPI_Waitall(in_server_size, reqs1, MPI_STATUSES_IGNORE);
    memory::destroy_3d(scratch);
    memory::destroy_1d(counts);


    // MPI_Barrier(local_comm);

    // if(rank == 0) {
    // 	int dummy = 0;
    // 	for(int i=0;i<in_server_size;i++) {
    // 	    MPI_Send(&dummy, 1, MPI_INT,
    // 		     i + iris_offset, IRIS_EVENT_ATOMS_EOF, MPI_COMM_WORLD);
    // 	}
    // }

}


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

    int rank, size, required = MPI_THREAD_MULTIPLE, provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    printf("required = %d, provided = %d\n", required, provided);
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


    if(mode == 0) {
	// In mode 0, all nodes are both client and server.
	// Thus client_size  = size and local_comm is just MPI_COMM_WORLD
	client_size = size;
	server_size = size;
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
    iris_real **my_atoms = NULL;
    int my_count = 0;
    if(x->is_client()) {
	read_atoms(x, fname, rank, client_size, local_comm, my_atoms, my_count);
	x->m_logger->trace("Client has %d atoms", my_count);
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


    // The client needs to know the domain decomposition of the server nodes
    // so it can know which client node send which atoms to which server node.
    // So, each client node must ask all server nodes about their local boxes.
    // This must be done once after each commit.
    //
    // However, to minimize communication, only the client leader gets the 
    // server procs' local boxes and must then distribute them by whatever
    // means it sees fit.
    // 
    // In mode 0 this is not really needed, but it still works. No communication
    // is done in that case, since client and server leader are the same proc.
    box_t<iris_real> *local_boxes = x->get_local_boxes(server_size);

    // Main simulation loop in the client
    if(x->is_client()) {
	// On each step...
	for(int i=0;i<NSTEPS;i++) {

	    // The client must send the atoms which befall into the server
	    // procs' local boxes to the corrseponding server node.
	    send_atoms(x, my_atoms, my_count, local_boxes, server_size);
	}
    }

    MPI_Barrier(MPI_COMM_WORLD);

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
