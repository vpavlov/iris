#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <iris/iris.h>
#include <iris/memory.h>
#include <iris/logger.h>
#include <iris/mesh.h>

#define NSTEPS 1

#define M 128
#define N 128
#define P 128

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

void read_nacl(char *fname, iris_real **&charges)
{

    FILE *fp = fopen(fname, "r");
    if(fp == NULL) {
	printf("Cannot open NaCl.data!");
	return;
    }

    memory::create_2d(charges, 27000, 4);

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
	    charges[atom_id-1][3] = charge;
	}
	delete [] tokens;

	// read coords
	getline(&line, &sz, fp);
	tokens = split(line, 3);
	charges[atom_id-1][0] = (iris_real) atof(tokens[0]);
	charges[atom_id-1][1] = (iris_real) atof(tokens[1]);
	charges[atom_id-1][2] = (iris_real) atof(tokens[2]);
	delete [] tokens;

	getline(&line, &sz, fp);  // skip next two lines
	getline(&line, &sz, fp);
    }

    free(line);
    fclose(fp);
}

void read_charges(iris *in_iris, char *fname, int rank, int pp_size,
		  MPI_Comm local_comm,
		  iris_real **&out_my_charges, int &out_my_count)
{
    if(rank == 0) {

	// the global box; for simplicity, hardcoded in this example
	box_t<iris_real> gb {
	    -50.39064, -50.39064, -50.39064,
		50.39064,  50.39064,  50.39064,
		100.78128, 100.78128, 100.78128};

	// read the charges from the input file
	iris_real **charges;
	read_nacl(fname, charges);


	// "domain decomposition" in the client. In this example, we use a
	// simple domain decomposition in X direction: each client proc gets a
	// strip in X direction and contains all charges in the YZ planes that
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

	// Figure out which charges go to which client processor. Again, this
	// is maybe not the best way to do it, but this is outside IRIS and the
	// client MD code should have already have mechanisms for this in place.
	std::vector<int> *vcharges = new std::vector<int>[pp_size];
	for(int i=0;i<27000;i++) {
	    for(int j=0;j<pp_size;j++) {
		if(charges[i][0] >= xmin[j] && charges[i][0] < xmax[j]) {
		    vcharges[j].push_back(i);
		    break;
		}
	    }
	}
	
	memory::destroy_1d(xmin);
	memory::destroy_1d(xmax);

	// Package and send the charges for each target client node
	for(int i=0;i<pp_size;i++) {
	    iris_real **sendbuf;
	    memory::create_2d(sendbuf, vcharges[i].size(), 4);
	    int j = 0;
	    for(auto it = vcharges[i].begin(); it != vcharges[i].end(); it++) {
		sendbuf[j][0] = charges[*it][0];
		sendbuf[j][1] = charges[*it][1];
		sendbuf[j][2] = charges[*it][2];
		sendbuf[j][3] = charges[*it][3];
		j++;
	    }

	    if(i != 0) {
		MPI_Send(&(sendbuf[0][0]), 4*vcharges[i].size(), IRIS_REAL,
			 i, 1, local_comm);
	    }else {
		memory::create_2d(out_my_charges, vcharges[i].size(), 4);
		memcpy(&(out_my_charges[0][0]), &(sendbuf[0][0]),
		       4*vcharges[i].size()*sizeof(iris_real));
		out_my_count = vcharges[i].size();
	    }
	    memory::destroy_2d(sendbuf);
	}

	delete [] vcharges;
	memory::destroy_2d(charges);
	    
    }else {
	MPI_Status status;
	MPI_Probe(0, 1, local_comm, &status);
	int nreals;
	MPI_Get_count(&status, IRIS_REAL, &nreals);
	int ncharges = nreals / 4;
	memory::create_2d(out_my_charges, ncharges, 4);
	MPI_Recv(&(out_my_charges[0][0]), nreals, IRIS_REAL, 0, 1, local_comm,
		 MPI_STATUS_IGNORE);
	out_my_count = ncharges;
    }
}


// Find out which charges on this client node belong to which server node.
// This implementation here is just an example and is not optimal. But this is
// outside IRIS's responsibility and is provided here only as means to execute
// the example.
void send_charges(iris *in_iris, iris_real **in_my_charges, size_t in_my_count,
		  box_t<iris_real> *in_local_boxes)
{
    iris_real *sendbuf = (iris_real *)memory::wmalloc(in_my_count * 4 * sizeof(iris_real));
    for(int i=0;i<in_iris->m_server_size;i++) {

	// get the sever local box
	iris_real x0 = in_local_boxes[i].xlo;
	iris_real y0 = in_local_boxes[i].ylo;
	iris_real z0 = in_local_boxes[i].zlo;
	iris_real x1 = in_local_boxes[i].xhi;
	iris_real y1 = in_local_boxes[i].yhi;
	iris_real z1 = in_local_boxes[i].zhi;

	// find those charges that reside in this box
	// Here -- non-optimal; once a charge is assigned to a server, there is
	// no need to go through it again...
	int idx = 0;
	for(int j=0;j<in_my_count;j++) {
	    iris_real x = in_my_charges[j][0];
	    iris_real y = in_my_charges[j][1];
	    iris_real z = in_my_charges[j][2];
	    iris_real q = in_my_charges[j][3];

	    if(x >= x0 && x < x1 &&
	       y >= y0 && y < y1 &&
	       z >= z0 && z < z1)
	    {
		sendbuf[idx++] = x;
		sendbuf[idx++] = y;
		sendbuf[idx++] = z;
		sendbuf[idx++] = q;
	    }
	}

	in_iris->broadcast_charges(i, sendbuf, idx/4);	
    }

    memory::wfree(sendbuf);
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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // debugging facility
    bool ready = false;
    // if(rank == 0) {
    // 	printf("Rank %d is PID %d\n", rank, getpid());
    // 	while (!ready) {
    // 	    sleep(5);
    // 	}
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
	x->set_grid_pref(0, 1, 1);  // to match our X-based domain decomposition
    }else if(mode == 1) {
	// split the world communicator in two groups:
	// - client group: the one that "uses" IRIS. It provides charge coords
	//                 and values to IRIS and receives forces, energies,
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


	x = new iris(client_size, server_size, role, local_comm,
		     MPI_COMM_WORLD, remote_leader);
    }else {
	printf("Unknown mode. Only 0 and 1 are supported\n");
	exit(-1);
    }


    // Client nodes must have somehow aquired knowledge about charges. In this
    // example, we read them from a DL_POLY CONFIG file. For demonstration
    // purposes we split the charges between the client processors in a 
    // straightforward way: first 27000/client_size charges go to the first
    // client proc, second -- to the second proc, etc. This is obviously not
    // optimal or anything, but this is not IRIS's responsibility -- the client
    // must have already done some kind of domain decomposition and scattering
    // of the charges in a way that it thinks is optimal for it.
    // So this part would have usually been done already in some other way.
    iris_real **my_charges = NULL;
    int my_count = 0;
    if(x->is_client()) {
	read_charges(x, fname, rank, client_size, local_comm, my_charges, my_count);
	x->m_logger->trace("Client has %d charges", my_count);
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
    x->set_mesh_size(M, N, P);
    x->set_order(3);
    x->set_laplacian(IRIS_LAPL_STYLE_TAYLOR, 4);
    x->commit();


    // The client needs to know the domain decomposition of the server
    // nodes so it can know which client node send which charges to which
    // server node. So, each client node must ask all server nodes about
    // their local boxes. This must be done once after each commit
    // (e.g. after global box changed)
    //
    // In shared mode this is not really needed, but it still works.
    // No communication between the groups is done in that case, since
    // client and server leader are the same proc.
    //
    // This must be called on both clients and servers collectively.
    box_t<iris_real> *local_boxes = x->get_local_boxes();


    // Main simulation loop in the clients
    if(x->is_client()) {

	// On each step...
	for(int i=0;i<NSTEPS;i++) {
	    // The client must send the charges which befall into the server
	    // procs' local boxes to the corrseponding server node.
	    // It finds out which client sends which charges to which server
	    // by whatever means it wants, and at the end it calls IRIS's
	    // broadcast_charges()
	    send_charges(x, my_charges, my_count, local_boxes);

	    // Let the servers know that there are no more charges, so it can
	    // go on and start calculating
	    x->commit_charges();
	}

	x->quit();  // this will break server loop

    }else if(x->is_server()) {
	// Meanwhile, on the servers: an endless loop that is broken by
	// the quit() above
	x->run();
    }

    if(x->is_server()) {
	//x->m_mesh->dump_log("RHO", x->m_mesh->m_rho);
	x->m_mesh->check_exyz();
	x->m_mesh->dump_exyz("field");
    }

    // Cleanup
    memory::wfree(local_boxes);
    delete x;
    MPI_Finalize();
}
