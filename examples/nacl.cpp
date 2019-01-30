#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <iris/iris.h>
#include <iris/memory.h>
#include <iris/logger.h>
#include <iris/mesh.h>
#include <iris/comm_rec.h>
#include <iris/utils.h>

iris_real g_boxx, g_boxy, g_boxz;

#define NATOMS 27000
#define CUTOFF 10  // 10 angstroms

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

#define SUBSTR(DEST, SRC, START, END) \
    memcpy((DEST), ((SRC)+(START)-1), ((END)-(START)+1));	\
    (DEST)[((END)-(START)+1)] = '\0';


iris_real read_nacl(char *fname, iris_real **&charges)
{

    FILE *fp = fopen(fname, "r");
    if(fp == NULL) {
	printf("Cannot open data file!");
	return 0.0;
    }

    memory::create_2d(charges, NATOMS, 4);

    char *line = NULL;
    size_t sz = 0;

    int count = 0;
    char tmp[80];
    iris_real qtot2 = 0.0;

    while(getline(&line, &sz, fp) != -1) {
	iris_real charge = 0.0;
	int atom_id = 0;
	SUBSTR(tmp, line, 1, 6);
	if(!strcmp(tmp, "CRYST1")) {
	    SUBSTR(tmp, line, 7, 15);
	    g_boxx = atof(tmp);

	    SUBSTR(tmp, line, 16, 24);
	    g_boxy = atof(tmp);

	    SUBSTR(tmp, line, 25, 33);
	    g_boxz = atof(tmp);

	}else if(!strcmp(tmp, "ATOM  ")) {
	    SUBSTR(tmp, line, 7, 11);
	    atom_id = atoi(tmp);

	    SUBSTR(tmp, line, 31, 38);
	    charges[atom_id-1][0] = (iris_real) atof(tmp);

	    SUBSTR(tmp, line, 39, 46);
	    charges[atom_id-1][1] = (iris_real) atof(tmp);

	    SUBSTR(tmp, line, 47, 54);
	    charges[atom_id-1][2] = (iris_real) atof(tmp);

	    SUBSTR(tmp, line, 13, 16);
	    if(!strcmp(tmp, " Na+")) {
		charge = (iris_real)1.0;
	    }else if(!strcmp(tmp, " Cl-")) {
		charge = (iris_real)-1.0;
	    }
	    charges[atom_id-1][3] = charge;
	    qtot2 += (charge * charge);
	}else if(!strcmp(tmp, "END   ")) {
	    break;
	}
    }

    free(line);
    fclose(fp);
    return qtot2;
}

iris_real read_charges(iris *in_iris, char *fname, int rank, int pp_size,
		       MPI_Comm local_comm,
		       iris_real **&out_my_charges, int &out_my_count)
{
    iris_real qtot2 = 0.0;
    if(rank == 0) {

	// read the charges from the input file
	iris_real **charges;
	qtot2 = read_nacl(fname, charges);

	// the global box; for simplicity, hardcoded in this example
	box_t<iris_real> gb {
	    -g_boxx/2.0, -g_boxy/2.0, -g_boxz/2.0,
		g_boxx/2.0, g_boxy/2.0, g_boxz/2.0,
		g_boxx, g_boxy, g_boxz};

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
	for(int i=0;i<NATOMS;i++) {
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
	    memory::create_2d(sendbuf, vcharges[i].size(), 5);
	    int j = 0;
	    for(auto it = vcharges[i].begin(); it != vcharges[i].end(); it++) {
		sendbuf[j][0] = charges[*it][0];
		sendbuf[j][1] = charges[*it][1];
		sendbuf[j][2] = charges[*it][2];
		sendbuf[j][3] = charges[*it][3];
		sendbuf[j][4] = (iris_real) *it;
		j++;
	    }

	    if(i != 0) {
		MPI_Send(&(sendbuf[0][0]), 5*vcharges[i].size(), IRIS_REAL,
			 i, 1, local_comm);
	    }else {
		memory::create_2d(out_my_charges, vcharges[i].size(), 5);
		memcpy(&(out_my_charges[0][0]), &(sendbuf[0][0]),
		       5*vcharges[i].size()*sizeof(iris_real));
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
	int ncharges = nreals / 5;
	memory::create_2d(out_my_charges, ncharges, 5);
	MPI_Recv(&(out_my_charges[0][0]), nreals, IRIS_REAL, 0, 1, local_comm,
		 MPI_STATUS_IGNORE);
	out_my_count = ncharges;
    }

    return qtot2;
}


// Find out which charges on this client node belong to which server node.
// This implementation here is just an example and is not optimal. But this is
// outside IRIS's responsibility and is provided here only as means to execute
// the example.
void send_charges(iris *in_iris, iris_real **in_my_charges, size_t in_my_count,
		  box_t<iris_real> *in_local_boxes)
{
    iris_real *sendbuf = (iris_real *)memory::wmalloc(in_my_count * 5 * sizeof(iris_real));
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
	    iris_real id = in_my_charges[j][4];

	    if(x >= x0 && x < x1 &&
	       y >= y0 && y < y1 &&
	       z >= z0 && z < z1)
	    {
		sendbuf[idx++] = x;
		sendbuf[idx++] = y;
		sendbuf[idx++] = z;
		sendbuf[idx++] = q;
		sendbuf[idx++] = id;
	    }
	}

	in_iris->send_charges(i, sendbuf, idx/5);
    }

    memory::wfree(sendbuf);
}

// Do whatever is needed with the forces that came back from IRIS
// In this example we just sum them up to check if they come up as 0
void handle_forces(iris *iris, int *nforces, iris_real *forces)
{
    iris_real fsum[3];
    iris_real tot_fsum[3];

    char fname[256];
    sprintf(fname, "forces%d.dat", iris->m_local_comm->m_rank);
    FILE *fp = fopen(fname, "wt");

    fsum[0] = fsum[1] = fsum[2] = 0.0;
    int n = 0;
    for(int i=0;i<iris->m_server_size;i++) {
	for(int j=0;j<nforces[i];j++) {
	    fprintf(fp, "%f %f %f\n",
		    forces[n*4 + 1], 
		    forces[n*4 + 2], 
		    forces[n*4 + 3]);

	    // forces[n*4 + 0] is the atom ID (encoded as iris_real)
	    fsum[0] += forces[n*4 + 1];
	    fsum[1] += forces[n*4 + 2];
	    fsum[2] += forces[n*4 + 3];
	    n++;
	}
    }
    MPI_Reduce(&fsum, &tot_fsum, 3, IRIS_REAL, MPI_SUM, iris->m_local_leader,
	       iris->m_local_comm->m_comm);
    if(iris->is_leader()) {
	iris->m_logger->info("Total Fsum = (%.15g, %.15g, %.15g)",
			      tot_fsum[0], tot_fsum[1], tot_fsum[2]);
    }

    fclose(fp);
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
    // straightforward way: first NATOMS/client_size charges go to the first
    // client proc, second -- to the second proc, etc. This is obviously not
    // optimal or anything, but this is not IRIS's responsibility -- the client
    // must have already done some kind of domain decomposition and scattering
    // of the charges in a way that it thinks is optimal for it.
    // So this part would have usually been done already in some other way.
    iris_real **my_charges = NULL;
    int my_count = 0;
    iris_real qtot2;
    if(x->is_client()) {
	qtot2 = read_charges(x, fname, rank, client_size, local_comm, my_charges, my_count);
	x->m_logger->trace("Client has %d charges", my_count, qtot2);
    }

    MPI_Bcast(&g_boxx, 1, IRIS_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_boxy, 1, IRIS_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_boxz, 1, IRIS_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qtot2, 1, IRIS_REAL, 0, MPI_COMM_WORLD);

    // Setup IRIS. Although called by both client and server nodes, only
    // nodes for which this information is meaningful will do something with it
    // others will just noop.
    //
    // At the end of the configuration, call commit in order to apply all
    // the configuration and make the IRIS server nodes perform any preliminary
    // calculations in order to prepare for the calculation proper.
    x->set_global_box(-g_boxx/2.0, -g_boxy/2.0, -g_boxz/2.0,
		      g_boxx/2.0,  g_boxy/2.0,  g_boxz/2.0);
    x->config_auto_tune(NATOMS, qtot2, CUTOFF);
    x->set_order(3);
    x->set_accuracy(1e-6, true);
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
	for(int i=0;i<1;i++) {
	    
	    // The client must send the charges which befall into the server
	    // procs' local boxes to the corrseponding server node.
	    // It finds out which client sends which charges to which server
	    // by whatever means it wants, and at the end it calls IRIS's
	    // broadcast_charges()
	    send_charges(x, my_charges, my_count, local_boxes);


	    // Let the servers know that there are no more charges, so it can
	    // go on and start calculating
	    x->commit_charges();


	    // Receive back the forces from the server
	    int *nforces;
	    iris_real *forces = x->receive_forces(&nforces);
	    

	    handle_forces(x, nforces, forces);	    

	    iris_real etot = x->global_energy();
	    x->m_logger->info("Total long-range energy = %f [%s]", etot, x->m_units->energy_unit);

	    delete [] nforces;
	    memory::wfree(forces);
	}

	x->quit();  // this will break server loop

    }else if(x->is_server()) {
	// Meanwhile, on the servers: an endless loop that is broken by
	// the quit() above
	x->run();
    }

    if(x->is_server()) {
	//x->m_mesh->dump_log("RHO", x->m_mesh->m_rho);
	//x->m_mesh->check_fxyz();
	//x->m_mesh->dump_exyz("field");
    }

    // Cleanup
    memory::wfree(local_boxes);
    delete x;
    MPI_Finalize();
}
