#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iris/iris.h>
#include <iris/memory.h>
#include <iris/logger.h>
#include <iris/mesh.h>
#include <iris/comm_rec.h>
#include <iris/utils.h>
#include <iris/timer.h>
#include <iris/factorizer.h>

#define NSTEPS 2
#define CUTOFF 1.2  // 12 angstroms

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


struct atom_t
{
    iris_real xyzqi[5];
};
    
struct input_t
{
    int natoms;
    iris_real qtot;
    iris_real qtot2;
    iris_real box[3];
    std::vector<atom_t> atoms;
    
    input_t()
    {
	natoms = 0;
	qtot = qtot2 = 0.0;
	atoms.clear();
	box[0] = box[1] = box[2] = 0.0;
    };
};

bool read_frame0(char *dirname, comm_rec *in_local_comm, input_t *out_input)
{
    int in_client_size = in_local_comm->m_size;
    int in_client_rank = in_local_comm->m_rank;
    
    char fname[1024];
    
    snprintf(fname, 1024, "%s/bob0-ch.pdb", dirname);

    FILE *fp = fopen(fname, "r");
    if(fp == NULL) {
	return false;
    }
    
    char *line = NULL;
    size_t sz = 0;
    char tmp[80];
    iris_real qtot = 0.0;
    iris_real qtot2 = 0.0;

    out_input->natoms = 0;
    while(getline(&line, &sz, fp) != -1) {
	SUBSTR(tmp, line, 1, 6);
	if(!strcmp(tmp, "CRYST1")) {
	    SUBSTR(tmp, line, 7, 15);
	    out_input->box[0] = atof(tmp) / 10.0;

	    SUBSTR(tmp, line, 16, 24);
	    out_input->box[1] = atof(tmp) / 10.0;

	    SUBSTR(tmp, line, 25, 33);
	    out_input->box[2] = atof(tmp) / 10.0;
	}else if(!strcmp(tmp, "ATOM  ")) {
	    if(out_input->natoms % in_client_size == in_client_rank) {
		atom_t atom;
		SUBSTR(tmp, line, 31, 38);
		atom.xyzqi[0] = (iris_real) atof(tmp) / 10.0;
		
		SUBSTR(tmp, line, 39, 46);
		atom.xyzqi[1] = (iris_real) atof(tmp) / 10.0;
		
		SUBSTR(tmp, line, 47, 54);
		atom.xyzqi[2] = (iris_real) atof(tmp) / 10.0;
		
		SUBSTR(tmp, line, 55, 61);
		atom.xyzqi[3] = (iris_real) atof(tmp);

		atom.xyzqi[4] = out_input->natoms * (iris_real) 1.0;

		out_input->atoms.push_back(atom);
		qtot += atom.xyzqi[3];
		qtot2 += (atom.xyzqi[3] * atom.xyzqi[3]);
	    }
	    out_input->natoms++;
	}else if(!strcmp(tmp, "END   ")) {
	    break;
	}
    }

    MPI_Allreduce(&qtot, &(out_input->qtot), 1, IRIS_REAL, MPI_SUM, in_local_comm->m_comm);
    MPI_Allreduce(&qtot2, &(out_input->qtot2), 1, IRIS_REAL, MPI_SUM, in_local_comm->m_comm);
    free(line);
    fclose(fp);
    return true;
}

bool read_frameN(int N, char *dirname, comm_rec *in_local_comm, input_t *out_input)
{
    int in_client_size = in_local_comm->m_size;
    int in_client_rank = in_local_comm->m_rank;
    
    char fname[1024];
    
    snprintf(fname, 1024, "%s/bob%d.pdb", dirname, N);

    FILE *fp = fopen(fname, "r");
    if(fp == NULL) {
	return false;
    }
    
    char *line = NULL;
    size_t sz = 0;
    char tmp[80];
    iris_real qtot = 0.0;
    iris_real qtot2 = 0.0;

    int i = 0, j = 0;
    while(getline(&line, &sz, fp) != -1) {
	SUBSTR(tmp, line, 1, 6);
	if(!strcmp(tmp, "CRYST1")) {
	    SUBSTR(tmp, line, 7, 15);
	    out_input->box[0] = atof(tmp) / 10.0;

	    SUBSTR(tmp, line, 16, 24);
	    out_input->box[1] = atof(tmp) / 10.0;

	    SUBSTR(tmp, line, 25, 33);
	    out_input->box[2] = atof(tmp) / 10.0;
	}else if(!strcmp(tmp, "ATOM  ")) {
	    if(i % in_client_size == in_client_rank) {
		atom_t atom;
		
		SUBSTR(tmp, line, 31, 38);
		
		iris_real x = (iris_real) atof(tmp) / 10.0;
		if(x < 0.0) {
		    x += out_input->box[0];
		}
		if(x > out_input->box[0]) {
		    x -= out_input->box[0];
		}
		
		atom.xyzqi[0] = x;
		
		SUBSTR(tmp, line, 39, 46);
		
		x = (iris_real) atof(tmp) / 10.0;
		if(x < 0.0) {
		    x += out_input->box[1];
		}
		if(x > out_input->box[1]) {
		    x -= out_input->box[1];
		}
		atom.xyzqi[1] = x;
		
		SUBSTR(tmp, line, 47, 54);

		x = (iris_real) atof(tmp) / 10.0;
		if(x < 0.0) {
		    x += out_input->box[2];
		}
		if(x > out_input->box[2]) {
		    x -= out_input->box[2];
		}
		atom.xyzqi[2] = x;
		
		atom.xyzqi[3] = out_input->atoms[j].xyzqi[3];
		atom.xyzqi[4] = out_input->atoms[j].xyzqi[4];

		out_input->atoms[j++] = atom;
	    }
	    i++;
	}else if(!strcmp(tmp, "END   ")) {
	    break;
	}
    }

    free(line);
    fclose(fp);
    return true;
}

// iris_real read_charges(iris *in_iris, char *fname, int rank, int pp_size,
// 		       MPI_Comm local_comm,
// 		       iris_real **&out_my_charges, int &out_my_count)
// {
//     iris_real qtot2 = 0.0;
//     if(rank == 0) {

// 	// read the charges from the out_input file
// 	iris_real **charges;
// 	qtot2 = read_nacl(fname, charges);

// 	// the global box; for simplicity, hardcoded in this example
// 	box_t<iris_real> gb {
// 	    -g_boxx/2, -g_boxy/2, -g_boxz/2,
// 		g_boxx/2, g_boxy/2, g_boxz/2,
// 		g_boxx, g_boxy, g_boxz};

// 	// "domain decomposition" in the client. In this example, we use a
// 	// simple domain decomposition in X direction: each client proc gets a
// 	// strip in X direction and contains all charges in the YZ planes that
// 	// fall into that strip. This is obviously not the proper way to do it,
// 	// but since this is outside IRIS and in the domain of the client, it
// 	// is supposed that a proper MD package will do its own DD
// 	iris_real *xmin;
// 	iris_real *xmax;
// 	memory::create_1d(xmin, pp_size);
// 	memory::create_1d(xmax, pp_size);
// 	for(int i=0;i<pp_size;i++) {
// 	    iris_real xsplit1 = i * 1.0 / pp_size;
// 	    iris_real xsplit2 = (i+1) * 1.0 / pp_size;
// 	    xmin[i] = gb.xlo + gb.xsize * xsplit1;
// 	    if(i < pp_size - 1) {
// 		xmax[i] = gb.xlo + gb.xsize * xsplit2;
// 	    }else {
// 		xmax[i] = gb.xhi;
// 	    }
// 	}

// 	// Figure out which charges go to which client processor. Again, this
// 	// is maybe not the best way to do it, but this is outside IRIS and the
// 	// client MD code should have already have mechanisms for this in place.
// 	std::vector<int> *vcharges = new std::vector<int>[pp_size];
// 	for(int i=0;i<natoms;i++) {
// 	    for(int j=0;j<pp_size;j++) {
// 		if(charges[i][0] >= xmin[j] && charges[i][0] < xmax[j]) {
// 		    vcharges[j].push_back(i);
// 		    break;
// 		}
// 	    }
// 	}
	
// 	memory::destroy_1d(xmin);
// 	memory::destroy_1d(xmax);

// 	// Package and send the charges for each target client node
// 	for(int i=0;i<pp_size;i++) {
// 	    iris_real **sendbuf;
// 	    memory::create_2d(sendbuf, vcharges[i].size(), 5);
// 	    int j = 0;
// 	    for(auto it = vcharges[i].begin(); it != vcharges[i].end(); it++) {
// 		sendbuf[j][0] = charges[*it][0];
// 		sendbuf[j][1] = charges[*it][1];
// 		sendbuf[j][2] = charges[*it][2];
// 		sendbuf[j][3] = charges[*it][3];
// 		sendbuf[j][4] = (iris_real) *it;
// 		j++;
// 	    }

// 	    if(i != 0) {
// 		MPI_Send(&(sendbuf[0][0]), 5*vcharges[i].size(), IRIS_REAL,
// 			 i, 1, local_comm);
// 	    }else {
// 		memory::create_2d(out_my_charges, vcharges[i].size(), 5);
// 		memcpy(&(out_my_charges[0][0]), &(sendbuf[0][0]),
// 		       5*vcharges[i].size()*sizeof(iris_real));
// 		out_my_count = vcharges[i].size();
// 	    }
// 	    memory::destroy_2d(sendbuf);
// 	}

// 	delete [] vcharges;
// 	memory::destroy_2d(charges);
	    
//     }else {
// 	MPI_Status status;
// 	MPI_Probe(0, 1, local_comm, &status);
// 	int nreals;
// 	MPI_Get_count(&status, IRIS_REAL, &nreals);
// 	int ncharges = nreals / 5;
// 	memory::create_2d(out_my_charges, ncharges, 5);
// 	MPI_Recv(&(out_my_charges[0][0]), nreals, IRIS_REAL, 0, 1, local_comm,
// 		 MPI_STATUS_IGNORE);
// 	out_my_count = ncharges;
//     }

//     return qtot2;
// }


// Find out which charges on this client node belong to which server node.
// This implementation here is just an example and is not optimal. But this is
// outside IRIS's responsibility and is provided here only as means to execute
// the example.
void send_charges(iris *in_iris, input_t *in_input, box_t<iris_real> *in_local_boxes)
{
    iris_real *sendbuf = (iris_real *)memory::wmalloc(in_input->atoms.size() * sizeof(atom_t));
    for(int i=0;i<in_iris->m_server_size;i++) {
	int cnt = 0;
	for(int j=0;j<in_input->atoms.size();j++) {
	    if(in_local_boxes[i].in(in_input->atoms[j].xyzqi)) {
		memcpy(sendbuf+cnt*5, in_input->atoms[j].xyzqi, sizeof(atom_t));
		cnt++;
	    }
	}
	in_iris->send_charges(i, sendbuf, cnt);
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

int main(int argc, char **argv)
{
    if(argc != 3) {
	printf("Usage: %s <path-to-bob-trj dir> <mode>\n", argv[0]);
	printf("  mode = 0 is all nodes are client/server\n");
	printf("  mode = 1 is half nodes are clients, half nodes are server\n");
	exit(-1);
    }

    char *dirname = argv[1];
    int mode = atoi(argv[2]);
    
    char proc_name[256];
    int name_len;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Get_processor_name(proc_name, &name_len);
    proc_name[name_len] = 0;
    
    // debugging facility
    
    bool ready = false;
    if(rank == 4) {
    	printf("Rank %d is PID %d\n", rank, getpid());
    	while (!ready) {
    	    sleep(5);
    	}
    }
    MPI_Barrier(MPI_COMM_WORLD);

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
	x = new iris(IRIS_SOLVER_FMM, MPI_COMM_WORLD);
	//x->set_grid_pref(0, 1, 1);  // to match our X-based domain decomposition
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


	x = new iris(IRIS_SOLVER_FMM, client_size, server_size, role, local_comm,
		     MPI_COMM_WORLD, remote_leader);
    }else {
	printf("Unknown mode. Only 0 and 1 are supported\n");
	exit(-1);
    }

    x->set_pbc(true, true, true);
    x->set_units(md);
    x->m_logger->info("Node name = %s; PID = %d", proc_name, getpid());

    input_t input;
    if(x->is_client()) {
	if(!read_frame0(dirname, x->m_local_comm, &input)) {
	    x->m_logger->error("Cannot read input file, quitting!");
	    delete x;
	    MPI_Finalize();
	    exit(-1);
	}
	x->m_logger->info("Client %d has %d charges", rank, input.atoms.size());
    }

    // rank 0 is client, so it can broadcast its header to all in world, including servers
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&(input.natoms), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input.qtot), 5, IRIS_REAL, 0, MPI_COMM_WORLD);

    x->m_logger->info("Total number of atoms = %d", input.natoms);
    x->m_logger->info("Q = %f", input.qtot);
    x->m_logger->info("Q2 = %f", input.qtot2);

    // Setup IRIS. Although called by both client and server nodes, only
    // nodes for which this information is meaningful will do something with it
    // others will just noop.
    //
    // At the end of the configuration, call commit in order to apply all
    // the configuration and make the IRIS server nodes perform any preliminary
    // calculations in order to prepare for the calculation proper.
    x->config_auto_tune(input.natoms, input.qtot2, CUTOFF);
    
    solver_param_t nsigmas;
    nsigmas.r = 6.0;
    x->set_solver_param(IRIS_SOLVER_CG_NSIGMAS, nsigmas);

    solver_param_t pade;
    pade.i = 0;
    x->set_solver_param(IRIS_SOLVER_CG_STENCIL_PADE_M, pade);

    pade.i = 2;
    x->set_solver_param(IRIS_SOLVER_CG_STENCIL_PADE_N, pade);
    
    x->set_order(4);
    x->set_mesh_size(128, 128, 128);
    x->set_alpha(2.6028443952840625);
    x->set_accuracy(1e-4, true);

    for(int i=1;i<=NSTEPS;i++) {
	if (x->is_client()) {
	    box_t<iris_real> gbox;
	    gbox.xlo = gbox.ylo = gbox.zlo = 0.0;
	    gbox.xhi = input.box[0];
	    gbox.yhi = input.box[1];
	    gbox.zhi = input.box[2];
	    x->set_global_box(&gbox);
	    x->commit();
	    box_t<iris_real> *local_boxes = x->get_local_boxes();
	    
	    int *nforces;
	    send_charges(x, &input, local_boxes);
	    x->commit_charges();
	    
	    iris_real Ek, Es, Ecorr;
	    iris_real virial[6];
	    iris_real *forces = x->receive_forces(&nforces, &Ek, virial);
	    x->m_logger->info("Ek(partial) = %f [%s]", Ek, x->m_units->energy_unit);
	    x->m_logger->info("Virial[0] = %f", virial[0]);
	    x->m_logger->info("Virial[1] = %f", virial[1]);
	    x->m_logger->info("Virial[2] = %f", virial[2]);
	    x->m_logger->info("Virial[3] = %f", virial[3]);
	    x->m_logger->info("Virial[4] = %f", virial[4]);
	    x->m_logger->info("Virial[5] = %f", virial[5]);
	    x->get_global_energy(&Ek, &Es, &Ecorr);
	    x->m_logger->info("E(total) = %f (%f, %f, %f) [%s]", Ek + Es + Ecorr, Ek, Es, Ecorr, x->m_units->energy_unit);

	    handle_forces(x, nforces, forces);
	    delete [] nforces;
	    memory::wfree(forces);

	    // the clients don't have a mesh; this needs to be moved to the server
	    // x->m_mesh->dump_ascii("bob-rho", x->m_mesh->m_rho);
	    // x->m_mesh->dump_ascii("bob-phi", x->m_mesh->m_phi);
	    
	    read_frameN(i, dirname, x->m_local_comm, &input);
	}else {
	    x->run();
	}
    }

    if(x->is_client()) {
	x->quit();
    }
    
    MPI_Finalize();
    exit(0);
    
}
