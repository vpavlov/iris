#include <new>
#include <exception>
#include <stdio.h>
#include <mpi.h>
#include <iris/real.h>
#include <iris/memory.h>
#include <iris/iris.h>
#include <iris/domain.h>
#include <iris/utils.h>
#include <iris/event_codes.h>

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

void __read_atoms(char *fname, int rank, int pp_size, MPI_Comm mycomm,
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
		0, mycomm);

    MPI_Scatter(fromq, 27000/pp_size, IRIS_REAL,
		&(my_q[0]), 27000/pp_size, IRIS_REAL,
		0, mycomm);
    
    if(rank == 0) {
	memory::destroy_2d(x);
	memory::destroy_1d(q);
    }
}

void __send_atoms(MPI_Comm mycomm, int rank,
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

    MPI_Request *reqs = new MPI_Request[nboxes];

    for(int i=0;i<nboxes;i++) {
	MPI_Isend(&(scratch[i][0][0]), counts[i] * 4, IRIS_REAL,
		  i + iris_offset, IRIS_EVENT_ATOMS, MPI_COMM_WORLD,
		  &reqs[i]);
    }

    MPI_Waitall(nboxes, reqs, MPI_STATUSES_IGNORE);
    memory::destroy_3d(scratch);
    memory::destroy_1d(counts);

    MPI_Barrier(mycomm);

    if(rank == 0) {
	int dummy = 0;
	for(int i=0;i<nboxes;i++) {
	    MPI_Send(&dummy, 1, MPI_INT,
		     i + iris_offset, IRIS_EVENT_ATOMS_EOF, MPI_COMM_WORLD);
	}
    }

}

main(int argc, char **argv)
{
    if(argc < 2) {
	printf("Usage: %s <path-to-NaCl.data>\n", argv[0]);
	exit(-1);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm mycomm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int pp_size = size/2;
    int duty = (rank < pp_size)?1:2;
    int iris_size = size - pp_size;
    MPI_Comm_split(MPI_COMM_WORLD, duty, rank, &mycomm);
    
    if(duty == 2) {
	// IRIS nodes
	iris *x = new iris(MPI_COMM_WORLD, mycomm, 0);
	x->domain_set_box(-50.39064, -50.39064, -50.39064,
			   50.39064,  50.39064,  50.39064);
	x->mesh_set_size(128, 128, 128);
	x->apply_conf();

	// timesteps
	for(int i=0;i<NSTEPS;i++) {
	    x->run();
	}

	delete x;
    }else {
	// PP nodes
	iris_real **my_x;
	iris_real *my_q;
	iris_real *local_boxes;
	size_t natoms = 27000/pp_size;

	__read_atoms(argv[1], rank, pp_size, mycomm, my_x, my_q);
	iris::recv_local_boxes(iris_size, rank, 0, MPI_COMM_WORLD, mycomm,
			       local_boxes);

	for(int i=0;i<NSTEPS;i++) {
	    __send_atoms(mycomm, rank, my_x, my_q, natoms, local_boxes, iris_size, pp_size);  // this is to be done for every timestep
	}

	memory::destroy_2d(my_x);
	memory::destroy_1d(my_q);
	memory::destroy_1d(local_boxes);
    }

    MPI_Finalize();
}
