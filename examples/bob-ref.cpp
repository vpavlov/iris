#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define iris_real double

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

bool read_frame0(const char *dirname, input_t *out_input)
{
    int in_client_size = 1;
    int in_client_rank = 0;
    
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

    out_input->qtot = qtot;
    out_input->qtot2 = qtot2;
    free(line);
    fclose(fp);
    return true;
}

int main()
{
    input_t input;
    read_frame0("../examples/bob-trj", &input);
    iris_real fx_tot = 0.0;
    iris_real fy_tot = 0.0;
    iris_real fz_tot = 0.0;
    iris_real ener = 0.0;
    for(int i=0;i<input.natoms;i++) {
	iris_real fx = 0.0;
	iris_real fy = 0.0;
	iris_real fz = 0.0;

	iris_real rx = input.atoms[i].xyzqi[0];
	iris_real ry = input.atoms[i].xyzqi[1];
	iris_real rz = input.atoms[i].xyzqi[2];
	iris_real q = input.atoms[i].xyzqi[3];
	
	for(int j=0;j<input.natoms;j++) {
	    if(j == i) {
		continue;
	    }
	    iris_real rix = input.atoms[j].xyzqi[0];
	    iris_real riy = input.atoms[j].xyzqi[1];
	    iris_real riz = input.atoms[j].xyzqi[2];
	    iris_real qi = input.atoms[j].xyzqi[3];

	    iris_real dx = rx - rix;
	    iris_real dy = ry - riy;
	    iris_real dz = rz - riz;

	    iris_real ri2 = dx*dx + dy*dy + dz*dz;
	    iris_real ri = sqrt(ri2);

	    iris_real e = q * qi / ri;
	    iris_real ee = e / ri2;

	    fx += ee * dx;
	    fy += ee * dy;
	    fz += ee * dz;
	    ener += e;
	}
	printf("F[%d] = (%f, %f, %f)\n", i, fx, fy, fz);
	fx_tot += fx;
	fy_tot += fy;
	fz_tot += fz;
    }
    printf("Ftot = (%f, %f, %f)\n", fx_tot, fy_tot, fz_tot);
    printf("Ener = %f\n", ener*0.5);
}
