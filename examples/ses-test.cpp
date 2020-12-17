#include <time.h>
#include <stdio.h>
#include <math.h>
#include "iris/utils.h"
#include "iris/point.h"
#include "iris/sphere.h"
#include "iris/ses.h"

#define N 100

using namespace ORG_NCSA_IRIS;

iris_real rand_float(iris_real top)
{
    return (iris_real)rand()/(iris_real)(RAND_MAX/top);
}

int main()
{
    srand(time(NULL));

    sphere_t s[N];
    for(int i=0;i<3*N;i++) {
	int d = i%3;
	s[i/3].c.r[d] = rand_float((iris_real)100.0);
	s[i/3].r = rand_float((iris_real)100.0);
    }

    sphere_t ses;

    for(int i=0;i<N;i++) {
    	printf("%f %f %f (%f)\n", s[i].c.r[0], s[i].c.r[1], s[i].c.r[2], s[i].r);
    }
	
    printf("\n");
    ses_of_spheres(s, N, &ses);
	
    // for(int i=0;i<N;i++) {
    // 	point_t tt = p[i].minus(&(ses.c));
    // 	iris_real dist2 = tt.dot(&tt);
    // 	iris_real deffect = ses.r - sqrt(dist2);
    // 	if(deffect < 0) {
    // 	    printf("%d: %e\n", i, deffect);
    // 	}
    // }
    printf("Sphere is at %f %f %f (%f)\n", ses.c.r[0], ses.c.r[1], ses.c.r[2], ses.r);
}

// THIS TESTS ses_of_points

// int main()
// {
//     srand(time(NULL));

//     for(int k=0;k<100000;k++) {
// 	point_t p[N];
// 	for(int i=0;i<3*N;i++) {
// 	    int d = i%3;
// 	    p[i/3].r[d] = rand_float((iris_real)100.0);
// 	}
	
// 	point_t r[4];
// 	sphere_t ses;
	
// 	// for(int i=0;i<N;i++) {
// 	// 	printf("%f %f %f\n", p[i].r[0], p[i].r[1], p[i].r[2]);
// 	// }
	
// 	// printf("\n");
// 	ses_of_points(p, N, &ses);
	
// 	for(int i=0;i<N;i++) {
// 	    point_t tt = p[i].minus(&(ses.c));
// 	    iris_real dist2 = tt.dot(&tt);
// 	    iris_real deffect = ses.r - sqrt(dist2);
// 	    if(deffect < 0) {
// 		printf("%d: %e\n", i, deffect);
// 	    }
// 	}
// 	//printf("Sphere is at %f %f %f, radius %f\n", ses.c.r[0], ses.c.r[1], ses.c.r[2], ses.r);
//     }
// }
