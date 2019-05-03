/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*

   Author: Marco Aldinucci.
   email:  aldinuc@di.unipi.it
   marco@pisa.quadrics.com
   date :  15/11/97

Modified by:

****************************************************************************
 *  Author: Dalvan Griebler <dalvangriebler@gmail.com>
 *  Author: Dinei Rockenbach <dinei.rockenbach@edu.pucrs.br>
 *
 *  Copyright: GNU General Public License
 *  Description: This program simply computes the mandelbroat set.
 *  File Name: mandel.cpp
 *  Version: 1.0 (25/05/2018)
 *  Compilation Command: make
 ****************************************************************************
*/


#include <stdio.h>
#if !defined(NO_DISPLAY)
#include "marX2.h"
#endif
#include <sys/time.h>
#include <math.h>

#include <iostream>
#include <chrono>


#define DIM 800
#define ITERATION 1024

double diffmsec(struct timeval  a,  struct timeval  b) {
    long sec  = (a.tv_sec  - b.tv_sec);
    long usec = (a.tv_usec - b.tv_usec);

    if(usec < 0) {
        --sec;
        usec += 1000000;
    }
    return ((double)(sec*1000)+ (double)usec/1000.0);
}



int main(int argc, char **argv) {
    double init_a=-2.125,init_b=-1.5,range=3.0;
    int dim = DIM, niter = ITERATION;
    // stats
    struct timeval t1,t2;
    int retries=1;
    double avg = 0;


    if (argc<4) {
        printf("Usage: %s size niterations retries\n\n", argv[0]);
        exit(-1);
    }
    else {
        dim = atoi(argv[1]);
        niter = atoi(argv[2]);
        retries = atoi(argv[3]);
    }
    double * runs = (double *) malloc(retries*sizeof(double));

    unsigned char *M = (unsigned char *) malloc(dim);

    double step = range/((double) dim);

#if !defined(NO_DISPLAY)
    SetupXWindows(dim,dim,1,NULL,"Sequential Mandelbroot");
#endif

    printf("bin;size;numiter;time (ms);workers;batch size\n");
    for (int r=0; r<retries; r++) {

        // Start time
        gettimeofday(&t1,NULL);

        for(int i=0; i<dim; i++) {
            double im=init_b+(step*i);
            for (int j=0; j<dim; j++)
            {
                double cr;
                double a=cr=init_a+step*j;
                double b=im;
                int k = 0;
                for (k=0; k<niter; k++)
                {
                    double a2=a*a;
                    double b2=b*b;
                    if ((a2+b2)>4.0) break;
                    b=2*a*b+im;
                    a=a2-b2+cr;
                }
                M[j]= (unsigned char) 255-((k*255/niter));
            }
#if !defined(NO_DISPLAY)
            ShowLine(M,dim,i);
#endif
        }
        // Stop time
        gettimeofday(&t2,NULL);

        avg += runs[r] = diffmsec(t2,t1);
        printf("%s;%d;%d;%.2f;1;1\n", argv[0], dim, niter, runs[r]);
    }
    avg = avg / (double) retries;
    double var = 0;
    for (int r=0; r<retries; r++) {
        var += (runs[r] - avg) * (runs[r] - avg);
    }
    var /= retries;

#if !defined(NO_DISPLAY)
    printf("Average on %d experiments = %f (ms) Std. Dev. %f\n\nPress a key\n",retries,avg,sqrt(var));
    getchar();
    CloseXWindows();
#endif

    return 0;
}
