#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


void pgmsize (char *filename, int *nx, int *ny);
void pgmread (char *filename, void *vx, int nx, int ny);
void pgmwrite(char *filename, void *vx, int nx, int ny);

#define FILENAME "edge192x128.pgm"

#define M 192
#define N 128

#define P 4

#define MAXITER   10000
#define CHECKFREQ  50
#define DEL XXXXX

int main (int argc, char **argv)
{
	//init output files
	char filename1[64];
	char filename2[64];
	char filename3[64];
	sprintf(filename1,"out_%f.dat",DEL);
	sprintf(filename2,"times_%f.dat",DEL);
	FILE *fp_outres; 
	FILE *fp_times;
	fp_outres = fopen(filename1,"w");
	fp_times = fopen(filename2,"a");
	fprintf(fp_outres, "itter,avg,del\n");

	// start MPI
	int rank, size, next, prev;
	MPI_Status status;
	MPI_Request request;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(size != P){
		if (rank == 0) printf("ERROR: size = %d, P = %d\n", size, P);
		MPI_Finalize();
		exit(-1);
	}



	// set up cartesian topology
	MPI_Comm cart;
	int split[2] = {0,0};
	int periods[2] = {0,0};
	MPI_Dims_create(P, 2, split);
	MPI_Cart_create(MPI_COMM_WORLD, 2, split, periods, 1, &cart);
	MPI_Comm_rank(cart, &rank);

	int MP = M/split[0];
	int NP = N/split[1];

	int coords[2];
	MPI_Cart_coords(cart, rank, 2, coords);

	// Array and var declarations 
	int i, locali, j, localj, iter, maxiter;

	double time1, time2;

	float old[MP+2][NP+2];
	float new[MP+2][NP+2];
	float local_padded_array[MP+2][NP+2];
	float masterarray[M][N];
	float local_array[MP][NP];

	// read in image on proc 0
	if (rank == 0){
		pgmread(FILENAME, masterarray, M, N);
	}

	// distribute to all procs
	MPI_Bcast(masterarray,M*N,MPI_FLOAT,0,cart);


	// extract local section of image
	for(i=0;i<MP;i++){
	    for(j=0;j<NP;j++){

			locali = i + coords[0] * MP;
			localj = j + coords[1] * NP;


			local_array[i][j] = masterarray[locali][localj];

		}
	}

	// put data into padded array allowing for halos
	for (i = 0;i<MP +2;i++){
		for (j = 0;j<NP +2;j++){
			local_padded_array[i][j] = 255.0;
		}
	}
	for (i = 1;i<MP +1;i++){
		for (j = 1;j<NP +1;j++){
			local_padded_array[i][j] = local_array[i-1][j-1];
		}
	}

	// set initial conditions for the iteration
	for (i = 0;i<MP +2;i++){
		for (j = 0;j<NP +2;j++){
			old[i][j] = 255.0;
		}
	}




	// find the ranks of neighbours 
	int up_rank, down_rank, left_rank, right_rank;
	MPI_Cart_shift(cart, 0, 1, &right_rank, &left_rank);
	MPI_Cart_shift(cart, 1, 1, &down_rank, &up_rank);

	// create col data type
	MPI_Datatype col;
  	MPI_Type_vector(MP,1,NP+2,MPI_FLOAT,&col);
  	MPI_Type_commit(&col);

  	// init global reduction vars
  	float globalsum = 0.0;
  	float globaldel = 1000.0;

  	time1 = MPI_Wtime(); // start timing

  	// start iteration 
	for(iter = 0;iter < MAXITER;iter++){

		// swap halos
	  	MPI_Issend(&old[1][NP], 1,col,up_rank, 0,cart, &request);
	    MPI_Recv(&old[1][0], 1,col, down_rank, 0,cart, &status);

		MPI_Issend(&old[1][1], 1,col ,down_rank, 0,cart, &request);
	    MPI_Recv(&old[1][NP+1], 1,col ,up_rank, 0,cart, &status);

	    MPI_Issend(&old[MP][1], NP,MPI_FLOAT,left_rank, 0,cart, &request);
	    MPI_Recv(&old[0][1], NP,MPI_FLOAT, right_rank, 0,cart, &status);

		MPI_Issend(&old[1][1], NP,MPI_FLOAT,right_rank, 0,cart, &request);
	    MPI_Recv(&old[MP+1][1], NP,MPI_FLOAT, left_rank, 0,cart, &status);

	    // do local iteration
		for (i = 1;i<MP +1;i++){
			for (j = 1;j<NP +1;j++){
				new[i][j] = 0.25*(old[i][j-1]+old[i][j+1]+old[i-1][j]+old[i+1][j] - local_padded_array[i][j]);
			}
		}

		// check avg pixels and delta
		int cont = 0;
		if (iter%CHECKFREQ == 0){
			// calculate local values
			float localdel = 0.0;
			float curdel = 0.0;
			float localsum =0.0;
				for (i = 1;i<MP +1;i++){
					for (j = 1;j<NP +1;j++){
						curdel = fabs(old[i][j] - new[i][j]);
						localsum += new[i][j];
						if (curdel > localdel) localdel = curdel;
					}
				}
			
			
			// reduce local vars to global on proc 0
			MPI_Reduce(&localdel, &globaldel, 1, MPI_FLOAT, MPI_MAX, 0, cart);
			MPI_Reduce(&localsum, &globalsum, 1,MPI_FLOAT, MPI_SUM, 0, cart);

			// print data to file
			if (rank == 0) fprintf(fp_outres,"%f,%d,%f\n",globaldel,iter,globalsum/(M*N));

			// check if early stopping criterion is met
			if (rank == 0){
				if (globaldel < DEL) cont = 1; 
			}
			// distribute cont var to all procs so that they all break
			// out of loop if needed
			MPI_Bcast(&cont,1,MPI_INT,0,MPI_COMM_WORLD);

		
		}

		// set old = new
		for (i = 1;i<MP +1;i++){
			for (j = 1;j<NP +1;j++){
				old[i][j] = new[i][j];
			}
		}

		// break if criterion is met
		if (cont == 1) break; 

	}
	time2 = MPI_Wtime(); // finish timing

	// print time to file
	if (rank ==0 )fprintf(fp_times,"%d,%f,%f\n",P,time2 - time1,(time2 - time1)/(double) iter);

	// send local image back to proc 0
	float outarray[M][N];
	for (i=0;i<M;i++){
		for (j=0;j<N;j++){
			outarray[i][j] = 0.0;
		}
	}
	for(i=0;i<MP;i++){
	    for(j=0;j<NP;j++){

	      locali = i + coords[0] * MP;
	      localj = j + coords[1] * NP;


	      outarray[locali][localj] = old[i][j];

		}
	}
	MPI_Reduce(outarray,masterarray,M*N,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	sprintf(filename3,"image_%f.pgm",DEL);
	// write image to disk
	if (rank == 0){
		pgmwrite(filename3, masterarray, M, N);
	}

	// close output files
	fclose(fp_outres);
	fclose(fp_times);

	// finish MPI
	MPI_Finalize();
}


 
