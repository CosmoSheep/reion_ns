#include "../Parameter_files/INIT_PARAMS.H"
#include "../Parameter_files/ANAL_PARAMS.H"

/*
  USAGE: delta_ps <deltax filename> <xh filename> <output filename> <global xh>

  box is assumed to be of the HII dimension defined in ANAL_PARAM.H
*/

#define FORMAT (int) 0 /* 0= unpadded binary box; 1= FFT padded binary box (outdated) */
#define CONVERT_TO_DELTA (int) 0 /* 1= convert the field to a zero-mean delta; 
		     be careful not to do this with fields which are already zero mean */

int main(int argc, char ** argv){
  char filename[100];
  FILE *F, *F_xh;
  float REDSHIFT;
  int x,y,z, format;
  fftwf_complex *deltax, *delta_xh;
  fftwf_plan plan, plan_xh;
  float k_x, k_y, k_z, k_mag, k_floor, k_ceil, k_max, k_first_bin_ceil, k_factor, *xh;
  int i,j,k, n_x, n_y, n_z, NUM_BINS;
  double dvdx, ave, new_ave, ave_xh, *p_box, *k_ave;
  unsigned long long ct, *in_bin_ct;

  // check arguments
  if (argc != 5){
    fprintf(stderr, "USAGE: delta_ps <deltax filename> <xh filename> <output filename>\nAborting\n");
    return -1;
  }
  // initialize and allocate thread info
  if (fftwf_init_threads()==0){
    fprintf(stderr, "init: ERROR: problem initializing fftwf threads\nAborting\n.");
    return -1;
  }
  fftwf_plan_with_nthreads(NUMCORES); // use all processors for init

  ave=0;
  //allocate and read-in the density array
  deltax = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
  if (!deltax){
    fprintf(stderr, "delta_T: Error allocating memory for deltax box\nAborting...\n");
    fftwf_cleanup_threads(); return -1;
  }
  F = fopen(argv[1], "rb");
  switch (FORMAT){
    // FFT format
  case 1:
    fprintf(stderr, "Reading in FFT padded deltax box\n");
    if (mod_fread(deltax, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS, 1, F)!=1){
      fftwf_free(deltax);
      fprintf(stderr, "deltax_xh_ps.c: unable to read-in file\nAborting\n");
      fftwf_cleanup_threads(); return -1;
    }
    break;

    // unpaded format
  case 0:
    fprintf(stderr, "Reading in deltax unpadded box\n");
    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
        for (k=0; k<HII_DIM; k++){
    if (fread((float *)deltax + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F)!=1){
      fprintf(stderr, "init.c: Read error occured!\n");
      fftwf_free(deltax);
      fftwf_cleanup_threads(); return -1;     
    }
          ave += *((float *)deltax + HII_R_FFT_INDEX(i,j,k));
       }
      }
    }
    ave /= (double)HII_TOT_NUM_PIXELS;
    fprintf(stderr, "Average is %e\n", ave);
    break;

  default:
    fprintf(stderr, "Wrong format code\naborting...\n");
    fftwf_free(deltax);
    fftwf_cleanup_threads(); return -1;     
  }
  fclose(F);

  ave_xh=0;
  //allocate and read-in the neutral hydrogen fraction array
  delta_xh = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
  xh = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
  if (!xh){
    fprintf(stderr, "delta_T: Error allocating memory for xh box\nAborting...\n");
    fftwf_cleanup_threads(); return -1;
  }
  F_xh = fopen(argv[2], "rb");
  switch (FORMAT){
    // FFT format
  case 1:
    fprintf(stderr, "Reading in FFT padded xh box\n");
    if (mod_fread(delta_xh, sizeof(float)*HII_TOT_NUM_PIXELS, 1, F_xh)!=1){
      fftwf_free(delta_xh);
      fprintf(stderr, "deltax_xh_ps.c: unable to read-in file\nAborting\n");
      fftwf_cleanup_threads(); return -1;
    }
    break;

    // unpaded format
  case 0:
    fprintf(stderr, "Reading in xh unpadded box\n");
    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
  for (k=0; k<HII_DIM; k++){
    if (fread((float *)xh + HII_R_INDEX(i,j,k), sizeof(float), 1, F_xh)!=1){
      fprintf(stderr, "init.c: Read error occured!\n");
      fftwf_free(delta_xh);
      fftwf_cleanup_threads(); return -1;     
    }
  }
      }
    }

    ave_xh = atof(argv[4]);
    fprintf(stderr, "Average xh is %e\n", ave_xh);
    break;

    default:
      fprintf(stderr, "Wrong format code\naborting...\n");
      fftwf_free(delta_xh);
      fftwf_cleanup_threads(); return -1;     
    }
    fclose(F_xh);


  // convert absolute density to overdensiy with zero mean
  if (CONVERT_TO_DELTA){
    new_ave = 0;
    fprintf(stderr, "Now converting field to zero-mean delta\n");
    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
      	for (k=0; k<HII_DIM; k++){
	  *((float *)deltax + HII_R_FFT_INDEX(i,j,k)) /= ave;
	  *((float *)deltax + HII_R_FFT_INDEX(i,j,k)) -= 1;
	  new_ave += *((float *)deltax + HII_R_FFT_INDEX(i,j,k));
      	}
      }
    }
    new_ave /= (double) HII_TOT_NUM_PIXELS;
    fprintf(stderr, "The mean value of the field is now %e\n", new_ave);
  }



  // convert fraction to delta_xh
    fprintf(stderr, "Now converting xh to delta_xh\n");
    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
        for (k=0; k<HII_DIM; k++){
    *((float *)delta_xh + HII_R_FFT_INDEX(i,j,k))= *((float *)xh + HII_R_INDEX(i,j,k)) - ave_xh;
        }
      }
    }



  // do the FFTs
  plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deltax, (fftwf_complex *)deltax, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  fftwf_cleanup();
  for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
     deltax[ct] *= VOLUME/(HII_TOT_NUM_PIXELS+0.0);
  }

  plan_xh = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)delta_xh, (fftwf_complex *)delta_xh, FFTW_ESTIMATE);
  fftwf_execute(plan_xh);
  fftwf_destroy_plan(plan_xh);
  fftwf_cleanup();
  for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
      delta_xh[ct] *= VOLUME/(HII_TOT_NUM_PIXELS+0.0);
    }
 /***
  for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
     xh[ct] *= VOLUME/(HII_TOT_NUM_PIXELS+0.0);
  }
***/

  /******  PRINT OUT THE POWERSPECTRUM  *********/

  k_factor = 1.4;
  k_first_bin_ceil = DELTA_K;
  k_max = DELTA_K*HII_DIM;
  // initialize arrays
   NUM_BINS = 0;
  k_floor = 0;
  k_ceil = k_first_bin_ceil;
  while (k_ceil < k_max){
    NUM_BINS++;
    k_floor=k_ceil;
    k_ceil*=k_factor;
  }

  p_box =  (double *)malloc(sizeof(double)*NUM_BINS);
  k_ave =  (double *)malloc(sizeof(double)*NUM_BINS);
  in_bin_ct = (unsigned long long *)malloc(sizeof(unsigned long long)*NUM_BINS);
  if (!p_box || !in_bin_ct || !k_ave){ // a bit sloppy, but whatever..
    fprintf(stderr, "delta_T.c: Error allocating memory.\nAborting...\n");
    fftwf_free(deltax);
    fftwf_free(delta_xh);
    fftwf_cleanup_threads(); return -1;
  }
  for (ct=0; ct<NUM_BINS; ct++){
    p_box[ct] = k_ave[ct] = 0;
    in_bin_ct[ct] = 0;
  }


  // now construct the power spectrum file
  for (n_x=0; n_x<HII_DIM; n_x++){
    if (n_x>HII_MIDDLE)
      k_x =(n_x-HII_DIM) * DELTA_K;  // wrap around for FFT convention
    else
      k_x = n_x * DELTA_K;

    for (n_y=0; n_y<HII_DIM; n_y++){
      if (n_y>HII_MIDDLE)
	k_y =(n_y-HII_DIM) * DELTA_K;
      else
	k_y = n_y * DELTA_K;

      for (n_z=0; n_z<=HII_MIDDLE; n_z++){ 
	k_z = n_z * DELTA_K;
	
	k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

	// now go through the k bins and update
	ct = 0;
	k_floor = 0;
	k_ceil = k_first_bin_ceil;
	while (k_ceil < k_max){
	  // check if we fal in this bin
	  if ((k_mag>=k_floor) && (k_mag < k_ceil)){
	    in_bin_ct[ct]++;
	    p_box[ct] +=  pow(k_mag,3)*(creal(deltax[HII_C_INDEX(n_x, n_y, n_z)])*creal(delta_xh[HII_C_INDEX(n_x, n_y, n_z)])+cimag(deltax[HII_C_INDEX(n_x, n_y, n_z)])*cimag(delta_xh[HII_C_INDEX(n_x, n_y, n_z)]))/ (2.0*PI*PI*VOLUME);
	    // note the 1/VOLUME factor, which turns this into a power density in k-space

	    k_ave[ct] += k_mag;
	    break;
	  }

	  ct++;
	  k_floor=k_ceil;
	  k_ceil*=k_factor;
	}
      }
    }
  } // end looping through k box
  fftwf_free(deltax);
  fftwf_free(delta_xh);

  // now lets print out the k bins
  F = fopen(argv[3], "w");
  if (!F){
    fprintf(stderr, "delta_T.c: Couldn't open file %s for writting!\n", filename);
    fftwf_cleanup_threads(); return -1;
  }
  for (ct=1; ct<NUM_BINS; ct++){
    fprintf(F, "%e\t%e\t%e\n", k_ave[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0)/sqrt(in_bin_ct[ct]+0.0));
  }
  fclose(F);

  /****** END POWER SPECTRUM STUFF   ************/

  free(p_box); free(k_ave); free(in_bin_ct);

  fftwf_cleanup_threads(); return 0;
}
