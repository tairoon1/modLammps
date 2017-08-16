/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include "compute_paris_atom.h"
#include "atom.h"
#include "update.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "fix.h"
#include "fix_peri_neigh.h"
#include "memory.h"
#include "error.h"
#include <vector>
#include <algorithm>

using namespace LAMMPS_NS;

enum{NOBIAS,BIAS};

/* ---------------------------------------------------------------------- */

ComputeParisAtom::ComputeParisAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  id_temp(NULL), stress(NULL)
{
  if (narg < 9) error->all(FLERR,"Illegal compute stress/atom command");

  peratom_flag = 1;
  size_peratom_cols = 6;
  pressatomflag = 1;
  timeflag = 1;
  comm_reverse = 6;

  // store temperature ID used by stress computation
  // insure it is valid for temperature computation

  id_temp = NULL;

  // process optional args
  keflag = 0;
  pairflag = 1;
  bondflag = angleflag = dihedralflag = improperflag = 1;
  kspaceflag = fixflag = 1;
  stress_component = atoi(arg[3]);
  A = atof(arg[4]);
  m = atof(arg[5]);
  omega = atof(arg[6]);  
  dt = atof(arg[7]);
  volume = atof(arg[8]);
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeParisAtom::~ComputeParisAtom()
{
  delete [] id_temp;
  memory->destroy(stress);
}

/* ---------------------------------------------------------------------- */

void ComputeParisAtom::init()
{
  // set temperature compute, must be done in init()
  // fixes could have changed or compute_modify could have changed it

  if (id_temp) {
    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute stress/atom temperature ID");
    temperature = modify->compute[icompute];
    if (temperature->tempbias) biasflag = BIAS;
    else biasflag = NOBIAS;
  } else biasflag = NOBIAS;
}

/* ---------------------------------------------------------------------- */

void ComputeParisAtom::compute_peratom()
{
  int i,j;
  double onemass;

  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local stress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(stress);
    nmax = atom->nmax;
    memory->create(stress,nmax,6,"paris/atom:stress");
    array_atom = stress;
  }

  // npair includes ghosts if either newton flag is set
  //   b/c some bonds/dihedrals call pair::ev_tally with pairwise info
  // nbond includes ghosts if newton_bond is set
  // ntotal includes ghosts if either newton flag is set
  // KSpace includes ghosts if tip4pflag is set

  int nlocal = atom->nlocal;
  int npair = nlocal;
  int nbond = nlocal;
  int ntotal = nlocal;
  int nkspace = nlocal;
  if (force->newton) npair += atom->nghost;
  if (force->newton_bond) nbond += atom->nghost;
  if (force->newton) ntotal += atom->nghost;
  if (force->kspace && force->kspace->tip4pflag) nkspace += atom->nghost;

  // clear local stress array

  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 6; j++)
      stress[i][j] = 0.0;

  // add in per-atom contributions from each force

  if (pairflag && force->pair) {
    double **vatom = force->pair->vatom;
    for (i = 0; i < npair; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (bondflag && force->bond) {
    double **vatom = force->bond->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (angleflag && force->angle) {
    double **vatom = force->angle->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (dihedralflag && force->dihedral) {
    double **vatom = force->dihedral->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (improperflag && force->improper) {
    double **vatom = force->improper->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (kspaceflag && force->kspace) {
    double **vatom = force->kspace->vatom;
    for (i = 0; i < nkspace; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  // add in per-atom contributions from relevant fixes
  // skip if vatom = NULL
  // possible during setup phase if fix has not initialized its vatom yet
  // e.g. fix ave/spatial defined before fix shake,
  //   and fix ave/spatial uses a per-atom stress from this compute as input

  if (fixflag) {
    for (int ifix = 0; ifix < modify->nfix; ifix++)
      if (modify->fix[ifix]->virial_flag) {
        double **vatom = modify->fix[ifix]->vatom;
        if (vatom)
          for (i = 0; i < nlocal; i++)
            for (j = 0; j < 6; j++)
              stress[i][j] += vatom[i][j];
      }
  }

  // communicate ghost virials between neighbor procs

  if (force->newton || (force->kspace && force->kspace->tip4pflag))
    comm->reverse_comm_compute(this);

  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) {
      stress[i][0] = 0.0;
      stress[i][1] = 0.0;
      stress[i][2] = 0.0;
      stress[i][3] = 0.0;
      stress[i][4] = 0.0;
      stress[i][5] = 0.0;
    }

  // include kinetic energy term for each atom in group
  // apply temperature bias is applicable
  // mvv2e converts mv^2 to energy

  if (keflag) {
    double **v = atom->v;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int *type = atom->type;
    double mvv2e = force->mvv2e;

    if (biasflag == NOBIAS) {
      if (rmass) {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    onemass = mvv2e * rmass[i];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	  }

      } else {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    onemass = mvv2e * mass[type[i]];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	  }
      }

    } else {

      // invoke temperature if it hasn't been already
      // this insures bias factor is pre-computed

      if (keflag && temperature->invoked_scalar != update->ntimestep)
	temperature->compute_scalar();

      if (rmass) {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    temperature->remove_bias(i,v[i]);
	    onemass = mvv2e * rmass[i];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	    temperature->restore_bias(i,v[i]);
	  }

      } else {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    temperature->remove_bias(i,v[i]);
	    onemass = mvv2e * mass[type[i]];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	    temperature->restore_bias(i,v[i]);
	  }
      }
    }
  }

  // convert to stress*volume units = -pressure*volume

  double nktv2p = -force->nktv2p;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      stress[i][0] *= nktv2p;
      stress[i][1] *= nktv2p;
      stress[i][2] *= nktv2p;
      stress[i][3] *= nktv2p;
      stress[i][4] *= nktv2p;
      stress[i][5] *= nktv2p;
    }

  // PARIS LAW
  // DETERMINE LOCAL MAX STRESS AND INDEX OF ATOM
  int rank;
  double **x = atom->x;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double maxStress=-10000.0,globalMaxStress=-99999.0;
  double secondmaxStress=-10000.0;
  int maxStressIndex=-1,secondmaxStressIndex=0;
  for (i = 0; i < nlocal; i++){
    if (x[i][2]>0.011 || x[i][2]<0.009)
         continue;
    double stress_comp; 
    if (stress_component==1) stress_comp = stress[i][0];
    else if (stress_component==2) stress_comp = stress[i][1];
    else if (stress_component==3) stress_comp = stress[i][2];
    else if (stress_component==4) stress_comp = stress[i][3];
    else if (stress_component==5) stress_comp = stress[i][4];
    else if (stress_component==6) stress_comp = stress[i][5];
    stress_comp = MAX(stress_comp,0.0);
    if (stress_comp>=maxStress-0.000001){
      secondmaxStress = maxStress;
      secondmaxStressIndex = maxStressIndex;
      maxStress = stress_comp;
      maxStressIndex = i;
    }    
  }

  // DETERMINE GLOBAL MAX STRESS
  MPI_Allreduce(&maxStress,&globalMaxStress,1,MPI_DOUBLE,MPI_MAX,world);
  int ifix_peri = 3;
  tagint **partner = ((FixPeriNeigh *) modify->fix[ifix_peri])->partner;
  int *npartner = ((FixPeriNeigh *) modify->fix[ifix_peri])->npartner;
  std::vector<int> overlappingIndex;
  // IF GLOBAL == LOCAL, APPLY PARIS LAW
  if(maxStress==globalMaxStress){
    // apply on center
    atom->lambda[maxStressIndex] = atom->lambda[maxStressIndex]-A*pow(maxStress/volume/1.0e6,m)*omega*dt;
    if (atom->lambda[maxStressIndex] <= 0.0)
      atom->lambda[maxStressIndex] = 0.0;
    int jnum = npartner[maxStressIndex];

    // apply on neighbours
    for (int jj = 0; jj < jnum; jj++){
      if (partner[maxStressIndex][jj] == 0) continue;
        // look up local index of jj of i
      j = atom->map(partner[maxStressIndex][jj]);
      
      // j = -1 means not existent bond
      // j = 0 means ??? MAYBE ON ANOTHER PROCESSOR???
      if (j < 0) {
        partner[maxStressIndex][jj] = 0;
        continue;
      }

      if (atom->lambda[j] == 0.0)
        continue;
      double stress_comp; 
      if (stress_component==1) stress_comp = stress[j][0];
      else if (stress_component==2) stress_comp = stress[j][1];
      else if (stress_component==3) stress_comp = stress[j][2];
      else if (stress_component==4) stress_comp = stress[j][3];
      else if (stress_component==5) stress_comp = stress[j][4];
      else if (stress_component==6) stress_comp = stress[j][5];
      stress_comp = MAX(stress_comp,0.0);

      atom->lambda[j] = atom->lambda[j]-A*pow(stress_comp/volume/1.0e6,m)*omega*dt;
      if (atom->lambda[j] <= 0.0)
        atom->lambda[j] = 0.0;
      overlappingIndex.push_back(j);
    }
  }

    // IF GLOBAL == SECONDLOCAL, APPLY PARIS LAW SYMMETRY!
  if(secondmaxStress>=globalMaxStress-0.000001){
    // apply on center
    atom->lambda[secondmaxStressIndex] = atom->lambda[secondmaxStressIndex]-A*pow(secondmaxStress/volume/1.0e6,m)*omega*dt;
    if (atom->lambda[secondmaxStressIndex] <= 0.0)
      atom->lambda[secondmaxStressIndex] = 0.0;
    int jnum = npartner[secondmaxStressIndex];

    // apply on neighbours
    for (int jj = 0; jj < jnum; jj++){
      if (partner[secondmaxStressIndex][jj] == 0) continue;
        // look up local index of jj of i
      j = atom->map(partner[secondmaxStressIndex][jj]);
      
      // j = -1 means not existent bond
      // j = 0 means ??? MAYBE ON ANOTHER PROCESSOR???
      if (j < 0) {
        partner[secondmaxStressIndex][jj] = 0;
        continue;
      }
      // if fatigue was already applied on this atom!
      if(std::find(overlappingIndex.begin(), overlappingIndex.end(), j) != overlappingIndex.end()) {
          continue;
      } 
      if (atom->lambda[j] == 0.0)
        continue;
      double stress_comp; 
      if (stress_component==1) stress_comp = stress[j][0];
      else if (stress_component==2) stress_comp = stress[j][1];
      else if (stress_component==3) stress_comp = stress[j][2];
      else if (stress_component==4) stress_comp = stress[j][3];
      else if (stress_component==5) stress_comp = stress[j][4];
      else if (stress_component==6) stress_comp = stress[j][5];
      stress_comp = MAX(stress_comp,0.0);

      atom->lambda[j] = atom->lambda[j]-A*pow(stress_comp/volume/1.0e6,m)*omega*dt;
      if (atom->lambda[j] <= 0.0)
        atom->lambda[j] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeParisAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = stress[i][0];
    buf[m++] = stress[i][1];
    buf[m++] = stress[i][2];
    buf[m++] = stress[i][3];
    buf[m++] = stress[i][4];
    buf[m++] = stress[i][5];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeParisAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    stress[j][0] += buf[m++];
    stress[j][1] += buf[m++];
    stress[j][2] += buf[m++];
    stress[j][3] += buf[m++];
    stress[j][4] += buf[m++];
    stress[j][5] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeParisAtom::memory_usage()
{
  double bytes = nmax*6 * sizeof(double);
  return bytes;
}
