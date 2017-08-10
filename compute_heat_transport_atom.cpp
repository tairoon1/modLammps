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

/* ----------------------------------------------------------------------
   Contributing author: Mauricio Ponga (UBC) - Based on Centro code 
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "compute_heat_transport_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeHeatTransportAtom::ComputeHeatTransportAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
//  compute 1 all transport/atom Kt 1.0 dt 1.0e-5 Tmax 500
  if (narg != 11) error->all(FLERR,"Illegal compute transport/atom command");

  if (strcmp(arg[3],"Kt") == 0) Kt = force->numeric(FLERR,arg[4]);

  if (strcmp(arg[5],"dt") == 0) HeatTransportTimeStep = force->numeric(FLERR,arg[6]);

  if (strcmp(arg[7],"Tmax") == 0) Tmax = force->numeric(FLERR,arg[8]);

  if (strcmp(arg[9],"loop") == 0) loop = force->numeric(FLERR,arg[10]);

  if (narg != 11) error->all(FLERR,"Illegal compute transport/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  centro = NULL;
  maxneigh = 0;
  distsq = NULL;
  nearest = NULL;

  //From ComputePEperatom
  if (narg < 3) error->all(FLERR,"Illegal compute pe/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;
  peatomflag = 1;
  timeflag = 1;
  comm_reverse = 1;

  pairflag = 1;
  bondflag = angleflag = dihedralflag = improperflag = 1;
  kspaceflag = 1;

  nmax = 0;
  energy = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeHeatTransportAtom::~ComputeHeatTransportAtom()
{
  memory->destroy(centro);
  memory->destroy(distsq);
  memory->destroy(nearest);
  memory->destroy(energy);
}

/* ---------------------------------------------------------------------- */

void ComputeHeatTransportAtom::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Compute transport/atom requires a pair style be defined");

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"transport/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute transport/atom");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeHeatTransportAtom::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeHeatTransportAtom::compute_peratom()
{
  int i;

  invoked_peratom = update->ntimestep;
  if (update->eflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom energy was not tallied on needed timestep");

  // grow local energy array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(energy);
    nmax = atom->nmax;
    memory->create(energy,nmax,"pe/atom:energy");
    vector_atom = energy;
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

  // zero energy of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) energy[i] = 0.0;

  int j,k,ii,jj,kk,n,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,value;
  int *ilist,*jlist,*numneigh,**firstneigh;

  invoked_peratom = update->ntimestep;

  // grow centro array if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(centro);
    nmax = atom->nmax;
    memory->create(centro,nmax,"transport/atom:transport");
    vector_atom = centro;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int nhalf = nnn/2;

  // compute mass transport for each atom in group
  // use full neighbor list

  double **x = atom->x;

  int *type = atom->type; //Needed for mass transport
  double *temp = atom->temperature; //Needed for max-ent
  double cutsq = force->pair->cutforce * force->pair->cutforce;
  double dTi_dt = 0.0;
  double *mass = atom->mass; //Needed for coarse-grained simulations?
  int  iloop = 0;

  double kB = 8.6173324e-5; // Boltzmann constant eV/K //1.38064852e-23 J/K

  for (iloop = 0; iloop < loop; iloop++) {
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    double beta_i = 1.0/(kB * temp[i]);

    dTi_dt = 0.0;

    if (mask[i] & groupbit) {
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];

      jlist = firstneigh[i];
      jnum = numneigh[i];

      n = 0;
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        jtype = type[j];

          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;
          if (rsq < cutsq) {

	      dTi_dt += temp[i]*Kt*( temp[j]/Tmax*(1.0-temp[i]/Tmax)*exp(-2.0*(temp[j] - temp[i])/(temp[j]+temp[i]) ) 
			           - temp[i]/Tmax*(1.0-temp[j]/Tmax)*exp(-2.0*(temp[i] - temp[j])/(temp[i]+temp[j]) ));

          }
      } //end for jj
    }
    temp[i] += (dTi_dt*HeatTransportTimeStep);

  } //end ii atom
  } //end iloop
}

/* ---------------------------------------------------------------------- */

int ComputeHeatTransportAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) buf[m++] = energy[i];
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeHeatTransportAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    energy[j] += buf[m++];
  }
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeHeatTransportAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
