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
   Contributing author: Mike Parks (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "pair_peri_pmb.h"
#include "atom.h"
#include "domain.h"
#include "lattice.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_peri_neigh.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include <vector>
#include <algorithm>
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairPeriPMB::PairPeriPMB(LAMMPS *lmp) : Pair(lmp)
{
  for (int i = 0; i < 6; i++) virial[i] = 0.0;
  no_virial_fdotr_compute=1;

  ifix_peri = -1;

  nmax = 0;
  s0_new = NULL;

  kspring = NULL;
  s00 = NULL;
  alpha = NULL;
  cut = NULL;
}

/* ---------------------------------------------------------------------- */

PairPeriPMB::~PairPeriPMB()
{
  if (ifix_peri >= 0) modify->delete_fix("PERI_NEIGH");

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(kspring);
    memory->destroy(s00);
    memory->destroy(alpha);
    memory->destroy(cut);
    memory->destroy(s0_new);
  }
}

/* ---------------------------------------------------------------------- */

void PairPeriPMB::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double xtmp0,ytmp0,ztmp0,delx0,dely0,delz0,rsq0;
  double rsq,r,dr,rk,evdwl,fpair,fbond;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double d_ij,delta,stretch;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **f = atom->f;
  double **x = atom->x;
  double *lambda = atom->lambda;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double *vfrac = atom->vfrac;
  double *s0 = atom->s0;
  double **x0 = atom->x0;
  double **r0   = ((FixPeriNeigh *) modify->fix[ifix_peri])->r0;
  tagint **partner = ((FixPeriNeigh *) modify->fix[ifix_peri])->partner;
  int *npartner = ((FixPeriNeigh *) modify->fix[ifix_peri])->npartner;

  double *temp = atom->temperature; //Needed for thermo-mechanic formulation

  // lc = lattice constant
  // init_style guarantees it's the same in x, y, and z

  double lc = domain->lattice->xlattice;
  double half_lc = 0.5*lc;
  double vfrac_scale = 1.0;

  // short-range forces

  int newton_pair = force->newton_pair;
  int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // need minimg() for x0 difference since not ghosted

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    xtmp0 = x0[i][0];
    ytmp0 = x0[i][1];
    ztmp0 = x0[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      if (periodic) domain->minimum_image(delx0,dely0,delz0);
      rsq0 = delx0*delx0 + dely0*dely0 + delz0*delz0;
      jtype = type[j];

      r = sqrt(rsq);

      // short-range interaction distance based on initial particle position
      // 0.9 and 1.35 are constants

      d_ij = MIN(0.9*sqrt(rsq0),1.35*lc);

      // short-range contact forces
      // 15 is constant taken from the EMU Theory Manual
      // Silling, 12 May 2005, p 18

      if (r < d_ij) {
        dr = r - d_ij;

        rk = (15.0 * kspring[itype][jtype] * vfrac[j]) *
          (dr / cut[itype][jtype]);
        if (r > 0.0) fpair = -(rk/r);
        else fpair = 0.0;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) evdwl = 0.5*(rk - thermal_coeff*(temp[i] - Tref))*dr;
        if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,
                             fpair*vfrac[i],delx,dely,delz);
      }
    }
  }

  // grow bond forces array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(s0_new);
    nmax = atom->nmax;
    memory->create(s0_new,nmax,"pair:s0_new");
  }



  /* In the following all the bonds are broken in order for the crack to grow. The crack grows when a point is lambda = 0.
   * 1. Find the direction of the crack
   * 2. Save the local index of broken points and all the local indices of neighbor points on the side of the crack.
   * 3. Loop over every broken point and its neighbor points, loop over all the neighbor points and find their neighbor points. 
   * 4. Find the intersection of the neighbor points of every broken point with the neighbor points of the neighbor points.
   * 5. All the points in this set are checked and bonds are deleted between every two points when they are on different faces of the crack.
   * BTW: Here, the bonds from the lambda=0 point to his surrounding points are not broken. Those are broken in the loop afterwards!
   * author: tairoon1
   * date: 8th August, 2017
   */
  double *chemPotential = atom->chemPotential;
  int rank;
  MPI_Comm_rank(world,&rank);
  // This is momentarily used for debugging purpose! To see the points and which side was determined!
  double *concentration = atom->concentration;
  // used to determine positions to prevent rounding errors
  double epstolerance = 0.0002;
  enum Direction {LEFT,RIGHT,UP,DOWN,LEFTUP,RIGHTUP,LEFTDOWN,RIGHTDOWN};
  
  std::vector<std::vector<int> > localindexPartner;
  std::vector<int> indexBrokenPoints;
  std::vector<int> directionBrokenPoint;
  for (i = 0; i < nlocal; i++) {
    chemPotential[i] = rank;
    int neighCrackDirection = 999;
    // if one point is broken
    if (lambda[i]==0.0){
      // coordinates of broken point
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      // total number of neighbours
      jnum = npartner[i];
      
      /*----------------check on which side crack is!-------------*/
      // Check if there are broken points nearby to determine in which direction the crack is propagatin.
      // THIS DOES NOT WORK VERY WELL!!! SHOULD CONSIDER A DIFFERENT ALGORITHM FOR THE DIRECTION DETECTION.
      for (jj = 0; jj < jnum; jj++){
        // if already broken skip
        if (partner[i][jj] == 0) continue;
        // look up local index of jj of i
        j = atom->map(partner[i][jj]);
        if (j < 0) {
          partner[i][jj] = 0;
          continue;
        }
        // distance to neighbor point
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        

        // skip points that are not on the same xy plane
        if (fabs(delz)>epstolerance)
          continue;
        // skip points that are not direct neighbors
        if(sqrt(delx*delx+dely*dely)>sqrt(neighbor->cutneighmax/4.0*neighbor->cutneighmax/4.0*2.0)+epstolerance){
          continue;
        }

        // if neighbor point is broken, check on which side and determine where the crack comes from
        // THIS IS A BIG PROBLEM WHEN THE CRACK CHANGES DIRECTION SINCE THE CONNECTION AND FINDING THE POINTS BOTH DEPEND ON partner
        if(lambda[j]==0.0){
          if (delx>epstolerance && dely<-epstolerance){
            neighCrackDirection = LEFTUP;
            continue;
          }
          else if (delx>epstolerance && dely>epstolerance){
            neighCrackDirection = LEFTDOWN;
            continue;
          }
          else if (delx<-epstolerance && dely<-epstolerance){
            neighCrackDirection = RIGHTUP;
            continue;
          }
          else if (delx<-epstolerance && dely>epstolerance){
            neighCrackDirection = RIGHTDOWN;
            continue;
          }
          // if neighbor in right left top bottom direction, cannot be diagonal, so overwrite previous direction
          else if (fabs(delx)<epstolerance && dely>epstolerance){
            neighCrackDirection = DOWN;
            break;
          }
          else if (fabs(delx)<epstolerance && dely<-epstolerance){
            neighCrackDirection = UP;
            break;
          }
          else if (delx<-epstolerance && fabs(dely)<epstolerance){
            neighCrackDirection = RIGHT;
            break;
          }
          else if (delx>epstolerance && fabs(dely)<epstolerance){
            neighCrackDirection = LEFT;
            break;
          }
          // if there is no nearby point
          else
            continue;
        } 
      }


      /*----------------find all points where bonds have to be broken-------------*/
      // If no direction, skip this part!
      if(neighCrackDirection!=999){
        // to save all local indices of neighbours
        std::vector<int> localVector;
        for (jj = 0; jj < jnum; jj++){    
          // if already broken skip
          if (partner[i][jj] == 0) continue;
          // look up local index of jj of i
          j = atom->map(partner[i][jj]);
          if (j < 0) {
            partner[i][jj] = 0;
            continue;
          }

          // calculate distances to determine on which side the current neighbour is
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];

          // THE CONCENTRATION IS JUST FOR DEBUGGING PURPOSES TO SEE IF THE CRACKDIRECTION AND THE ADDED POINTS MAKE SENSE!
          if (neighCrackDirection==LEFT){
            if (delx>-epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 100.0;
            }
          }
          else if (neighCrackDirection==RIGHT){
            if (delx<epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 200.0;
            }
          }
          else if (neighCrackDirection==UP){
            if (dely<epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 300.0;
            }
          }
          else if (neighCrackDirection==DOWN){
            if (dely>-epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 400.0;
            }
          }
          else if (neighCrackDirection==LEFTDOWN){
            if (atan2(-dely,-delx)>3./4*M_PI-epstolerance || atan2(-dely,-delx)<-1./4*M_PI+epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 500.0;
            }
          }
          else if (neighCrackDirection==LEFTUP){
            if (atan2(-dely,-delx)>1./4*M_PI-epstolerance || atan2(-dely,-delx)<-3./4*M_PI+epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 600.0;
            }
          }
          else if (neighCrackDirection==RIGHTDOWN){
            if (atan2(-dely,-delx)>-3./4*M_PI-epstolerance && atan2(-dely,-delx)<1./4*M_PI+epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 700.0;
            }
          }
          else if (neighCrackDirection==RIGHTUP){
            if (atan2(-dely,-delx)>-1./4*M_PI-epstolerance && atan2(-dely,-delx)<3./4*M_PI+epstolerance){
              // save the local index of current neighbour
              localVector.push_back(j);
              concentration[j] = 800.0;
            }
          }
        }
        // save local indices of all neighbours which bonds have to be broken
        localindexPartner.push_back(localVector);
        indexBrokenPoints.push_back(i);
        directionBrokenPoint.push_back(neighCrackDirection);
      }
    }
  }

  /*----------------break the bonds of those points!-------------*/
  // iterate through broken points
  for (int k = 0; k < localindexPartner.size(); k++){
    // those are the coordinates
    xtmp = x[indexBrokenPoints[k]][0];
    ytmp = x[indexBrokenPoints[k]][1];
    ztmp = x[indexBrokenPoints[k]][2];
    // iterate through neighbours of current broken point
    for (i = 0; i < localindexPartner[k].size(); i++){
      jnum=npartner[localindexPartner[k][i]];
      // iterate through neighbours of current neighbour point
      for(jj = 0; jj < jnum; jj++){
        // if bond between neighbour and its neighbour is gone skip
        if (partner[localindexPartner[k][i]][jj] == 0) continue;
        // look up local index of jj of i
        j = atom->map(partner[localindexPartner[k][i]][jj]);
        // j = -1 means not existent bond
        // j = 0 means ???
        if (j < 0) {
          partner[localindexPartner[k][i]][jj] = 0;
          continue;
        }
        // if the neighborpoint of the neighbor is also a neighbor point of the broken point, break the bonds by setting partner = 0
        // COULD USE A NEW 2D LIST? lambdaList = 1 or 0 and can be multiplied with bondforce
        if (std::find(localindexPartner[k].begin(), localindexPartner[k].end(), j) != localindexPartner[k].end()){
          if (directionBrokenPoint[k]==LEFT || directionBrokenPoint[k]==RIGHT){
            if ((x[localindexPartner[k][i]][1] > ytmp+epstolerance && x[j][1] < ytmp+epstolerance) || (x[localindexPartner[k][i]][1] < ytmp-epstolerance && x[j][1] > ytmp-epstolerance))
              partner[localindexPartner[k][i]][jj] = 0;
          }
          else if (directionBrokenPoint[k]==UP || directionBrokenPoint[k]==DOWN){
            if ((x[localindexPartner[k][i]][0] > xtmp+epstolerance && x[j][0] < xtmp+epstolerance) || (x[localindexPartner[k][i]][0] < xtmp-epstolerance && x[j][0] > xtmp-epstolerance))
              partner[localindexPartner[k][i]][jj] = 0;
          }
          else if (directionBrokenPoint[k]==LEFTUP){
            if ((((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>3./4*M_PI-epstolerance) || (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<-3./4*M_PI+epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)<3./4*M_PI-epstolerance) && (atan2(x[j][1]-ytmp,x[j][0]-xtmp)>1./4*M_PI-epstolerance))) || 
               (((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<3./4*M_PI+epstolerance) && (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>1./4*M_PI-epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)>3./4*M_PI+epstolerance) || (atan2(x[j][1]-ytmp,x[j][0]-xtmp)<-3./4*M_PI+epstolerance))))
              partner[localindexPartner[k][i]][jj] = 0;
          }
          else if (directionBrokenPoint[k]==LEFTDOWN){
            if ((((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>3./4*M_PI-epstolerance) || (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<-3./4*M_PI+epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)>-3./4*M_PI+epstolerance) && (atan2(x[j][1]-ytmp,x[j][0]-xtmp)<-1./4*M_PI+epstolerance))) || 
               (((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>-3./4*M_PI-epstolerance) && (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<-1./4*M_PI+epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)>3./4*M_PI-epstolerance) || (atan2(x[j][1]-ytmp,x[j][0]-xtmp)<-3./4*M_PI-epstolerance))))
              partner[localindexPartner[k][i]][jj] = 0;
          }
          else if (directionBrokenPoint[k]==RIGHTUP){
            if ((((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>1./4*M_PI-epstolerance) && (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<3./4*M_PI+epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)>-1./4*M_PI-epstolerance) && (atan2(x[j][1]-ytmp,x[j][0]-xtmp)<1./4*M_PI-epstolerance))) || 
               (((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>-1./4*M_PI-epstolerance) && (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<1./4*M_PI+epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)>1./4*M_PI+epstolerance) && (atan2(x[j][1]-ytmp,x[j][0]-xtmp)<3./4*M_PI+epstolerance))))
              partner[localindexPartner[k][i]][jj] = 0;
          }
          else if (directionBrokenPoint[k]==RIGHTDOWN){
            if ((((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>-1./4*M_PI-epstolerance) && (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<1./4*M_PI+epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)>-3./4*M_PI-epstolerance) && (atan2(x[j][1]-ytmp,x[j][0]-xtmp)<-1./4*M_PI-epstolerance))) || 
               (((atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)>-3./4*M_PI-epstolerance) && (atan2(x[localindexPartner[k][i]][1]-ytmp,x[localindexPartner[k][i]][0]-xtmp)<-1./4*M_PI+epstolerance))
            && ((atan2(x[j][1]-ytmp,x[j][0]-xtmp)>-1./4*M_PI+epstolerance) && (atan2(x[j][1]-ytmp,x[j][0]-xtmp)<1./4*M_PI+epstolerance))))
              partner[localindexPartner[k][i]][jj] = 0;
          }
        }
        // delete bond between neighbour neighbour to current broken point because broken point is not in its own horizon
        else if (j == indexBrokenPoints[k]){
            partner[localindexPartner[k][i]][jj] = 0;
        }
      }
    }
  }
  


  // loop over my particles and their partners
  // partner list contains all bond partners, so I-J appears twice
  // if bond already broken, skip this partner
  // first = true if this is first neighbor of particle i

  bool first;

  for (i = 0; i < nlocal; i++) {
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jnum = npartner[i];
    s0_new[i] = DBL_MAX;
    first = true;

    for (jj = 0; jj < jnum; jj++) {
      if (partner[i][jj] == 0) continue;
      j = atom->map(partner[i][jj]);

      // check if lost a partner without first breaking bond

      if (j < 0) {
        partner[i][jj] = 0;
        continue;
      }

      // break bonds from broken point to every point around it
      if (lambda[i] == 0.0){
        partner[i][jj] = 0;
        continue;
      }


      // compute force density, add to PD equation of motion

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      if (periodic) domain->minimum_image(delx,dely,delz);
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      delta = cut[itype][jtype];
      r = sqrt(rsq);
      dr = r - r0[i][jj];

      // avoid roundoff errors

      if (fabs(dr) < 2.2204e-016) dr = 0.0;

      // scale vfrac[j] if particle j near the horizon

      if ((fabs(r0[i][jj] - delta)) <= half_lc)
        vfrac_scale = (-1.0/(2*half_lc))*(r0[i][jj]) +
          (1.0 + ((delta - half_lc)/(2*half_lc) ) );
      else vfrac_scale = 1.0;

      stretch = dr / r0[i][jj];
      rk = (kspring[itype][jtype] * vfrac[j]) * vfrac_scale * stretch;
      if (r > 0.0) fbond = -(rk/r);
      else fbond = 0.0;

      f[i][0] += delx*fbond;
      f[i][1] += dely*fbond;
      f[i][2] += delz*fbond;

      
      // since I-J is double counted, set newton off & use 1/2 factor and I,I

      if (eflag) evdwl = 0.5*(rk - thermal_coeff*(temp[i] - Tref))*dr;
      if (evflag) ev_tally(i,i,nlocal,0,0.5*evdwl,0.0,0.5*fbond*vfrac[i],delx,dely,delz);

      // find stretch in bond I-J and break if necessary
      // use s0 from previous timestep

      if (stretch > MIN(s0[i],s0[j])) partner[i][jj] = 0;

      // update s0 for next timestep

      if (first)
         s0_new[i] = s00[itype][jtype] - (alpha[itype][jtype] * stretch);
      else
         s0_new[i] = MAX(s0_new[i],s00[itype][jtype] - (alpha[itype][jtype] * stretch));
      first = false;

    }
  }


    
  

  // store new s0
  for (i = 0; i < nlocal; i++) s0[i] = s0_new[i];
    
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairPeriPMB::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(kspring,n+1,n+1,"pair:kspring");
  memory->create(s00,n+1,n+1,"pair:s00");
  memory->create(alpha,n+1,n+1,"pair:alpha");
  memory->create(cut,n+1,n+1,"pair:cut");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPeriPMB::settings(int narg, char **arg)
{
  if (narg) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPeriPMB::coeff(int narg, char **arg)
{
  if (narg != 8) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double kspring_one = force->numeric(FLERR,arg[2]);
  double cut_one = force->numeric(FLERR,arg[3]);
  double s00_one = force->numeric(FLERR,arg[4]);
  double alpha_one = force->numeric(FLERR,arg[5]);
  thermal_coeff = force->numeric(FLERR,arg[6]);
  Tref = force->numeric(FLERR,arg[7]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      kspring[i][j] = kspring_one;
      s00[i][j] = s00_one;
      alpha[i][j] = alpha_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPeriPMB::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  kspring[j][i] = kspring[i][j];
  alpha[j][i] = alpha[i][j];
  s00[j][i] = s00[i][j];
  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairPeriPMB::init_style()
{
  // error checks

  if (!atom->peri_flag)
    error->all(FLERR,"Pair style peri requires atom style peri");
  if (atom->map_style == 0)
    error->all(FLERR,"Pair peri requires an atom map, see atom_modify");

  if (domain->lattice->xlattice != domain->lattice->ylattice ||
      domain->lattice->xlattice != domain->lattice->zlattice ||
      domain->lattice->ylattice != domain->lattice->zlattice)
    error->all(FLERR,"Pair peri lattice is not identical in x, y, and z");

  // if first init, create Fix needed for storing fixed neighbors

  if (ifix_peri == -1) {
    char **fixarg = new char*[3];
    fixarg[0] = (char *) "PERI_NEIGH";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "PERI_NEIGH";
    modify->add_fix(3,fixarg);
    delete [] fixarg;
  }

  // find associated PERI_NEIGH fix that must exist
  // could have changed locations in fix list since created

  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"PERI_NEIGH") == 0) ifix_peri = i;
  if (ifix_peri == -1) error->all(FLERR,"Fix peri neigh does not exist");

  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPeriPMB::write_restart(FILE *fp)
{
  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&kspring[i][j],sizeof(double),1,fp);
        fwrite(&s00[i][j],sizeof(double),1,fp);
        fwrite(&alpha[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPeriPMB::read_restart(FILE *fp)
{
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&kspring[i][j],sizeof(double),1,fp);
          fread(&s00[i][j],sizeof(double),1,fp);
          fread(&alpha[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&kspring[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&s00[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&alpha[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ---------------------------------------------------------------------- */

double PairPeriPMB::single(int i, int j, int itype, int jtype, double rsq,
                           double factor_coul, double factor_lj,
                           double &fforce)
{
  double delx0,dely0,delz0,rsq0;
  double d_ij,r,dr,rk,vfrac_scale;

  double *vfrac = atom->vfrac;
  double **x0 = atom->x0;
  double **r0   = ((FixPeriNeigh *) modify->fix[ifix_peri])->r0;
  tagint **partner = ((FixPeriNeigh *) modify->fix[ifix_peri])->partner;
  int *npartner = ((FixPeriNeigh *) modify->fix[ifix_peri])->npartner;

  double lc = domain->lattice->xlattice;
  double half_lc = 0.5*lc;

  delx0 = x0[i][0] - x0[j][0];
  dely0 = x0[i][1] - x0[j][1];
  delz0 = x0[i][2] - x0[j][2];
  int periodic = domain->xperiodic || domain->yperiodic || domain->zperiodic;
  if (periodic) domain->minimum_image(delx0,dely0,delz0);
  rsq0 = delx0*delx0 + dely0*dely0 + delz0*delz0;

  d_ij = MIN(0.9*sqrt(rsq0),1.35*lc);
  r = sqrt(rsq);

  double energy = 0.0;
  fforce = 0.0;

  if (r < d_ij) {
    dr = r - d_ij;
    rk = (15.0 * kspring[itype][jtype] * vfrac[j]) *
      (dr / sqrt(cutsq[itype][jtype]));
    if (r > 0.0) fforce += -(rk/r);
    energy += 0.5*rk*dr;
  }

  int jnum = npartner[i];
  for (int jj = 0; jj < jnum; jj++) {
    if (partner[i][jj] == 0) continue;
    if (j < 0) continue;
    if (j == atom->map(partner[i][jj])) {
      dr = r - r0[i][jj];
      if (fabs(dr) < 2.2204e-016) dr = 0.0;
      if ( (fabs(r0[i][jj] - sqrt(cutsq[itype][jtype]))) <= half_lc)
        vfrac_scale = (-1.0/(2*half_lc))*(r0[i][jj]) +
          (1.0 + ((sqrt(cutsq[itype][jtype]) - half_lc)/(2*half_lc)));
      else vfrac_scale = 1.0;
      rk = (kspring[itype][jtype] * vfrac[j] * vfrac_scale) *
        (dr / r0[i][jj]);
      if (r > 0.0) fforce += -(rk/r);
      energy += 0.5*rk*dr;
    }
  }

  return energy;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairPeriPMB::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
