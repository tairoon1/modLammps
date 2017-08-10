/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(heattransport/atom,ComputeHeatTransportAtom)

#else

#ifndef COMPUTE_HEAT_TRANSPORT_ATOM_H
#define COMPUTE_HEAT_TRANSPORT_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeHeatTransportAtom : public Compute {
 public:
  ComputeHeatTransportAtom(class LAMMPS *, int, char **);
  ~ComputeHeatTransportAtom();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  double memory_usage();
  //From ComputePEperatom
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 private:
  int nmax,maxneigh,nnn, diffuse_flag;
  double Kt, HeatTransportTimeStep, rdcut, Tmax;
  double *distsq;
  int *nearest;
  class NeighList *list;
  double *centro;

  //From ComputePEperatom
  int pairflag,bondflag,angleflag,dihedralflag,improperflag,kspaceflag;
  //int nmax;
  double *energy;
  int loop;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute transport/atom requires a pair style be defined

This is because the computation of the centro-symmetry values
uses a pairwise neighbor list.

W: More than one compute transport/atom

It is not efficient to use compute transport/atom more than once.

*/
