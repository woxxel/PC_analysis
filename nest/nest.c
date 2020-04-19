//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//             Nested Sampling
// 
// Filename:   nest.c, version 1.20, 4 September 2004
//
// Purpose:    Bayesian computation.
//
//                    Pr(x)  Pr(D|x)  =   Pr(D)  Pr(x|D)
//                   Prior.Likelihood = Evidence.Posterior
//
//             Obtain Evidence and quantified properties of x,
//             with posterior samples as optional by-product.
//
//             John Skilling, Kenmare, Ireland, April-September 2004
//             email: skilling@eircom.net
//=============================================================================
/*
    Copyright (c) 2004, Maximum Entropy Data Consultants Ltd,
                        114c Milton Road, Cambridge CB4 1XE, England

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation Inc.,
    59 Temple Place, Suite 330, Boston, MA  02111-1307  USA; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.
*/
//=============================================================================
//
//                                                 MAIN
//                                                   |
//                                               NestSampler
//                                                   |
//      ____________________________________________/|\_______________
//     |        |           |               |        |                |
// NestSet  NestReset  NestIntegrate  NestProperties |          NestPosterior
//                             |       |             |                |
//                           NestStatistics          |                |
//                                                   |                |
//                            ______________________/|\____           |
//                           |            |          |     |          |
//                       UserPrior  UserProperties   |   UserExplore  |
//                                                   |     |     :    |
//                                                  NestLhood    :    |
//                                                          \    :   /
//                                                           NestCopy
//                                                               |
//                                                           UserCopy
//-----------------------------------------------------------------------------
#include <stdlib.h>
#include <stdarg.h>
#include <float.h>
#include <math.h>
#include "nest.h"                 // My header
#include "random.h"               // Random library

/***********************/
/* Internal prototypes */
/***********************/
static void    NestSet       (GlobalStr*);
static int     NestReset     (GlobalStr*);
static void    NestIntegrate (GlobalStr*, int);
static int     NestPosterior (GlobalStr*, ObjectStr*, ObjectStr*);
static void    NestProperties(GlobalStr*);
static void    NestStatistics(ResultStr*);

/*********************************/
/* Internal constants and macros */
/*********************************/
#undef  CALLOC    // allocates vector p[0:n-1] of type t
#undef  FREE      // frees CALLOC or NULL vector p[0:*], sets NULL
#undef  CALL      // call mechanism enabling clean abort if return code < 0
#undef  PLUS      // x+y when numbers are stored as logarithms
#undef  GT        // ">" for extended values

#define CALLOC(p,n,t) {p=NULL;\
 if((n)>0&&!(p=(t*)calloc((size_t)(n),sizeof(t))))\
 {CALLvalue=E_MALLOC;goto Exit;}/*printf("%p %d\n",p,(size_t)(n)*sizeof(t));*/}
#define FREE(p) {if(p){/*printf("%p -1\n",p);*/(void)free((void*)p);} p=NULL;}
#define CALL(x)    {if( (CALLvalue = (x)) < 0 ) goto Exit;}
#define PLUS(x,y)  ((x)>(y)?(x)+log(1.+exp((y)-(x))):(y)+log(1.+exp((x)-(y))))
#define GT(L1,e1, L2,e2) ((L1)>(L2) || ((L1)==(L2) && (e1)>(e2)))

static const double logINF  =  DBL_MAX * DBL_EPSILON;   // ln(inf) = +infinity
static const double logZERO = -DBL_MAX * DBL_EPSILON;   // ln(0)   = -infinity

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestSampler
// 
// Purpose:   Bayesian computation
//
//                   Pr(x)  Pr(D|x)   =    Pr(D)  Pr(x|D)
//                  Prior Likelihood ==> Evidence Posterior
//
//            Use "NestSampler" to obtain:
//
//               lnEvidence  := ln INTEGRAL Likelihood d(Prior),
//                              with its numerical uncertainty;
//
//               Information := ln( prior-to-posterior compression factor )
//                              with its numerical uncertainty;
//            
//               Number      := effective # independent samples during run;
//
//               Nactive     := apparent # active constraints (if Gaussian);
//
//               Posterior properties (optional) as Mean and StdDev,
//                              with numerical uncertainties on each;
//
//               Posterior samples (optional).
//
// History:   John Skilling    4 Sep 2004
//-----------------------------------------------------------------------------
int NestSampler(       //   O  last UserProperties return code, or -ve error
GlobalStr* Global,     // I O  general information, see structure definition
ObjectStr* Objects,    //   O  object collection             [Global->Nobjects]
ObjectStr* Samples,    //  (O) posterior samples                   [MaxSamples]
int        MaxSteps,   // I    max # steps allowed            (-ve is inactive)
double     tol)        // I    fractional tolerance on Evidence (0 is inactive)
{
    int     MaxSamples = Global->MaxSamples;
    int     N          = Global->Nobjects;
    double* Xvec       = Global->PriorMass.values;
    double* Tvec       = Global->Tvec;
    int     edge;           // &(object with minimum L on entry)
    int     k;              // counter
    int     s;              // simulation counter
    int     Nplus;          // power-of-2 >= N
    double  t;
    int     CALLvalue  = 0;

// Allocate
    for( Nplus = 1; Nplus < N; Nplus <<= 1 ) ;
    Global->Nplus = Nplus;
    CALLOC(Global->height, MaxSamples, double)
    CALLOC(Global->Wmat, MaxSamples, Simul_t)
    CALLOC(Global->LminTree, 2*Nplus, ObjectStr*)
    CALLOC(Global->LmaxTree, 2*Nplus, ObjectStr*)

// Setup
    NestSet(Global);
// Prior
    for( k = 0; k < N; k++ )
    {
        CALL( UserPrior(Global, &Objects[k]) )
        NestLhood(Global, Objects, 1, k, &Objects[k]);  // must be OK
    }

// Trapezoidally special initial step
    for( s = 0; s < SIMUL; s++ )
    {
        t = log(Randouble(Global->Rand)) / N;
        Xvec[s] = log(2.0 - exp(t));
        Tvec[s] = t - Xvec[s];
    }

// Convergence steps, ordered by likelihood
    while( Global->Nsteps != MaxSteps )
    {
// update system statistics for each successive edge
        Global->Nsteps++;
        edge = NestReset(Global);
        NestIntegrate(Global, N);
// optional samples and user statistics
        CALL( NestPosterior (Global, &Objects[edge], Samples) )
        CALL( UserProperties(Global, Objects, edge, Samples) )
        NestProperties(Global);
// explore
        CALL( UserExplore(Global, Objects, edge) )
// termination
        if( tol > 0.0  &&  Global->end < log(tol) )
            break;
    }

// Remainder steps, in likelihood order
    for( k = N; k > 0; k-- )
    {
// update system statistics for each survivor
        Global->Nsteps++;
        edge = NestReset(Global);
        NestIntegrate(Global, k);
// optional samples and user statistics
        CALL( NestPosterior (Global, &Objects[edge], Samples) )
        CALL( UserProperties(Global, Objects, edge, Samples) )
        NestProperties(Global);
// shrink
        t = Objects[edge].lnLhood;
        Objects[edge].lnLhood = logINF;
        NestLhood(Global, Objects, 1, edge, &Objects[edge]);
    }

// Trapezoidally special final step
    Global->lnLhood = t;
    for( s = 0; s < SIMUL; s++ )
    {
        Xvec[s] += log(1.0 + exp(Tvec[s]));
        Tvec[s] = logZERO;
    }
// update final system statistics
    NestIntegrate(Global, 1);

Exit:
    FREE(Global->Wmat)
    FREE(Global->height)
    FREE(Global->LmaxTree)
    FREE(Global->LminTree)
    return CALLvalue;  
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestCopy
// 
// Purpose:   Copy complete object,  " Dest := Src ".
// 
// History:   John Skilling        4 Sep 2004
//-----------------------------------------------------------------------------
int NestCopy(          //   O  0, or -ve error
ObjectStr* Dest,       //   O  destination object
ObjectStr* Src)        // I    source object
{
    int  CALLvalue = 0;

    if( Dest != Src )
    {
        Dest->lnLhood = Src->lnLhood;
        Dest->label   = Src->label;
        CALL( UserCopy(Dest->UserObject, Src->UserObject) )
    }
Exit:
    return CALLvalue;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestLhood
// 
// Purpose:   Check if new objects are all inside the box.
//            If all OK, insert each trial object into the system.
// 
// Method:    The NestSampler system uses binary trees of pointers-to-objects
//            to ensure that its internal workings remain O(log N), instead
//            of the simpler O(N) which could possibly dominate the CPU.
//            For example, N=5 Likelihoods = {L0,L1,L2,L3,L4} can be scanned
//            conventionally in O(N) operations.  In NestLhood, preceding
//            Nplus=8 (Nplus = power-of-2 >= N) entries hold partial maxima
//            that facilitate reaching the maximum object fast, in O(log N)
//            operations.  Blanks are pointers to 0 (though actual values are
//            logarithms, augmented by a low-order label).
//               -------------------------------------------------------
//     LmaxTree |1              MAX(L0,L1,L2,L3,L4,0,0,0)               |
//              |-------------------------------------------------------|
//              |2    MAX(L0,L1,L2,L3)      |3     MAX(L4,0,0,0)        |
//              |---------------------------|---------------------------|
//              |4 MAX(L0,L1) |5 MAX(L2,L3) |6 MAX(L4,0)  |7  MAX(0,0)  |
//              |-------------|-------------|-------------|-------------|
//     lnLhood->|8 L0  |9 L1  |10 L2 |11 L3 |12 L4 |13  0 |14  0 |15  0 |
//               -------------------------------------------------------
//            Another tree, called LminTree, locates bottom values by using
//            using MIN instead.  Its blank values are infinity.
//
// History:   John Skilling    4 Sep 2004
//-----------------------------------------------------------------------------
int NestLhood(      //   O  1 if all inside box, otherwise 0 with no action
GlobalStr* Global,  // I O  general information
ObjectStr* Objects, // I O  collection of objects
int        ntry,    // I    # trial object identifiers with trial objects
 ... )              // I    id,object , id,object , ...
{
    int         Nplus    = Global->Nplus;
    ObjectStr** LminTree = Global->LminTree;
    ObjectStr** LmaxTree = Global->LmaxTree;
    ObjectStr*  Object;         // object to be replaced
    ObjectStr*  ObjTry;         // trial object
    int         itry;           // counter for trial objects
    va_list     ap;             // scan variable-argument list
    int         j;              // address in trees

    va_start(ap, ntry);
    for( itry = 0; itry < ntry; itry++ )
    {
        j      = va_arg(ap, int);
        ObjTry = va_arg(ap, ObjectStr*);
        if( GT( Global->lnLhood,Global->label, ObjTry->lnLhood,ObjTry->label ))
            break;                                    // reject
    }
    va_end(ap);
    if( itry < ntry )                                 // reject
        return 0;
                                                      // else accept
    va_start(ap, ntry);
    for( itry = 0; itry < ntry; itry++ )
    {
        j      = va_arg(ap, int);
        Object = &Objects[j];
        ObjTry = va_arg(ap, ObjectStr*);
        NestCopy(Object, ObjTry);
        j += Nplus;
        LminTree[j] = LmaxTree[j] = Object;
        for( ; j > 1; j >>= 1 )
        {
            LminTree[j>>1] = GT( LminTree[j]->lnLhood,   LminTree[j]->label,
                                 LminTree[j^1]->lnLhood, LminTree[j^1]->label )
                            ? LminTree[j^1] : LminTree[j];
            LmaxTree[j>>1] = GT( LmaxTree[j]->lnLhood,   LmaxTree[j]->label,
                                 LmaxTree[j^1]->lnLhood, LmaxTree[j^1]->label )
                            ? LmaxTree[j] : LmaxTree[j^1];
        }
    }
    va_end(ap);
    return 1;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestSet
// 
// Purpose:   Initialise Global statistics
// 
// History:   John Skilling    15 Apr, 24 Aug 2004
//-----------------------------------------------------------------------------
static void NestSet(GlobalStr* Global)
{
    int         NumProps   = Global->NumProps;
    int         Nplus      = Global->Nplus;
    unsigned*   Rand       = Global->Rand;
    double*     Properties = Global->Properties;
    double*     Evec       = Global->lnEvidence.values;
    double*     Ivec       = Global->Information.values;
    double*     Xvec       = Global->PriorMass.values;
    double*     Tvec       = Global->Tvec;
    ObjectStr** LminTree   = Global->LminTree;
    ObjectStr** LmaxTree   = Global->LmaxTree;
    ResultStr*  Mean       = Global->Mean;
    ResultStr*  StdDev     = Global->StdDev;
    double*     Qvec;
    double*     Qdev;
    int         k;                           // counter
    int         s;                           // simulation counter

    Global->Iseed = RanInit(Rand, Global->Iseed);
    RanInit(Global->Rpost, Global->Iseed | 1);
    Global->Nsteps   = 0;
    Global->Nsamples = 0;
    Global->Number   = 1.0;
    Global->step     = 0.0;
    Global->lnLhood  = logZERO;
    Global->label    = 0.0;
    Global->Omin.lnLhood    = logZERO;
    Global->Omin.label      = 0.0;
    Global->Omin.UserObject = NULL;
    Global->Omax.lnLhood    = logINF;
    Global->Omax.label      = 1.0;
    Global->Omax.UserObject = NULL;
    for( s = 0; s < SIMUL; s++ )
    {
        Evec[s] = logZERO;
        Ivec[s] = -logZERO;  // to cancel Evec[s] when NestIntegrate starts
        Xvec[s] = 0.0;
        Tvec[s] = 0.0;
    }
    for( k = 0; k < Nplus + Nplus; k++ )
    {
        LminTree[k] = &Global->Omax;        // lnLedge = +infinity
        LmaxTree[k] = &Global->Omin;        // lnLtop  = -infinity
    }
    for( k = 0; k < NumProps; k++ )
    {
        Properties[k] = 0.0;
        Qvec = Mean[k].values;
        Qdev = StdDev[k].values;
        for( s = 0; s < SIMUL; s++ )
            Qvec[s] = Qdev[s] = 0.0;
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestReset
// 
// Purpose:   Find min and max lnLikelihood values.
//            Return object with min lnLikelihood.
// 
// History:   John Skilling    15 Apr, 4 Sep 2004
//-----------------------------------------------------------------------------
static int NestReset( //   O  &(outer object with min likelihood)
GlobalStr* Global)    // I O  general information
{
    int         Nplus    = Global->Nplus;
    ObjectStr** LminTree = Global->LminTree;
    ObjectStr** LmaxTree = Global->LmaxTree;
    int         j;

    Global->lnLtop   = LmaxTree[1]->lnLhood;
    Global->lnLhood  = LminTree[1]->lnLhood;
    Global->label    = LminTree[1]->label;

// Find edge object with min likelihood
    for( j = 1; j < Nplus; )
    {
        j <<= 1;
        if( LminTree[j] != LminTree[1] )
           j++;
    }
    return j - Nplus;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestIntegrate
// 
// Purpose:   Update Global statistics, integrating Z by trapezoidal rule.
//
// Method:    Each of s = [0,SIMUL) simulations of the exploration i = 0,1,2,..
//            has enclosed prior masses
//                                                   N-1
//              X[i][s] = t  t  ...  t ;  Pr(t) = N t    separately for each s
//                         0  1       i
//            where N is the number of objects in collection.
//            The importance weights are
//                  W[i][s] = L[i] DeltaX[i][s]
//            where L is likelihood, and DeltaX is the local range, defined in
//            this implementation by the trapezoidal rule
//                  DeltaX[i] = (X[i-1] - X[i+1]) / 2
//            (with edges dealt with by external adjustment of X).
//            Importance weights accumulate to give estimates of evidence
//                         now
//                  Z[s] = SUM W[i][s]
//                         i=0
//            and information  I = <log L> - log Z  for each s.
//
// History:   John Skilling    15 Apr 2004, 4 Sep 2004
//-----------------------------------------------------------------------------
static void NestIntegrate(
GlobalStr* Global,       // I O  general information
int        N)            // I    # objects in current collection
{
static const double PIE    =  8.53973422267356706540;   // pi * e
static const double lnHALF = -0.69314718055994530941;   // log(1/2)
    ResultStr* PriorMass= &Global->PriorMass;  // I O  ln(enclosed prior mass)
    ResultStr* lnEvid   = &Global->lnEvidence; // I O  lnEvid with uncertainty
    ResultStr* Info     = &Global->Information;// I O  Info with uncertainty
    double*    Xvec     =  PriorMass->values;  // I O  X = prior mass   [SIMUL]
    double*    Evec     =  lnEvid->values;     // I O  Z = evidence     [SIMUL]
    double*    Ivec     =  Info->values;       // I O  I = <lnLhood>/Z  [SIMUL]
    double*    Tvec     =  Global->Tvec;       // I O  prior mass ratio [SIMUL]
    double*    Pvec     =  Global->Pvec;       //   O  prev evidence    [SIMUL]
    double*    Wvec     =  Global->Wvec;       //   O  W = importance wt[SIMUL]
    double*    Wnew     =  Global->Wnew;       //   O  new weight W/Z   [SIMUL]
    double*    Wold     =  Global->Wold;       //   O  old wt 1-Wnew    [SIMUL]
    double*    Number   = &Global->Number;     //   O  # independent samples
    double*    Nactive  = &Global->Nactive;    //   O  # active constraints
    double*    end      = &Global->end;        //   O  termination test
    unsigned*  Rand     =  Global->Rand;       // I O  generator state
    double     Nobjects =  Global->Nobjects;   // I    # objects in collection
    double     L        =  Global->lnLhood;    // I    edge (minimum) value
    double     Ltop     =  Global->lnLtop;     // I    top  (maximum) value
    double     t;                              // log(random top out of N)
    int        s;                              // simulation counter
    double     wold;
    double     wnew;

    wnew = wold = 0.0;
    for( s = 0; s < SIMUL; s++ )
    {
        t        = log(Randouble(Rand)) / N;
        Wvec[s]  = lnHALF + log(1.0 - exp(t + Tvec[s])) + Xvec[s] + L;
        Xvec[s] += Tvec[s];
        Tvec[s]  = t;
        Ivec[s] += Evec[s];
        Pvec[s]  = Evec[s];
        Evec[s]  = PLUS(Pvec[s], Wvec[s]);
        Wnew[s]  = exp(Wvec[s] - Evec[s]);
        Wold[s]  = exp(Pvec[s] - Evec[s]);
        Ivec[s]  = Wold[s] * Ivec[s] + Wnew[s] * L - Evec[s];
        wold    += Pvec[s];
        wnew    += Evec[s];
    }
    NestStatistics(PriorMass);
    NestStatistics(lnEvid);
    NestStatistics(Info);
    wold = exp((wold - wnew) / SIMUL);
    if( wold > 1.0 )
        wold = 1.0;
    wnew = 1.0 - wold;
    if( wold > 0.0 )
    {
        *Number = pow(*Number / wold, wold);
        if( wnew > 0.0 )
            *Number /= pow(wnew, wnew);
    }
    else
        *Number = 1.0;
    *Nactive = *Number * *Number / (PIE * Nobjects * Nobjects);
    *end = Ltop + PriorMass->mean - lnEvid->mean;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestPosterior
// 
// Purpose:   Update posterior ensemble with new candidate object.
//
// Method:    Draw posterior samples by averaging "<..>" over the SIMUL
//            equivalent runs:
//                 Pr(i)  =  < W[i][s] / Z[s] >
//            for i from start to now.
//
//            At each step, update the posterior ensemble by eroding the
//            existing samples (because each updated Z[s] has diminished)
//            in favour of the new candidate object.
//
// History:   John Skilling    24 Aug 2004
//-----------------------------------------------------------------------------
static 
int NestPosterior(      //   O  >=0 OK, or -ve error code
GlobalStr* Global,      // I    general information
ObjectStr* Object,      // I    candidate object
ObjectStr* Samples)     // I O  posterior samples
{
static const double Z = (unsigned)(-1) + 1.0;
    int       MaxSamples = Global->MaxSamples;          // I
    int       Nsamples   = Global->Nsamples;            // I O
    double    step       = Global->step;                // I O
    double*   height     = Global->height;              //  (O)
    double*   Evec       = Global->lnEvidence.values;   // I
    double*   Pvec       = Global->Pvec;                // I
    double*   Wvec       = Global->Wvec;                // I O
    Simul_t*  Wmat       = Global->Wmat;                // I O
    unsigned* Rand       = Global->Rand;                // I O
    double    t;             // staircase
    double    p;             // old weight of existing samples
    double    q;             // new weight of existing samples
    double    r;             // weight of candidate object
    int       j;             // sample counter
    int       s;             // SIMUL counter
    int       CALLvalue = 0;

    if( MaxSamples > 0 && Samples )
    {
// New staircase heights
        for( j = 0; j < Nsamples; j++ )
        {
            p = q = logZERO;                    // old and new weights
            for( s = 0; s < SIMUL; s++ )
            {
                p = PLUS(p, Wmat[j][s] - Pvec[s]);
                q = PLUS(q, Wmat[j][s] - Evec[s]);
            }
            height[j] = step * exp(q - p);      // sample stair height
        }
        r = 0.0;
        for( s = 0; s < SIMUL; s++ )
            r += exp(Wvec[s] - Evec[s]);
        r /= SIMUL;                             // object stair height
// Max step height
        step = r;                               // candidate object <= one step
        for( j = 0; j < Nsamples; j++ )
            if( step < height[j] )              // existing sample <= one step
                step = height[j];
// Overflow protection
        t = r / step;
        for( j = 0; j < Nsamples; j++ )
            t += height[j] / step;
        if( t > MaxSamples )
            step *= t / MaxSamples;
// New ensemble
        t = (unsigned)Ranint(Rand) / Z;         // safely < 1
        for( j = 0; j < Nsamples; j++ )
        {
            t += height[j] / step;              // erode old sample
            if( t >= 1.0 )                      // staircase
                t -= 1.0;                       // keep ...
            else                                // ... or kill by copying onto
            {
                Nsamples--;
                if( j < Nsamples )
                {
                    CALL( NestCopy(&Samples[j], &Samples[Nsamples]) )
                    height[j] = height[Nsamples];
                    for( s = 0; s < SIMUL; s++ )
                        Wmat[j][s] = Wmat[Nsamples][s];
                }
                j--;
            }
        }
        t += r / step;                          // candidate weight
// Candidate object
        if( t >= 1.0 )                          // staircase
        {                                       // append object
            CALL( NestCopy(&Samples[Nsamples], Object) )
            height[Nsamples] = r;
            for( s = 0; s < SIMUL; s++ )
                Wmat[Nsamples][s] = Wvec[s];
            Nsamples++;
        }
        Global->Nsamples = Nsamples;
        Global->step     = step;
    }
Exit:
    return CALLvalue;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestProperties
// 
// Purpose:   Update mean and std.dev. of properties.
// 
// History:   John Skilling    15 Apr 2004
//-----------------------------------------------------------------------------
static void NestProperties(GlobalStr* Global)
{
    int        n      = Global->NumProps;  // I   #properties
    double*    Prop   = Global->Properties;// I   properties
    double*    Wnew   = Global->Wnew;      // I   individual new weight
    double*    Wold   = Global->Wold;      // I   individual old weight
    ResultStr* Mean   = Global->Mean;      // I O each property mean
    ResultStr* StdDev = Global->StdDev;    // I O each property stddev
    double*    value;
    double*    dev;
    double     x;
    int        j;
    int        s;

    for( j = 0; j < n; j++ )
    {
        value = Mean[j].values;
        dev = StdDev[j].values;
        for( s = 0; s < SIMUL; s++ )
        {
            x = Prop[j] - value[s];
            value[s] += Wnew[s] * x;
            dev[s] = sqrt(Wold[s] * (dev[s] * dev[s] + Wnew[s] * x * x));
        }
        NestStatistics(&Mean[j]);
        NestStatistics(&StdDev[j]);
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestStatistics
// 
// Purpose:   Return central value with uncertainty, using SIMUL samples.
//            Generate both "mean & std.dev." and "median with quartiles".
//            For central value, use  mean  or  median.
//            For uncertainty  , use  dev   or  (quartile3-quartile1)/1.348980.
//
// Note:      The "median with quartiles" choice allows easy rescaling.
// 
// History:   John Skilling    15 Apr, 24 Aug 2004
//-----------------------------------------------------------------------------
static void NestStatistics(
ResultStr* result)           // I O  mean+-stddev, median, quartiles
{
    double* y = result->values;   // independent estimates of result r  [SIMUL]
    double* p[SIMUL];             // &(increasing-order components of vec)
    double* q;
    double  mean, var, diff, wnew;
    int     i, j, k, m;

#if 1    // generate mean with standard deviation
// mean +- stddev
    mean = var = 0.0;
    for( i = 0; i < SIMUL; i++ )
    {
        wnew = 1.0 / (i + 1.0);
        diff = y[i] - mean;
        mean += wnew * diff;
        var = (1.0 - wnew) * (var + wnew * diff * diff);
    }
    result->mean = mean;
    result->dev = sqrt(var);
#endif

#if 1    // generate median with quartiles
// Index y to increasing order
    for( j = 0; j < SIMUL; j++ )
        p[j] = y + j;
    i = SIMUL / 2;
    k = SIMUL - 1;
    for( ; ; )
    {
        if( i > 0 )
            q = p[--i];
        else
        {
            q = p[k];
            p[k--] = p[0];
            if( k == 0 )
            {
                p[0] = q;
                break;
            }
        }
        m = i;
        j = i + i + 1;
        while( j <= k )
        {
            if( j < k && *p[j] < *p[j+1] )
                j++;
            if( *q < *p[j] )
            {
                p[m] = p[j];
                m = j;
                j = j + j + 1;
            }
            else
                j = k + 1;
        }
        p[m] = q;
    }
// median and quartiles (assuming SIMUL divisible by 4)
    j = SIMUL / 2;
    i = j - 1;
    k = SIMUL / 4;
    result->median = (*p[i] + *p[j]) / 2.0;
    result->quartile1 = (*p[i-k] + *p[j-k]) / 2.0;
    result->quartile3 = (*p[i+k] + *p[j+k]) / 2.0;
// minimum and maximum
    result->min = *p[0];
    result->max = *p[SIMUL - 1];
#endif
}
