//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//            Nested Sampling
// 
// Filename:  nest.h
// 
// Purpose:   Header for nest.c.  Documentation for user-accessible procedures.
// 
// History:   JS     4 Sep 2004
//-----------------------------------------------------------------------------
#ifndef NESTSAMPLER
#define NESTSAMPLER

#undef  SIMUL               // # simulations of exploration, DIVISIBLE BY 4
#define SIMUL  100
typedef double Simul_t[SIMUL];

/**************/
/* Structures */
/**************/

typedef struct              // <<<< INDIVIDUAL OBJECT >>>>
{
// External
    double      lnLhood;    //   O  lnLikelihood
    double      label;      //   O  low-order mantissa for lnLhood

// USER object information
    void*       UserObject;
} ObjectStr;


typedef struct              // A RESULT, WITH NUMERICAL UNCERTAINTY
{                           // The "values" simulate statistics of a result
    double      mean;       //   O  arithmetical mean of values
    double      dev;        //   O  std.dev. of values
    double      median;     //   O  median (50%) value of values
    double      quartile1;  //   O  first (25%) quartile of values
    double      quartile3;  //   O  third (75%) quartile of values
    double      min;        //   O  minimum value
    double      max;        //   O  maximum value
    Simul_t     values;     //   O  underlying list of simulated values [SIMUL]
} ResultStr;      // mean ~ median, stddev ~ (quartile3-quartile1)/1.348980


typedef struct              // <<<< GENERAL INFORMATION AND WORKSPACE >>>>
{
// External
    int         Iseed;      // I O  Random seed, +ve = fixed, -ve = time seed
    int         Nobjects;   // I    # objects in evolving collection
    int         NumProps;   // I    # properties to be accumulated
    double*     Properties; // I    properties to be quantified      [NumProps]
    int         MaxSamples; // I    requested # posterior samples (if wanted)
    int         Nsamples;   //   O  actual # posterior samples    (if wanted)
    double      lnLhood;    //   O  edge (minimum)  lnLikelihood
    double      label;      //   O  low-order extension of edge lnLikelihood
    ResultStr   PriorMass;  //   O  ln(enclosed prior mass)
    ResultStr   lnEvidence; //   O  ln(Evidence) = ln Pr(D)
    ResultStr   Information;//   O  <ln(posterior)> = ln(compression factor)
    ResultStr*  Mean;       //   O  <Q> = INTEGRAL dx Pr(x|D) Q(x)   [NumProps]
    ResultStr*  StdDev;     //   O  sqr(INT dx Pr(x|D) (Q(x)-<Q>)^2) [NumProps]
    double      Number;     //   O  effective # independent samples during run
    double      Nactive;    //   O  effective # active constraints
    int         Nsteps;     //   O  total # objects computed and used
    unsigned    Rand[4];    //(I)O random generator for general use

// Internal
    double      lnLtop;     //  (O) top (maximum) lnLikelihood
    double      end;        //  (O) termination test
    double      step;       //  (O) staircase step-height for posterior
    int         Nplus;      //  (O) power-of-2 >= Nobjects
    ObjectStr** LminTree;   //  (O) binary tree for min lnLhood       [2*Nplus]
    ObjectStr** LmaxTree;   //  (O) binary tree for max lnLhood       [2*Nplus]
    ObjectStr   Omin;       //  (O) constant object with minimum Lhood
    ObjectStr   Omax;       //  (O) constant object with maximum Lhood
    Simul_t     Pvec;       //  (O) individual previous evidence        [SIMUL]
    Simul_t     Wnew;       //  (O) individual new weight               [SIMUL]
    Simul_t     Wold;       //  (O) individual old weight               [SIMUL]
    Simul_t     Tvec;       //  (O) individual trapezoidal relation     [SIMUL]
    Simul_t     Wvec;       //  (O) individual importance weight        [SIMUL]
    Simul_t*    Wmat;       //  (O) posterior importance wt [MaxSamples][SIMUL]
    double*     height;     //  (O) staircase heights       [MaxSamples]
    unsigned    Rpost[4];   //  (O) separate random generator for posterior
} GlobalStr;


/**************/
/* Procedures */
/**************/
#ifdef __cplusplus
extern "C" {
#endif

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
//-----------------------------------------------------------------------------
extern int  NestSampler(   //   O  UserProperties return code, or -ve error
    GlobalStr* Global,     // I O  general information
    ObjectStr* Objects,    //   O  object collection         [Global->Nobjects]
    ObjectStr* Samples,    //  (O) posterior samples            [<= MaxSamples]
    int        MaxSteps,   // I    max # steps allowed        (-ve is inactive)
    double     tol);       // I    tolerance on Evidence        (0 is inactive)

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestLhood
// 
// Purpose:   Check if new objects are all inside the box.
//            If all OK, insert each trial object into the system.
//-----------------------------------------------------------------------------
extern int  NestLhood(  //   O  1 if all inside box, otherwise 0 with no action
    GlobalStr* Global,  // I O  general information
    ObjectStr* Objects, // I O  collection of objects
    int        ntry,    // I    # trial object identifiers with trial lnLhood
     ... );             // I    id,lnLhood , id,lnLhood , ...

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  NestCopy
// 
// Purpose:   Copy complete object,  " Dest := Src ".
//-----------------------------------------------------------------------------
extern int NestCopy(       //   O  0, or -ve error from UserCopy
    ObjectStr* Dest,       //   O  destination object
    ObjectStr* Src);       // I    source object

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserPrior
// 
// Purpose:   Set one object, sampled from the prior alone.
//-----------------------------------------------------------------------------
extern int UserPrior(      //   O  >=0 OK, or -ve error code
    GlobalStr* Global,     // I    general information
    ObjectStr* Object);    //   O  object being set

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserExplore
// 
// Purpose:   Re-randomise collection of objects inside the box.
//
//            An object is included in the current "box" if its attributes obey
//                    lnLhood > Global->lnLhood 
//               OR ( lnLhood = Global->lnLhood  AND  label > Global->label ).
//            Basically, this is a constraint on likelihood: the "label"
//            attribute only matters if likelihoods are expected to coincide.
//
//            Please do the best you can to give me likelihoods and labels
//            for  N = Global->Nobjects  objects independently and uniformly
//            (with respect to the prior) sampled inside the box.
//
//            I supply you with an initial collection of  N  objects, N-1 of
//            which should be uniformly distributed inside the box already.
//            The other one, Objects[edge], is on the edge of the box with
//                    Objects[edge].lnLhood = Global->lnLhood
//                 AND  Objects[edge].label = Global->label,
//            and this at least needs to be replaced.
//
//            You can use any method you choose, e.g.
//        (1) Re-sample "edge" object ab initio leaving all others intact;
//        (2) Replace "edge" object with the result of a MCMC exploration from
//            one of the other objects, possibly using remaining objects as
//            stationary guides;
//        (3) Genetic mixing of "edge" object and any others you wish.
//
//            When you have m trial objects Oi,Oj,... which might
//            replace object(s) i,j,...., accept them if and only if
//                 NestLhood(Global, Objects, m, i,Oi, j,Oj, ... ) = 1
//            This mandatory call lets me check that the objects are all
//            correctly inside the box.  If they are, NestLhood then calls
//            NestCopy (thence UserCopy) to copy each of your m trial objects
//            into the collection,
//                 Objects[i] = Oi, Objects[j] = Oj, ...
//            and also resets the binary trees on which my updating relies.
//            If any trial object is not inside the box, NestLhood will
//            return 0 without further action, and you must make sure that
//            the original Objects are preserved.
//-----------------------------------------------------------------------------
extern int UserExplore(    //   O  >=0 OK, or -ve error code
    GlobalStr* Global,     // I    general information
    ObjectStr* Objects,    // I O  collection of objects     [Global->Nobjects]
    int        edge);      // I    edge object with bounding likelihood

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserCopy
// 
// Purpose:   Copy UserObject substructure,  " Dest := Src ".
//            I have already copied the lnLhood and label of the parent object.
//-----------------------------------------------------------------------------
extern int UserCopy(       //   O  >=0 OK, or -ve error code
    void* UserDest,        //   O  destination, to be over-written
    void* UserSrc);        // I    input object, to be copied

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserProperties
// 
// Purpose:   Objects[edge] is the object on the edge of the box,
//            which should be used now before it is overwritten.
//
//            Calculate any of its Properties that are to be quantified.
//
//            You may also attend to diagnostics here.
//-----------------------------------------------------------------------------
extern int UserProperties( //   O  >=0 OK, or -ve error code
    GlobalStr* Global,     // I    general information, incl. Properties list
    ObjectStr* Objects,    // I    object collection         [Global->Nobjects]
    int        edge,       // I    edge object with minimum Lhood
    ObjectStr* Samples);   // I    posterior samples         [Global->Nsamples]

#ifdef __cplusplus
};
#endif

/*************/
/* Constants */
/*************/
#undef  E_MALLOC     // Memory allocation error
#undef  E_RAN_ARITH  // Random generator requires 32 bits integer precision

#define E_MALLOC     -130
#define E_RAN_ARITH  -299

#endif

