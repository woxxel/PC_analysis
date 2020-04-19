//=============================================================================
//
//                  NESTED-SAMPLING EXAMPLE PROGRAM  sphere.c
//
// Link
//    sphere.c = {main,UserPrior,UserExplore,UserCopy,UserProperties}   *USER*
//    userstr.h = {definition of user structures and any error codes}   *USER*
// with
//    nest.c, nest.h                                                  *SYSTEM*
//    random.c, random.h                                              *SYSTEM*
//=============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nest.h"          // My header
#include "userstr.h"       // Your header
#include "random.h"        // Random library

int main(void)
{
// System parameters
    int     Nobjects   = 10;      // # objects in internal collection
    int     MaxSamples = 12;      // max # posterior Samples (can be 0)
    int     MaxSteps   = -1;      // +ve is limit (-ve is inactive)
    double  tol        = 1e-6;    // numerical tolerance (0 is inactive)
    int     Iseed      = 4321;    // random seed (-ve is time seed)
// User application parameters and/or Data
    int     Ndim       = 40;      // application is 40-dimensional,...
    double  Radius     = 50.0;    // ... within a sphere of radius 50.
 
    GlobalStr      Global[1];     // system information
    ParamStr       Param[1];      // common parameters
    ObjectStr*     Objects;       // objects in working collection  [Nobjects]
    UserObjectStr* UserObjects;   // user information               [Nobjects]
    ObjectStr*     Samples;       // posterior Samples, if needed [MaxSamples]
    UserObjectStr* UserSamples;   // user information             [MaxSamples]
    ResultStr*     result;        // pointer to result structure
    int            code;          // NestSampler return code (-ve is error)
    double         Z;             // true lnEvidence, for comparison
    int            j;             // counter

// System
    Global->Iseed      = Iseed;
    Global->Nobjects   = Nobjects;
    Global->NumProps   = 2;
    Global->MaxSamples = MaxSamples;
    Global->Properties = malloc(Global->NumProps * sizeof(double));
    Global->Mean       = malloc(Global->NumProps * sizeof(ResultStr));
    Global->StdDev     = malloc(Global->NumProps * sizeof(ResultStr));
// User application
    Param->Ndim   = Ndim;
    Param->Radius = Radius;
// Create Objects and (optionally) Samples
    Objects     = malloc(Nobjects * sizeof(ObjectStr));
    UserObjects = malloc(Nobjects * sizeof(UserObjectStr));
    for( j = 0; j < Nobjects; j++ )
    {
        Objects[j].UserObject = &UserObjects[j];
        UserObjects[j].x      = malloc(Ndim * sizeof(double));
        UserObjects[j].Param  = Param;
    }
    Samples     = malloc(MaxSamples * sizeof(ObjectStr));
    UserSamples = malloc(MaxSamples * sizeof(UserObjectStr));
    for( j = 0; j < MaxSamples; j++ )
    {
        Samples[j].UserObject = &UserSamples[j];
        UserSamples[j].x      = malloc(Ndim * sizeof(double));
        UserSamples[j].Param  = Param;
    }

// Run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    code = NestSampler(Global, Objects, Samples, MaxSteps, tol);
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

// Results
    printf("\nReturn code = %d\n", code);
    if( code < 0 )
        printf("\a ERROR STATE\n");
    else
    {
        Z = (Ndim & 1) ? 0.5 * log(3.14159265 / (2.0 * Radius * Radius)) : 0.0;
        for( j = Ndim; j > 1; j -= 2 )
            Z += log(j / (Radius * Radius));
        printf("Seed        = %d\n", Global->Iseed);
        printf("Nsteps      = %d\n\n", Global->Nsteps);

        result = &Global->lnEvidence;
        printf("lnEvidence  =%10.4f (+-%7.4f) from mean and std.dev.\n",
            result->mean, result->dev);
        printf("lnEvidence  =%10.4f (+-%7.4f) from median and quartiles\n",
            result->median, (result->quartile3 - result->quartile1) / 1.34898);
        printf("  should be  %10.4f\n\n", Z);

        result = &Global->Information;
        printf("Information =%10.4f (+-%7.4f) from mean and std.dev.\n",
            result->mean, result->dev);
        printf("Information =%10.4f (+-%7.4f) from median and quartiles\n",
            result->median, (result->quartile3 - result->quartile1) / 1.34898);
        printf("  should be  %10.4f\n\n",
            -0.5 * Ndim - Z);

        printf("Effective # active constraints = %6.0f", Global->Nactive);
        printf("  should be  %d\n", Ndim);

        printf("Effective independent objects  = %6.0f\n", Global->Number);

// Sample coordinates should be from unit Gaussian in this application
        for( j = 0; j < Global->Nsamples; j++ )
        {
            printf("    Sample%2d x[0] = %7.4f   Radius^2 = %7.2f\n",
                    j, UserSamples[j].x[0], -2.0 * Samples[j].lnLhood);
        }

// Statistics are accumulated during the run better than from the samples
        printf("Posterior Rsquared =%9.4f +-%7.4f (mean estimates)\n",
                    Global->Mean[0].mean, Global->StdDev[0].mean);
        printf("Posterior Rsquared =%9.4f +-%7.4f (median estimates)\n",
                    Global->Mean[0].median, Global->StdDev[0].median);
        printf("          should be %4d.0000 +-%7.4f\n\n",
                    Ndim, sqrt(2.0 * Ndim));

        printf("Poaterior   x[0]   =%9.4f +-%7.4f (mean estimates)\n",
                    Global->Mean[1].mean, Global->StdDev[1].mean);
        printf("Posterior   x[0]   =%9.4f +-%7.4f (median estimates)\n",
                    Global->Mean[1].median, Global->StdDev[1].median);
        printf("          should be         0 +- 1\n");
    }
    for( j = 0; j < MaxSamples; j++ )
        free(UserSamples[j].x);
    free(UserSamples);
    free(Samples);
    for( j = 0; j < Nobjects; j++ )
        free(UserObjects[j].x);
    free(UserObjects);
    free(Objects);
    free(Global->StdDev);
    free(Global->Mean);
    free(Global->Properties);
    return 0;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserPrior
// 
// Purpose:   Set one object sampled from the prior alone.
//
// Example:   This code uses flat prior in Ndim-dimensional sphere of
//            radius Radius and volume (pi Radius^2)^(Ndim/2) / (Ndim/2)!
//            Likelihood is Gaussian: at intermediate radius r,
//                lnLikelihood = - r^2 / 2
//
//            This has sufficient symmetry to enable exact ab initio sampling.
//            Correct results for large Radius (> sqrt(Ndim)+3 suffices) are
//                lnEvidence  = ln( 2^(Ndim/2) (Ndim/2)! / Radius^Ndim )
//                Information = - Ndim/2 - lnEvidence
//            e.g. Ndim = 40,  Radius = 50,
//                lnEvidence  = -100.2824
//                Information =   80.2824, a factor exp(80.2824) compression
// 
// History:   JS        24 Aug 2004
//-----------------------------------------------------------------------------
int UserPrior(               //   O  >=0 OK, or -ve error code
GlobalStr* Global,           // I    general information
ObjectStr* Object)           //   O  object being set
{
    UserObjectStr* UserObject = Object->UserObject;
    ParamStr*      Param = UserObject->Param;
    int            Ndim  = Param->Ndim;        // dimension
    double         a     = Param->Radius;      // prior (maximum) radius
    unsigned*      Rand  = Global->Rand;       // random generator
    double*        x     = UserObject->x;      // shortcut to coordinates
    double         rr;                         // local radius^2
    double         s;                          // scale factor
    int            i;

    if( a < sqrt((double)Ndim) + 3.0 )
        return E_ANALYSIS;    // example of abort with user's -ve return code
// New radius enclosing fractional volume from Uniform[0,1]
    a *= pow(Randouble(Rand), 1.0 / Ndim);
// Sample from unit Gaussian ball
    for( i = 0; i < Ndim; i++ )
        x[i] = Rangauss(Rand);
    rr = 0.0;    for( i = 0; i < Ndim; i++ )    rr += x[i] * x[i];
// Scale onto new radius
    s = a / sqrt(rr);    for( i = 0; i < Ndim; i++ )    x[i] *= s;
// New lnLikelihood
    Object->lnLhood = -0.5 * a * a;
    Object->label = Randouble(Rand);
    return  0;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserExplore
// 
// Purpose:   Re-randomise object(s) within box lnLikelihood >= Global->lnLmin
//
// History:   JS        15 Apr, 4 Sep 2004
//-----------------------------------------------------------------------------
int UserExplore(             //   O  # successful moves, or -ve error code
GlobalStr* Global,           // I    general information
ObjectStr* Objects,          // I O  collection of objects   [Global->Nobjects]
int        edge)             // I    edge object with bounding likelihood
{
    unsigned*      Rand    = Global->Rand;         // random generator
    double         lnLedge = Global->lnLhood;      // bounding lnLikelihood
    double         radius  = sqrt(-2.0 * lnLedge); // bounding radius
// Old object
    ObjectStr*     OldObj  = &Objects[edge];
    UserObjectStr* OldUser = OldObj->UserObject;
    ParamStr*      Param   = OldUser->Param;
    double*        x       = OldUser->x;  // shortcut to coordinates
// Miscellaneous
    int            Ndim = Param->Ndim;    // dimension
    double         s;                     // Markov chain Monte Carlo step-size
    double         rr;                    // trial radius^2
    int            i;                     // coordinate counter
    int            nsuccess = 0;          // # of successful moves (diagnostic)

#if 1   // EITHER MCMC APPROXIMATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    int            N = Global->Nobjects;  // # objects in collection
    int            copy;                  // seed object for MCMC
    int            MCMC;                  // Markov chain Monte Carlo counter
// Declare trial object
    ObjectStr      TryObj[1];
    UserObjectStr  TryUser[1];
    double*        try;                   // trial position
    TryUser->x = try = malloc(Ndim * sizeof(double));
    TryUser->Param = Param;
    TryObj->UserObject = TryUser;
// Find random alternate object and duplicate it as initial suggestion
    if( N > 1 )
    {
        do     copy = Rangrid(Rand, (unsigned)(N-1));
        while( copy == edge );
        NestCopy(&Objects[edge], &Objects[copy]);
    }
// MCMC step length, optimised for speed
    s = radius * 1.8507 / Ndim;
// Guess number of steps
    for( MCMC = 0; MCMC < 40; MCMC++ )
    {
// Trial object
        for( i = 0; i < Ndim; i++ )
            try[i] = x[i] + s * Rangauss(Rand);
        rr = 0.0;    for( i = 0; i < Ndim; i++ )    rr += try[i] * try[i];
        TryObj->lnLhood = -0.5 * rr;
        TryObj->label = Randouble(Rand);
// Accept/reject: if all OK, trial object(s) are copied into collection
        nsuccess += NestLhood(Global, Objects, 1, edge, TryObj);
    }
    free(try);

#else   // OR EXACT AB INITIO SAMPLING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// New object radius
    radius *= pow(Randouble(Rand), 1.0 / Ndim);
// Sample from unit Gaussian ball
    for( i = 0; i < Ndim; i++ )
        x[i] = Rangauss(Rand);
    rr = 0.0;    for( i = 0; i < Ndim; i++ )    rr += x[i] * x[i];
// Scale onto new radius
    s = radius / sqrt(rr);    for( i = 0; i < Ndim; i++ )    x[i] *= s;
// Use new object, guaranteed inside bounding contour
    OldObj->lnLhood = -0.5 * radius * radius;
    OldObj->label = Randouble(Rand);
    nsuccess += NestLhood(Global, Objects, 1, edge, OldObj);

#endif  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return  nsuccess;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserCopy
// 
// Purpose:   Copy UserObject substructure,  " dest := src ".
// 
// History:   JS        15 Apr, 4 Sep 2004
//-----------------------------------------------------------------------------
int UserCopy(           //   O  >=0 OK, or -ve error code
void* dest,             //   O  destination, to be over-written
void* src)              // I    input object, to be copied
{
    UserObjectStr* UserSrc  = src;
    UserObjectStr* UserDest = dest;
    ParamStr*      Param    = UserSrc->Param;
    int            Ndim     = Param->Ndim;
    double*        in       = UserSrc->x;
    double*        out      = UserDest->x;
    int            i;

    for( i = 0; i < Ndim; i++ )
        out[i] = in[i];
    return 0;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  UserProperties
// 
// Purpose:   Properties and diagnostics.
//
// History:   JS        15 Apr, 4 Sep 2004
//-----------------------------------------------------------------------------
int UserProperties(          //   O  >=0 OK, or -ve error code
GlobalStr* Global,           // I    general information, incl. Properties list
ObjectStr* Objects,          // I    object collection       [Global->Nobjects]
int        edge,             // I    edge object with minimum Lhood
ObjectStr* Samples)          // I    posterior samples       [Global->Nsamples]
{
    ObjectStr*     Object = &Objects[edge];
    UserObjectStr* pUserObject;

// Set desired properties, e.g.....
    pUserObject = Object->UserObject;
    Global->Properties[0] = -2.0 * Global->lnLhood; // radius-squared
    Global->Properties[1] = pUserObject->x[0];      // one of the coordinates

// Arbitrary diagnostics
    printf(
     "\nStep#%6d  lnEvid =%8.2f  lnLhood =%8.2f =%8.2f  lnPriorMass = %8.4f\n",
              Global->Nsteps, Global->lnEvidence.mean,
              Global->lnLhood, Object->lnLhood, Global->PriorMass.mean);
    pUserObject = Samples[0].UserObject;
    printf("    Sample#0 x[0] = %7.4f   Radius^2 = %7.2f\n",
            pUserObject->x[0], -2.0 * Samples[0].lnLhood);
    return 0;
}
