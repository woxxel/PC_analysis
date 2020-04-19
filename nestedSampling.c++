/*============================================================= TOY NESTED SAMPLING PROGRAM IN ‘C’ by John Skilling, Aug 2005 GNU GENERAL PUBLIC LICENSE software http://www.gnu.org/copyleft/gpl.html =============================================================*/ 

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <float.h> 
#define UNIFORM
((rand()+0.5) / (RAND MAX+1.0)) // Uniform(0,1)
#define logZERO (-DBL MAX * DBL EPSILON)  // log(0)
#define PLUS(x,y) (x>y ? x+log(1+exp(y-x)) : y+log(1+exp(x-y))) // logarithmic addition log(exp(x)+exp(y))
/* YOU MUST PROGRAM THIS FROM HERE ++++++++++++++++++++++++++++++ 
typedef struct {
ANYTYPE theta;    // YOUR coordinates 
double logL;      // logLikelihood = ln Prob(data | theta) 
double logWt;     // ln(Weight), summing to SUM(Wt) = Evidence Z
} Object; 
double logLhood(ANYTYPE theta){...}               // logLikelihood function
void Prior (Object* Obj){...}                     // Set Object according to prior
void Explore(Object* Obj, double logLstar){...}   // Evolve Object within likelihood constraint
--------------------------------------------------- UP TO HERE */

int main(void) {
#define N 100           // # Objects
#define MAX 9999        // max # Samples (allow enough)
Object Obj[N];          // Collection of N objects
Object Samples[MAX];    // Objects defining posterior
double logw;            // ln(width in prior mass)
double logLstar;        // ln(Likelihood constraint)
double H = 0.0;         // Information, initially 0
double logZ = logZERO;  // ln(Evidence Z, initially 0)
double logZnew;         // Updated logZ
int i;                  // Object counter
int copy;               // Duplicated object
int worst;              // Worst object
int nest;               // Nested sampling iteration count 
double end = 2.0;       // Termination condition nest = end * N * H

// Set prior objects 
for( i = 0; i < N; i++ ) 
  Prior( &Obj[i] );
// Outermost interval of prior mass 
logw = log(1.0 - exp(-1.0 / N));
// Begin Nested Sampling loop +++++++++++++++++++++++++++++++++++ 
for( nest = 0; nest <= end * N * H; nest++ ) {

// Worst object in collection, with Weight = width * Likelihood 
  worst = 0; 
  for( i = 1; i < N; i++ ) 
    if( Obj[i].logL < Obj[worst].logL ) 
      worst = i;

  Obj[worst].logWt = logw + Obj[worst].logL;

// Update Evidence Z and Information H 
  logZnew = PLUS(logZ, Obj[worst].logWt); 
  H = exp(Obj[worst].logWt - logZnew) * Obj[worst].logL + exp(logZ - logZnew) * (H + logZ) - logZnew;
  logZ = logZnew;
  
// Posterior Samples (optional, care with storage overflow) 
  Samples[nest] = Obj[worst];
// Kill worst object in favour of copy of different survivor 
  do copy = (int)(N * UNIFORM) % N;     // force 0 <= copy < N
  while( copy == worst && N > 1 );      // don’t kill if N=1
  
  logLstar = Obj[worst].logL;           // new likelihood constraint
  Obj[worst] = Obj[copy];               // overwrite worst object
// Evolve copied object within constraint 
  Explore( &Obj[worst], logLstar );
// Shrink interval 
  logw -= 1.0 / N;
} // -------------------------------- end nested sampling loop

// Begin optional final correction, should be small +++++++++++++ 
logw = -(double)nest / (double)N - log((double)N);    // width
for( i = 0; i < N; i++ ) {
  Obj[i].logWt = logw + Obj[i].logL; // width * Likelihood
// Update Evidence Z and Information H 
  logZnew = PLUS(logZ, Obj[i].logWt); 
  H = exp(Obj[i].logWt - logZnew) * Obj[i].logL + exp(logZ - logZnew) * (H + logZ) - logZnew;
  logZ = logZnew;
// Posterior Samples (optional, care with storage overflow) 
  Samples[nest++] = Obj[i];
} // --------------------------- end optional final correction

// Exit with evidence Z, information H, and posterior Samples 
printf("#samples = %d\n", nest); 
printf("Evidence: ln(Z) = %g +- %g\n", logZ, sqrt(H/N)); 
printf("Information: H = %g nats = %g bits\n", H, H/log(2.));
// You can now accumulate results from Samples[0...nest-1] 
return 0;
}