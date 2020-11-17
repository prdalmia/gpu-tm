#ifndef __PARDHP_H
#define __PARDHP_H
#define TM

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdint.h>

  typedef struct TRANSACTION {
    int tid;
    int numitem;
    int *item_list;
    int offset;
  } transaction;



  using namespace std;





#define CACHE_LNSIZ 128 //128 bytes
  extern int nthreads;
  extern int *hash_indx;
  extern struct timeval tp;

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

  class prof{
  public:
    double gen;
    double subset;
    double reduce;
    double large;
  };
#ifdef __cplusplus
}
#endif

extern //TM_PURE
	uint64_t rdtsc();

#endif //__PARDHP_H
