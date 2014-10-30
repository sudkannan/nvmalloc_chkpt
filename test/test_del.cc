/******************************************************************************
* FILE: mergesort.c
* DESCRIPTION:  
*   The master task distributes an array to the workers in chunks, zero pads for equal load balancing
*   The workers sort and return to the master, which does a final merge
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <fcntl.h>
#include <math.h>
#include <sched.h>
#include <nv_map.h>
#include <c_io.h>

  struct ByteLoggingWriteSetEntry
  {
      union {
          void**   addr;
          uint8_t* byte_addr;
      };

      union {
          void*   val;
          uint8_t byte_val[sizeof(void*)];
      };

      union {
          uintptr_t mask;
          uint8_t   byte_mask[sizeof(void*)];
      };

      ByteLoggingWriteSetEntry(void** paddr, void* pval, uintptr_t pmask)
      {
          addr = paddr;
          val  = pval;
          mask = pmask;
      }

      /**
       *  Called when we are WAW an address, and we want to coalesce the
       *  write. Trivial for the word-based writeset, but complicated for the
       *  byte-based version.
       *
       *  The new value is the bytes from the incoming log injected into the
       *  existing value, we mask out the bytes we want from the incoming word,
       *  mask the existing word, and union them.
       */
      void update(const ByteLoggingWriteSetEntry& rhs)
      {
    	  fprintf(stdout,"DEBUG WriteSet::update ByteLoggingWriteSetEntr"
    			  "  updating value\n");
          // fastpath for full replacement
          if (__builtin_expect(rhs.mask == (uintptr_t)~0x0, true)) {
              val = rhs.val;
              mask = rhs.mask;
          }

          // bit twiddling for awkward intersection, avoids looping
          uintptr_t new_val = (uintptr_t)rhs.val;
          new_val &= rhs.mask;
          new_val |= (uintptr_t)val & ~rhs.mask;
          val = (void*)new_val;

          // the new mask is the union of the old mask and the new mask
          mask |= rhs.mask;
      }

      /**
       *  Check to see if the entry is completely contained within the given
       *  address range. We have some preconditions here w.r.t. alignment and
       *  size of the range. It has to be at least word aligned and word
       *  sized. This is currently only used with stack addresses, so we
       *  don't include asserts because we don't want to pay for them in the
       *  common case writeback loop.
       *
       *  The byte-logging writeset can actually accommodate awkward
       *  intersections here using the mask, but we're not going to worry
       *  about that given the expected size/alignment of the range.
       */
      bool filter(void** lower, void** upper)
      {
          return !(addr + 1 < lower || addr >= upper);
      }

      /**
       *  If we're byte-logging, we'll write out each byte individually when
       *  we're not writing a whole word. This turns all subword writes into
       *  byte writes, so we lose the original atomicity of (say) half-word
       *  writes in the original source. This isn't a correctness problem
       *  because of our transactional synchronization, but could be a
       *  performance problem if the system depends on sub-word writes for
       *  performance.
       */
      void writeback() const
      {
          if (__builtin_expect(mask == (uintptr_t)~0x0, true)) {
              *addr = val;
              return;
          }

          // mask could be empty if we filtered out all of the bytes
          if (mask == 0x0)
              return;

          // write each byte if its mask is set
          for (unsigned i = 0; i < sizeof(val); ++i)
              if (byte_mask[i] == 0xff)
                  byte_addr[i] = byte_val[i];
      }

      /**
       *  Called during the rollback loop in order to write out buffered
       *  writes to an exception object (represented by the address
       *  range). We don't assume anything about the alignment or size of the
       *  exception object.
       */
      void rollback(void** lower, void** upper)
      {
          // two simple cases first, no intersection or complete intersection.
          if (addr + 1 < lower || addr >= upper)
              return;

          if (addr >= lower && addr + 1 <= upper) {
              writeback();
              return;
          }

          // odd intersection
          for (unsigned i = 0; i < sizeof(void*); ++i) {
              if ((byte_mask[i] == 0xff) &&
                  (byte_addr + i >= (uint8_t*)lower ||
                   byte_addr + i < (uint8_t*)upper))
                  byte_addr[i] = byte_val[i];
          }
      }
  };




/*************************************************************/
/*          Example of writing arrays in NVCHPT               */
/*                                                           */
/*************************************************************/
int main (int argc, char ** argv)
{
    char        filename [256];
    int         rank, size, i, j;
    int         NX = 10, NY = 100;
    double      t[NX][NY];
    int         *p;
    int mype = 0;

     ByteLoggingWriteSetEntry *a;// = (ByteLoggingWriteSetEntry *)malloc(sizeof(struct ByteLoggingWriteSetEntry));

     fprintf(stdout,"sizeof(ByteLoggingWriteSetEntry) %u\n",
			sizeof(a));


	mype = atoi(argv[1]);
	a = (ByteLoggingWriteSetEntry *)nvread_((char*)"list", mype);
        assert(a);

	for(unsigned int i=0;i<64;i++) {
		fprintf(stdout,"DEBUG WriteSet list.addr %lu\n",(unsigned long)a->addr);
		void *temp = (void*)a;
		temp += 16;
		a = (ByteLoggingWriteSetEntry *)temp;
	}


    return 0;
}





























































