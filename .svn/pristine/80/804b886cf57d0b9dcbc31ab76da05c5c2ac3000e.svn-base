/*
 * File:
 *   bank.c
 * Author(s):
 *   Pascal Felber <pascal.felber@unine.ch>
 *   Patrick Marlier <patrick.marlier@unine.ch>
 * Description:
 *   Bank stress test.
 *
 * Copyright (c) 2007-2011.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, version 2
 * of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "hashtable.h"
#include "hashtable_private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include "nv_def.h"
#include "c_io.h"

#ifdef DEBUG
# define IO_FLUSH                       fflush(NULL)
/* Note: stdio is thread-safe */
#endif

#define STORE(addr, value)  store_word((nvword_t *)addr, (nvword_t)value)
#define LOAD(addr)  load_word((nvword_t *)addr)
#define COMMIT  commit_noarg();

#if 0
#define STORE(addr, value) {*addr = value;}
#define LOAD(addr) ({ \
    *addr; \
})
#endif


/*
Credit for primes table: Aaron Krowne
 http://br.endernet.org/~akrowne/
 http://planetmath.org/encyclopedia/GoodHashTablePrimes.html
*/
static const unsigned int primes[] = {
53, 97, 193, 389,
769, 1543, 3079, 6151,
12289, 24593, 49157, 98317,
196613, 393241, 786433, 1572869,
3145739, 6291469, 12582917, 25165843,
50331653, 100663319, 201326611, 402653189,
805306457, 1610612741
};
const unsigned int prime_table_length = sizeof(primes)/sizeof(primes[0]);
const float max_load_factor = 0.65;


/*****************************************************************************/
int
hashtable_expand(struct hashtable *h)
{
    /* Double the size of the table to accomodate more entries */
    struct entry **newtable;
    struct entry *e;
    struct entry **pE;
    unsigned int newsize, i, index;

    /* Check we're not hitting max capacity */

#if 0	
    if ( LOAD(&h->primeindex) == (prime_table_length - 1)) return 0;

    STORE(&(h->primeindex), h->primeindex +1);		

    newsize = primes[LOAD(&h->primeindex)];	

    newtable = (struct entry **)malloc(sizeof(struct entry*) * newsize);
    if (NULL != newtable)
    {
        memset(newtable, 0, newsize * sizeof(struct entry *));
        /* This algorithm is not 'stable'. ie. it reverses the list
         * when it transfers entries between the tables */
        for (i = 0; i < h->tablelength; i++) {

	    STORE(&e , h->table[i]);

            while (NULL != (LOAD(&e))) {


                STORE(&h->table[i], LOAD(&e->next));

                index = indexFor(newsize, LOAD(&e->h));

                STORE(&e->next , newtable[index]);

                STORE(&e->next, newtable[index]); 

                newtable[index] = LOAD(&e);
                
                STORE(&e , h->table[i]);
            }
        }
        free(h->table);
        STORE(&h->table , newtable);
	
	COMMIT;
    }
#endif 

	 if (h->primeindex == (prime_table_length - 1)) return 0;
		newsize = primes[++(h->primeindex)];

	newtable = (struct entry **)malloc(sizeof(struct entry*) * newsize);
	if (NULL != newtable)
	{
    	memset(newtable, 0, newsize * sizeof(struct entry *));
	    /* This algorithm is not 'stable'. ie. it reverses the list
    	 * when it transfers entries between the tables */
	    for (i = 0; i < h->tablelength; i++) {
    	    while (NULL != (e = h->table[i])) {
        	    h->table[i] = e->next;
            	index = indexFor(newsize,e->h);
	            e->next = newtable[index];
    	        newtable[index] = e;
	        }
	    }
	    free(h->table);
    	h->table = newtable;
	}

    /* Plan B: realloc instead */
    else
    {
        newtable = (struct entry **)
                   realloc(h->table, newsize * sizeof(struct entry *));
        if (NULL == newtable) { (h->primeindex)--; return 0; }
        h->table = newtable;
        memset(newtable[h->tablelength], 0, newsize - h->tablelength);
        for (i = 0; i < h->tablelength; i++) {
            for (pE = &(newtable[i]), e = *pE; e != NULL; e = *pE) {
                index = indexFor(newsize,e->h);
                if (index == i)
                {
                    pE = &(e->next);
                }
                else
                {
                    *pE = e->next;
                    e->next = newtable[index];
                    newtable[index] = e;
                }
            }
        }
    }
    h->tablelength = newsize;
    h->loadlimit   = (unsigned int) ceil(newsize * max_load_factor);
    return -1;
}


/*****************************************************************************/

#if 1
int
hashtable_insert(struct hashtable *h, void *k, void *v)
{
    /* This method allows duplicate keys - but they shouldn't be used */
    unsigned int index;
    struct entry *e;
    unsigned int i;
	
    i = (unsigned int)LOAD(&h->entrycount);
    i++;
    STORE(&h->entrycount, i);

    if (LOAD(&h->entrycount) > h->loadlimit)	
    {
        /* Ignore the return value. If expand fails, we should
         * still try cramming just this value into the existing table
         * -- we may not have memory for a larger table, but one more
         * element may be ok. Next time we insert, we'll try expanding again.*/
        hashtable_expand(h);

    }
    e = (struct entry *)malloc(sizeof(struct entry));
    if (NULL == e) { --(h->entrycount); return 0; } /*oom*/

    STORE(&e->h, hash(h,k));
    index = indexFor(h->tablelength,LOAD(&e->h));

    e->k = NULL;
    STORE(&e->k, k);	
	e->k = LOAD(&e->k);

    STORE(&e->v, v);	
    STORE(&e->next, h->table[index]);
    STORE(&h->table[index], e);	

    COMMIT;

    return -1;
}
#endif


#if 0
int
hashtable_insert(struct hashtable *h, void *k, void *v)
{
    /* This method allows duplicate keys - but they shouldn't be used */
    unsigned int index;
    struct entry *e;
    unsigned int i;
	
    i = (unsigned int)LOAD(&h->entrycount);
    i++;
   // STORE(&h->entrycount, i);
   h->entrycount = i;	


    if ((h->entrycount) > h->loadlimit)
    //if (LOAD(&h->entrycount) > h->loadlimit)	
    {
        /* Ignore the return value. If expand fails, we should
         * still try cramming just this value into the existing table
         * -- we may not have memory for a larger table, but one more
         * element may be ok. Next time we insert, we'll try expanding again.*/
        hashtable_expand(h);
    }
    e = (struct entry *)malloc(sizeof(struct entry));
    if (NULL == e) { --(h->entrycount); return 0; } /*oom*/

    //e->h = hash(h,k);
    STORE(&e->h, hash(h,k));
    e->h = LOAD(&e->h);	
    //index = indexFor(h->tablelength,e->h);
    index = indexFor(h->tablelength,LOAD(&e->h));

    //e->k = k;
    STORE(&e->k, k);	
    if(h->entrycount == 65)	
	fprintf(stdout,"key addr %lu\n",k);

    //e->v = v;
    STORE(&e->v, v);	

    //e->next = h->table[index];
    STORE(&e->next, h->table[index]);

    //h->table[index] = e;
    STORE(&h->table[index], e);	

    COMMIT;

    return -1;
}
#endif


/*****************************************************************************/
#if 0
hashtable_insert(struct hashtable *h, void *k, void *v)
{
    /* This method allows duplicate keys - but they shouldn't be used */
    unsigned int index;
    unsigned int i;
    struct entry *e;

    i = (unsigned int)LOAD(&h->entrycount);
    i++;
    STORE(&h->entrycount, i);

    if (LOAD(&h->entrycount) > h->loadlimit)
    {
        /* Ignore the return value. If expand fails, we should
         * still try cramming just this value into the existing table
         * -- we may not have memory for a larger table, but one more
         * element may be ok. Next time we insert, we'll try expanding again.*/
        hashtable_expand(h);
    }
    STORE(&e , (struct entry *)malloc(sizeof(struct entry)));

    if (NULL == e) {

    	i =  LOAD(&h->entrycount);
    	i--;
    	STORE(&h->entrycount, i);
    	return 0;
    } /*oom*/

    STORE(&e->h ,hash(h,k));
    index = indexFor(LOAD(&h->tablelength),LOAD(&e->h));
    STORE(&e->k, k);
    STORE(&e->v, v);
    STORE(&e->next ,h->table[index]);
    STORE(&h->table[index],e);

    //COMMIT;	
    return -1;
}
#endif


/*****************************************************************************/
struct hashtable *
create_hashtable(unsigned int minsize,
                 unsigned int (*hashf) (void*),
                 int (*eqf) (void*,void*))
{
    struct hashtable *h;
    unsigned int pindex, size = primes[0];
    char *s;
    char *cm = NULL;


    /* Check requested hashtable isn't too large */
    if (minsize > (1u << 30)) return NULL;
    /* Enforce size as prime */
    for (pindex=0; pindex < prime_table_length; pindex++) {
        if (primes[pindex] > minsize) { size = primes[pindex]; break; }
    }

    h = (struct hashtable *)malloc(sizeof(struct hashtable));
    if (NULL == h) return NULL; /*oom*/
    h->table = (struct entry **)malloc(sizeof(struct entry*) * size);
    if (NULL == h->table) { free(h); return NULL; } /*oom*/
    memset(h->table, 0, size * sizeof(struct entry *));

	
    //h->tablelength  = size;
    //h->primeindex   = pindex;
    //h->entrycount = 0;	
	//h->hashfn = hashf;	
	//h->eqfn = eqf;	
	// h->loadlimit    = (unsigned int) ceil(size * max_load_factor);	

    STORE(&h->tablelength, size);
    STORE(&h->primeindex, pindex);
    STORE(&h->entrycount, 0);
    STORE(&h->hashfn,hashf);
    STORE(&h->eqfn, eqf);
    STORE(&h->loadlimit, (unsigned int) ceil(size * max_load_factor));
    COMMIT;	
    return h;
}

/*****************************************************************************/
unsigned int
hash(struct hashtable *h, void *k)
{
    /* Aim to protect against poor hash functions by adding logic here
     * - logic taken from java 1.4 hashtable source */
 
    //void *fn = LOAD(&h->hashfn);		
    unsigned int i = h->hashfn(k); // fn(k);
    i += ~(i << 9);
    i ^=  ((i >> 14) | (i << 18)); /* >>> */
    i +=  (i << 4);
    i ^=  ((i >> 10) | (i << 22)); /* >>> */
    return i;
}

/*****************************************************************************/
unsigned int
hashtable_count(struct hashtable *h)
{
    return h->entrycount;
}

/*****************************************************************************/
void * /* returns value associated with key */
hashtable_search(struct hashtable *h, void *k)
{
    struct entry *e;
    unsigned int hashvalue, index;
    hashvalue = hash(h,k);
    index = indexFor(h->tablelength,hashvalue);
    e = h->table[index];
    while (NULL != e)
    {
        /* Check hash value to short circuit heavier comparison */
        if ((hashvalue == e->h) && (h->eqfn(k, e->k))) return e->v;
        e = e->next;
    }
    return NULL;
}

/*****************************************************************************/
void * /* returns value associated with key */
hashtable_remove(struct hashtable *h, void *k)
{
    /* TODO: consider compacting the table when the load factor drops enough,
     *       or provide a 'compact' method. */

    struct entry *e;
    struct entry **pE;
    void *v;
    unsigned int hashvalue, index;

    hashvalue = hash(h,k);
    index = indexFor(h->tablelength,hash(h,k));
    pE = &(h->table[index]);
    e = *pE;
    while (NULL != e)
    {
        /* Check hash value to short circuit heavier comparison */
        if ((hashvalue == e->h) && (h->eqfn(k, e->k)))
        {
            *pE = e->next;
			h->entrycount = LOAD(&h->entrycount);
            STORE(&h->entrycount, h->entrycount-1);
            v = e->v;
			STORE(&e->k,NULL);
            freekey(e->k);
			STORE(&e,NULL);
            free(e);
	 		COMMIT; 
            return v;
        }
        pE = &(e->next);
        e = e->next;
	}
    return NULL;
}



/*****************************************************************************/
/* destroy */
void
hashtable_destroy(struct hashtable *h, int free_values)
{
    unsigned int i;
    struct entry *e, *f;
    struct entry **table = h->table;
    if (free_values)
    {
        for (i = 0; i < h->tablelength; i++)
        {
            e = table[i];
            while (NULL != e)
            { f = e; e = e->next; freekey(f->k); free(f->v); free(f); }
        }
    }
    else
    {
        for (i = 0; i < h->tablelength; i++)
        {
            e = table[i];
            while (NULL != e)
            { f = e; e = e->next; freekey(f->k); free(f); }
        }
    }
    free(h->table);
    free(h);
}

struct hashtable *
restart_hashtable(unsigned int minsize,
                 unsigned int (*hashf) (void*),
                 int (*eqf) (void*,void*))
{
       return NULL;
}



/*
 * Copyright (c) 2002, Christopher Clark
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the original author; nor the names of any contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


