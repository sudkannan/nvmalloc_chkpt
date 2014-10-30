/*-
 * Copyright 1997-1998, 2001 John-Mark Gurney.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	$Id: bt_output.c,v 1.3.2.2 2001/03/28 07:28:42 jmg Exp $
 *
 */

#include <btreepriv.h>

#include <limits.h>
#include <stdio.h>

static void dumpnode(struct btree *btr, struct btreenode *, int);
static int checkbtreenode(struct btree *btr, struct btreenode *x, void *kmin,
			  void *kmax, int isroot);
int treeheight(struct btree *btr);

void
bt_dumptree(struct btree *btr)
{
	bt_treestats(btr);

	if (btr->root != NULL)
		dumpnode(btr, btr->root, INT_MAX);
}

static void
dumpnode(struct btree *btr, struct btreenode *n, int nn)
{
	int i;

	printf("%p: leaf: %d, n: %d", n, n->leaf, n->n);

	for (i = 0; i < n->n; i++)
		printf(", key%d: %p", i, KEYS(btr, n)[i]);

	if (!n->leaf) {
		for (i = 0; i <= n->n; i++)
			printf(", nodeptr%d: %p", i, NODES(btr, n)[i]);
		puts("");

		if (nn) {
			nn--;
			for (i = 0; i <= n->n; i++)
				dumpnode(btr, NODES(btr, n)[i], nn);
		}
	} else
		puts("");
}

void
bt_treestats(struct btree *btr)
{
	printf("root: %p, keyoff: %d, nodeptroff: %d, nkeys: %d, t: %d, nbits: %d, textra: %d, height: %d",
		btr->root, btr->keyoff, btr->nodeptroff, btr->nkeys, btr->t, btr->nbits, btr->textra, treeheight(btr));

#ifdef STATS
	printf(", numkeys: %d, numnodes: %d", btr->numkeys, btr->numnodes);
#endif
	puts("");
}

int
bt_checktree(struct btree *btr, void *kmin, void *kmax)
{
	return checkbtreenode(btr, btr->root, kmin, kmax, 1);
}

static int
checkbtreenode(struct btree *btr, struct btreenode *x, void *kmin, void *kmax,
    int isroot)
{
	int i;

	if (x == NULL)
		/* check that the two keys are in order */
		if (btr->cmp(kmin, kmax) >= 0)
			return 0;
		else
			return 1;
	else {
		if (!isroot && (x->n < btr->t - 1 || x->n > 2 * btr->t - 1)) {
			printf("node, to few or to many: %d\n", x->n);
			bt_dumptree(btr);
			exit(1);
		}
		/* check subnodes */
		if (x->n == 0 && !x->leaf)
			if (!checkbtreenode(btr, NODES(btr, x)[0], kmin, kmax,
			    0))
				return 0;
			else
				return 1;
		else if (x->n == 0 && x->leaf && !isroot) {
			printf("leaf with no keys!!\n");
			bt_dumptree(btr);
			if (!checkbtreenode(btr, NULL, kmin, kmax, 0))
				return 0;
			else
				return 1;
		}
		if (!checkbtreenode(btr, NODES(btr, x)[0], kmin,
		    KEYS(btr, x)[0], 0))
			return 0;
		for (i = 1; i < x->n; i++)
			if (!checkbtreenode(btr, NODES(btr, x)[i],
			    KEYS(btr, x)[i - 1], KEYS(btr, x)[i], 0))
				return 0;
		if (!checkbtreenode(btr, NODES(btr, x)[i], KEYS(btr, x)[i - 1],
		    kmax, 0))
			return 0;
	}
	return 1;
}

int
treeheight(struct btree *btr)
{
	struct btreenode *x;
	int ret;

	x = btr->root;
	ret = 0;

	while (!x->leaf) {
		x = NODES(btr, x)[0];
		ret++;
	}

	return ++ret;
}
