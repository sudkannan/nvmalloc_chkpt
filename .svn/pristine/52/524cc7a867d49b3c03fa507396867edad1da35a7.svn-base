#include <btree.h>

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


void printtimevaldiff(struct timeval a, struct timeval b);
int intcmp(void *a, void *b);

int
checkdup(int array[], int n, int e)
{
	int i;

	for (i = 0; i < n; i++)
		if (array[i] == e)
			return 1;
	return 0;
}

void
printtimevaldiff(struct timeval a, struct timeval b)
{
	int dsec;
	int dusec;

	dsec = a.tv_sec - b.tv_sec;
	dusec = a.tv_usec - b.tv_usec;

	if (dusec < 0) {
		dsec--;
		dusec += 1000000;
	}
	printf("%d.%06d", dsec, dusec);
}

int
intcmp(void *a, void *b)
{
	unsigned int x = (unsigned int) a;
	unsigned int y = (unsigned int) b;
	if (x < y)
		return -1;
	if (x == y)
		return 0;
	/*if (x > y)*/
		return 1;
}

void
main(int argc, char **argv)
{
	int i;
	struct btree *bt;

#if 0
	if (argc != 2) {
		printf("usage: test <nodesize>\n");
		exit(1);
	}

	bt = bt_create(intcmp, atoi(argv[1]));
	printf("original tree:\n");
	bt_dumptree(bt);

	for (i = 0; i < 1000000; i++)
		bt_insert(bt, random());
	bt_treestats(bt);
#elif 1
	int *nums;
	int t;
	struct rusage start, end;

	if (argc != 3) {
		printf("usage: test <numnodes> <nodesize>\n");
		exit(1);
	}

	t = atoi(argv[1]);
	nums = malloc(sizeof *nums * t);
	bt = bt_create(intcmp, atoi(argv[2]));
	assert(bt);

	getrusage(RUSAGE_SELF, &start);
	for (i = 0; i < t; i++)
		bt_insert(bt, (void *)(nums[i] = random()));
	getrusage(RUSAGE_SELF, &end);

	printf("insert time: ");
	printtimevaldiff(end.ru_utime, start.ru_utime);
	puts("");

	getrusage(RUSAGE_SELF, &start);
	for (i = 0; i < t; i++)
		bt_find(bt, (void *)nums[random() % t]);
	getrusage(RUSAGE_SELF, &end);

	printf("search time: ");
	printtimevaldiff(end.ru_utime, start.ru_utime);
	puts("");

	getrusage(RUSAGE_SELF, &start);
	for (i = 0; i < t; i++)
		if (!bt_delete(bt, (void *)nums[i]))
			printf("failed to delete %p.\n", nums[i]);
	getrusage(RUSAGE_SELF, &end);

	printf("delete time: ");
	printtimevaldiff(end.ru_utime, start.ru_utime);
	puts("");
#else
	int *nums;
	int tnum;
	int t;

	if (argc != 2) {
		printf("usage: test <numnodes>\n");
		exit(1);
	}

	t = atoi(argv[1]);
	nums = malloc(sizeof *nums * t);
	bt = bt_create(intcmp, 4096);
	bt_dumptree(bt);

	for (i = 0; i < t; i++) {
		if (i == 1)
			puts("gcc suck rocks!");
		do
			nums[i] = random();
		while (checkdup(nums, i, nums[i]));
		bt_insert(bt, nums[i]);
		if (!bt_checktree(bt, 0x0, 0xffffffff)) {
			printf("tree isn't in order after insert!!, at %d:%p\n",
			    i, nums[i]);
			/*bt_dumptree(bt);*/
			exit(1);
		}
	}
	printf("done inserting, now searching...\n");
	/*bt_dumptree(bt);*/
	for (i = 0; i < t; i++)
		if ((tnum = bt_find(bt, nums[i])) != nums[i]) {
			printf("can't find node %d, found %d instead.\n",
			    nums[i], tnum);
			exit(1);
		}
	printf("done searching, now deleting...\n");
	/*bt_dumptree(bt);*/
	for (i = 0; i < t; i++) {
		/*printf("deleting %p\n", nums[i]);*/
		if (bt_delete(bt, nums[i]) != nums[i]) {
			printf("failed to delete %d:%p.\n", i, nums[i]);
			exit(1);
		}
		/*if (!bt_checktree(bt, 0x0, 0xffffffff)) {
			printf("tree isn't in order after del!!, at %d:%p\n", i,
			    nums[i]);
			bt_dumptree(bt);
			exit(1);
		}*/
		/*bt_dumptree(bt);*/
	}
	bt_dumptree(bt);
#endif
	exit(0);
}
