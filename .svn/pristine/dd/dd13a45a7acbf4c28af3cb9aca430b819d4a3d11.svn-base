# Generated automatically from Makefile.in by configure.
#	$Id: Makefile.in,v 1.3.2.3 2001/03/28 06:43:10 jmg Exp $
#

SHELL = /bin/sh

CXX=g++
CC=gcc
GNUFLAG :=  -std=gnu99

INCLUDE=$(NVMALLOC_HOME)
DEFINES := -DMALLOC_PRODUCTION -lstdc++
CFLAGS := -g -I$(INCLUDE) -Wall -pipe -fvisibility=hidden \
      -funroll-loops  -Wno-implicit -Wno-uninitialized \
      -Wno-unused-function -Wno-unused-variable -lm -lssl  -fPIC
LIBS= -L$(NVMALLOC_HOME) -lnvmchkpt -lpthread 




LIB = btree

SRCS = bt_code.c bt_output.c
OBJS = bt_code.o bt_output.o
SOBJS = bt_code.so bt_output.so
POBJS = bt_code.po bt_output.po

INCS = btree.h

TESTPROG = test
TESTSRCS = main.c
TESTOBJS = main.o

MAN3 = btree.3

DEBUG = -g
#OPT = -O2 # -DNO_INLINE
PROFILE = -pg
LIBOPTS = -DSTATS #-DSAFE_BTREE

srcdir = .

AFLAGS = -Wall -I$(srcdir) -I$(INCLUDE) $(DEBUG) $(OPT) $(LIBOPTS) $(CFLAGS) $(LIBS)

MAJOR = 1
ARNAME = lib$(LIB).a
SONAME = lib$(LIB).so.$(MAJOR)
PANAME = lib$(LIB)_p.a

prefix = /usr/local
exec_prefix = ${prefix}
LIBDIR = ${exec_prefix}/lib
LIBOWN = 0
LIBGRP = 0
LIBMOD = 0444
INCDIR = ${prefix}/include
INCOWN = 0
INCGRP = 0
INCMOD = 0444
MANDIR = ${prefix}/man
MANOWN = 0
MANGRP = 0
MANMOD = 0444

all: $(ARNAME) $(SONAME) $(PANAME)

$(ARNAME): $(OBJS)
	$(AR) rc $@ $(OBJS)

$(SONAME): $(SOBJS)
	$(CC) -shared -o $@ $(SOBJS)

$(PANAME): $(POBJS)
	$(AR) rc $@ $(POBJS)

$(TESTPROG): $(TESTOBJS) $(PANAME)
	$(CXX) -static $(CFLAGS) -o $@ $(TESTOBJS) $(PANAME) $(PROFILE) $(LIBS)

deepclean: clean
	rm -f Makefile config.cache config.log config.status configure

clean:
	rm -f $(ARNAME) $(OBJS) $(SONAME) $(SOBJS) $(PANAME) $(POBJS) \
	    $(TESTPROG) $(TESTOBJS) .depend

install: $(ARNAME) $(SONAME) $(PANAME)
	mkdir -p -m 755 $(LIBDIR)
	for i in $(ARNAME) $(SONAME) $(PANAME); do		\
		cp $$i $(LIBDIR);				\
		chown $(LIBOWN):$(LIBGRP) $(LIBDIR)/$$i;	\
		chmod $(LIBMOD) $(LIBDIR)/$$i;			\
	done
	mkdir -p -m 755 $(INCDIR)
	for i in $(INCS); do					\
		cp $$i $(INCDIR);				\
		chown $(INCOWN):$(INCGRP) $(INCDIR)/$$i;	\
		chmod $(INCMOD) $(INCDIR)/$$i;			\
	done
	mkdir -p -m 755 $(MANDIR)/man3
	for i in $(MAN3); do					\
		cp $$i $(MANDIR)/man3;				\
		chown $(MANOWN):$(MANGRP) $(MANDIR)/man3/$$i;	\
		chmod $(MANMOD) $(MANDIR)/man3/$$i;		\
	done

depend:
	mkdep $(AFLAGS) $(SRCS)

$(srcdir)/configure: configure.in
	cd $(srcdir) && autoconf

Makefile: Makefile.in config.status
	./config.status

config.status: configure
	./config.status --recheck

.SUFFIXES:
.SUFFIXES: .so .po .c .o
.c.so:
	$(CC) -fpic $(AFLAGS) -c $< -o $@

.c.po:
	$(CC) $(PROFILE) $(AFLAGS) -c $< -o $@

.c.o:
	$(CC) $(CPPFLAGS) $(AFLAGS) -c $< -o $@

.c: $(ARNAME)
	$(CC) $(CPPFLAGS) $(AFLAGS) -o $@ $< $(ARNAME)
