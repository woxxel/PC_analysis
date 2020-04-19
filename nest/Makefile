#
# Makefile for John Skilling's Nest sampling program of Thu 9/9/04
#                            			David MacKay 
#

tar: ALWAYS
	tar zcvfh nest.tar.gz  \
 random.c \
 random.h  \
 userstr.h \
 sphere.c nest.c nest.h  \
 Makefile
	zip nest.zip  \
 random.c \
 random.h  \
 userstr.h \
 sphere.c nest.c nest.h  \
 Makefile

ALWAYS:

CC = gcc

all: sphere

clean: ALWAYS
	rm *.o

sphere:     sphere.o random.o nest.o
	$(CC) sphere.o random.o nest.o -lm -o sphere 


%.dvi:	%.tex
		tex $*
		if grep Rerun $*.log > /dev/null; then tex $*; fi
		- grep Warning $*.log
%.ps:	%.dvi
		dvips $<  -G0  -z -Pcmz -t a4  -o $*.ps
%.pdf:	%.ps
		ps2pdf13 -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode -dPDFsettings=/prepress -sPAPERSIZE=a4   $< $*.pdf

