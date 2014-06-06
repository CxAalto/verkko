# This is a Makefile for doing various tests related to verkko, like
# building documentation.  It is in a Makefile for concise scripting,
# but could be moved else where later.  Maybe it would better fit in a
# setup.py file.


.PHONY: default test docs coverage cron

default:
	@echo "Usage:"
	@echo "  make clean - remove all temporary files created by these commands"
	@echo "  make test - run unit tests (making docs/test-results.html if the)"
	@echo "              html-output module is installed)"
	@echo "  make docs - make HTML documentation in build/html/"
	@echo "  make coverage - makes test coverage results in ./docs/coverage/"

clean:
	rm -rf docs/build/
	rm -rf docs/coverage/
	rm -f docs/test-results.html

# For HTML output this must be installed:
#  https://github.com/cboylan/nose-html-output
NOSE_HTML = $(if $(shell nosetests -p | grep html-output), \
                   --with-html-output --html-out-file=docs/test-results.html)

test:
	nosetests . $(NOSE_HTML)

# Generate all docs
docs:
	mkdir -p docs/
	python docs/apidoc.py ../verkko/ -o ./docs/api/ -f -d 0 --separate
#	PYTHONPATH=$PYTHONPATH:. sphinx-build -b html ./docs/ ./docs/build/html/
#	This is needed in order to handle 'import pylab' in scripts.
	python -c 'import matplotlib ; matplotlib.use("Agg"); import sphinx ; sphinx.main(argv="sphinx-build -E -a -b html ./docs/ ./docs/build/html/".split())'


# Make the coverage tests in ./docs/coverage/
coverage:
	nosetests --with-coverage ../verkko/ \
	--cover-erase --cover-package=verkko \
	--cover-html --cover-html-dir=docs/coverage/
#	--cover-inclusive

# Automatic script to fetch, update.  You should "cp Makefile
# Makefile.local" manually in order to use this (for slight
# safety/stability reasons - only a known-good makefile will be copied
# but this does *not* provide security).  The cron command should be:
#  ... cd /path/to/this/ && make -f Makefile.local cron > cron.output 2>&1
cron:
	git fetch
	git checkout origin/master
	make -f Makefile.local clean || true
	make -f Makefile.local test || true
	make -f Makefile.local coverage || true
	make -f Makefile.local docs || true
	git checkout master

