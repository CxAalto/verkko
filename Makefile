
.PHONY: docs coverage test nightly


clean:
	rm -r docs/build/
	rm -r docs/coverage/
	rm docs/test-results.html

# For HTML output this must be installed:
#  https://github.com/cboylan/nose-html-output
NOSE_HTML = $(if $(shell nosetests -p | grep html-output), \
                   --with-html-output --html-out-file=docs/test-results.html)

test:
	nosetests . $(NOSE_HTML)

docs:
	mkdir -p docs/
	python docs/apidoc.py . -o ./docs/api/ -f -d 0 --separate
#	cp docs/conf.py docs/build/
	PYTHONPATH=$PYTHONPATH:. sphinx-build -b html ./docs/ ./docs/build/html/

# Make a list of all top-level directories, _without_ ./ prefix.
MODULES=$(shell find . -maxdepth 1 -type d | sed -E 's@./(.*)@\1@')

coverage:
	nosetests --with-coverage . \
	--cover-erase  \
	$(foreach x, $(MODULES), --cover-package=$x) \
	--cover-html --cover-html-dir=docs/coverage/
#	--cover-inclusive


cron:
	git fetch
	git checkout origin/master
	make -f Makefile.local clean || true
	make -f Makefile.local test || true
	make -f Makefile.local coverage || true
	make -f Makefile.local docs || true
	git checkout master

