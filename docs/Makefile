# Minimal makefile for Sphinx documentation
# Copyright (c) Facebook, Inc. and its affiliates.

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)/en" "$(BUILDDIR)/en" $(SPHINXOPTS) $(O)
	# @$(SPHINXBUILD) -M $@ "$(SOURCEDIR)/zh_cn" "$(BUILDDIR)/zh_cn" $(SPHINXOPTS) $(O)
