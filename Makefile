TEXBIN = /Library/TeX/texbin
LECTURES = $(sort $(wildcard lectures/*/))

lecture-%: lectures/%/slides.tex
	cd lectures/$* && mkdir -p output && \
	$(TEXBIN)/pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=output slides.tex && \
	$(TEXBIN)/biber --output-directory=output slides && \
	$(TEXBIN)/pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=output slides.tex && \
	$(TEXBIN)/pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=output slides.tex

all: $(patsubst lectures/%/,lecture-%,$(LECTURES))

clean:
	rm -rf lectures/*/output

.PHONY: all clean
