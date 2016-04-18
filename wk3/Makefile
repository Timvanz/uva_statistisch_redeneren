# TODO: REPLACE STUDENT NAMES WITH YOURS AND WEEK WITH THE CORRECT NUMBER
ASSIGNMENT_NUMBER := 3
STUDENT_NAMES := maico_timmerman_tim_van_zalingen

DOCS := $(shell ls docs/*.pdf)
IMAGES := $(shell ls *.png 2> /dev/null) $(shell ls *.jpg 2> /dev/null)
DELIVERABLE := $(STUDENT_NAMES)_assignment_$(ASSIGNMENT_NUMBER).tar.gz
# TODO: CHANGE THIS IF YOU DO NOT WANT TO SUBMIT ALL PYTHON FILES
PYTHON_FILES := $(shell ls *.py)

all: $(DELIVERABLE)

check: $(PYTHON_FILES)
	pep8 $^
	pyflakes $^
	touch $(@)

clean:
	rm -f $(DELIVERABLE) check

$(DELIVERABLE): $(PYTHON_FILES) $(DOCS) $(IMAGES) check
	mkdir $(STUDENT_NAMES)_assignment_$(ASSIGNMENT_NUMBER)
	cp $(PYTHON_FILES) $(DOCS) $(IMAGES) $(STUDENT_NAMES)_assignment_$(ASSIGNMENT_NUMBER)
	tar -czf $@ $(STUDENT_NAMES)_assignment_$(ASSIGNMENT_NUMBER)
	rm -r $(STUDENT_NAMES)_assignment_$(ASSIGNMENT_NUMBER)
