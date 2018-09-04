VPATH = Presentation:Paper
mod=1
MODEL_NAME='test'
path="/Users/wyatt/Documents/Github_Repositories/RoutingProject/A3C/"

short_test :
	rm -rf ./logs/*
	rm -rf ./checkpoint/*
	rm -rf ./ini_files/*
	rm -rf ./figs/*
	python3 GA3C.py PATH=$(path) MODEL_SETTING=$(mod) MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
test :
	python3 GA3C.py PATH=$(path) MODEL_SETTING=1 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=2 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=3 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=4 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=5 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=6 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=7 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=8 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=9 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=10 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
	python3 GA3C.py PATH=$(path) MODEL_SETTING=11 MODEL_NAME=$(MODEL_NAME) TRAINING_MIN_BATCH_SIZE=10
res :
	python3 GA3C.py MODEL_NAME=$(MODEL_NAME) RESTORE=1
tsp :
	mkdir -p figs
	mkdir -p ini_files
	mkdir -p logs
	mkdir -p checkpoint
	python3 GA3C.py MODEL_SETTING=$(MODEL_SETTING) MODEL_NAME=$(MODEL_NAME)

pdf : RoutingProjectPaper.tex
	cd Paper;\
	ls;\
	pdflatex RoutingProjectPaper;\
	bibtex RoutingProjectPaper;\
	pdflatex RoutingProjectPaper;\
	pdflatex RoutingProjectPaper;\
	cd ../;

pdfclean :
	cd latex;\
	rm *.bbl;\
	rm *.blg;\
	rm *.log;\
	rm *.out;\
	rm *.aux;\
	cd ../;

view : RoutingProjectPaper.pdf
	cd Paper;\
	open RoutingProjectPaper.pdf;\
	cd ../;

clean :
	# rm -rf ./logs/*
	# rm -rf ./checkpoint/*
	# rm -rf ./ini_files/*
	find . -name "./figs/*" -print0 | xargs -0 rm -f
	# rm -rf ./figs/*
