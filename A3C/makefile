dir=1
run=604800
model=''
path="/Users/wyatt/Documents/Github_Repositories/RoutingProject/A3C/"

test :
	python3 GA3C.py PATH=$(path) DIRECTION=$(dir) AGENTS=8 NUM_OF_CUSTOMERS=5 MODEL_TO_RESTORE=$(model) TRAIN=False
res :
	python3 GA3C.py AGENTS=8 NUM_OF_CUSTOMERS=15 MODEL_TO_RESTORE=$(model)

cpu1 :
	python3 GA3C.py DIRECTION=1 TRAIN=True RUN_TIME=$(run) AGENTS=8
cpu2 :
	python3 GA3C.py DIRECTION=2 TRAIN=True RUN_TIME=$(run) AGENTS=8
cpu3 :
	python3 GA3C.py DIRECTION=3 TRAIN=True RUN_TIME=$(run) AGENTS=8
# cpu4 :
# 	python3 GA3C.py DIRECTION=4 TRAIN=True RUN_TIME=$(run) AGENTS=8
cpu5 :
	python3 GA3C.py DIRECTION=5 TRAIN=True RUN_TIME=$(run) AGENTS=8
cpu6 :
	python3 GA3C.py DIRECTION=6 TRAIN=True RUN_TIME=$(run) AGENTS=8
cpu7 :
	python3 GA3C.py DIRECTION=7 TRAIN=True RUN_TIME=$(run) AGENTS=8

gpu1 :
	python3 GA3C.py DIRECTION=1 TRAIN=True GPU=True RUN_TIME=$(run) AGENTS=12
gpu2 :
	python3 GA3C.py DIRECTION=2 TRAIN=True GPU=True RUN_TIME=$(run) AGENTS=12
gpu3 :
	python3 GA3C.py DIRECTION=3 TRAIN=True GPU=True RUN_TIME=$(run) AGENTS=12
gpu4 :
	python3 GA3C.py DIRECTION=4 TRAIN=True GPU=True RUN_TIME=$(run) AGENTS=12
gpu5 :
	python3 GA3C.py DIRECTION=5 TRAIN=True GPU=True RUN_TIME=$(run) AGENTS=12
gpu6 :
	python3 GA3C.py DIRECTION=6 TRAIN=True GPU=True RUN_TIME=$(run) AGENTS=12

clean :
	rm -rf ./logs/*
	rm -rf ./checkpoint/*
