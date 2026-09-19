#!/bin/bash

for alpha in $(seq 1 0.1 2)
do
    for beta in $(seq 0.1 0.1 1)
    do
        python fully_connected.py --size 10000 --ratio 0.5 --gen 1000 \
            --alpha $alpha --beta $beta --sigma 0.5 --T 200 --O 5 --A 10 --K 0.1 \
            --det False --seed-start 0 --seed-end 10
    done
done

# for alpha in $(seq 1 0.1 2)
# do
#     for beta in $(seq 0.1 0.1 1)
#     do
#         python lattice.py --size_x 100 --size_y 100 --ratio 0.5 --gen 1000 \
#             --alpha $alpha --beta $beta --sigma 0.5 --T 200 --O 5 --A 10 --K 0.1 \
#             --det False --seed-start 0 --seed-end 10
#     done
# done

# for alpha in $(seq 1 0.1 2)
# do
#     for beta in $(seq 0.1 0.1 1)
#     do
#         python ring.py --size 10000 --ratio 0.5 --gen 1000 \
#             --alpha $alpha --beta $beta --sigma 0.5 --T 200 --O 5 --A 10 --K 0.1 \
#             --det False --seed-start 0 --seed-end 10
#     done
# done

# for alpha in $(seq 1 0.1 2)
# do
#     for beta in $(seq 0.1 0.1 1)
#     do
#         python random_network.py --size 10000 --N 20 --ratio 0.5 --gen 1000 \
#             --alpha $alpha --beta $beta --sigma 0.5 --T 200 --O 5 --A 10 --K 0.1 \
#             --det False --seed-start 0 --seed-end 10
#     done
# done