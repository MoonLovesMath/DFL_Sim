#!/bin/bash
for alpha in $(seq 1 0.1 2)
do
    for beta in $(seq 0.1 0.1 1)
    do
        python scale_free.py --size 10000 --m_edge 2 --ratio 0.75 --gen 1000 \
            --alpha $alpha --beta $beta --sigma 2 --T 200 --O 5 --A 10 --K 0.01 \
            --seed-start 0 --seed-end 10
    done
done

# python visualize_heatmap.py --data_dir data/ring --save_dir images --ratio 0.5 --K 0.01

# python visualize_cooperator_ratio.py --data_dir data/fc --save_dir images \
#     --ratio 0.5 --K 0.01 --alpha 1.5 --beta 1.5
