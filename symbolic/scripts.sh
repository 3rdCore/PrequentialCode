for i in {0..14}
do
    python run.py -d mastermind -m gpt-4o -t 1 -n 100 -s $i -r "../experiments/runs"
done


# python analyze.pypy -r "../experiments/results/results_seed=0_1725058991/"
