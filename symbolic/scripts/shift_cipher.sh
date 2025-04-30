# for i in {0..4}
# do
#     python run.py -d shift_cipher -m llama-2-7b -t 30 -n 300 -w 1 -s $i -r "../experiments/runs"
# done

for i in {0..4}
do
    python run.py -d shift_cipher -m llama-3.1-8b -t 30 -n 300 -w 2 -s $i -r "../experiments/runs"
done
