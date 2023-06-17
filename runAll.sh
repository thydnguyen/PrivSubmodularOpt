iter=" 20"
epsilon=" 0.1"
method="--gumbel --laplace --random --nonprivate --batch --batchpriv"
python taxi_cab_grid.py --K 1e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 2e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 3e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 4e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 5e-2 --epsilon $epsilon --numIter $iter $method
epsilon="1"
python taxi_cab_grid.py --K 1e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 2e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 3e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 4e-2 --epsilon $epsilon --numIter $iter $method
python taxi_cab_grid.py --K 5e-2 --epsilon $epsilon --numIter $iter $method

epsilon="0.1"
python synth_data_grid.py --K 1e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 2e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 3e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 4e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 5e-2 --epsilon $epsilon --numIter $iter $method
epsilon="1"
python synth_data_grid.py --K 1e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 2e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 3e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 4e-2 --epsilon $epsilon --numIter $iter $method
python synth_data_grid.py --K 5e-2 --epsilon $epsilon --numIter $iter $method