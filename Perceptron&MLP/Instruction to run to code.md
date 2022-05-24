# Instruction to run to code

## Part 1

You can simply open the `train.ipynb` and run the code block by block.



## Part 2

You can run the following command:

```bash
python train_mlp_numpy.py --dnn_hidden_units 20 --learning_rate 1e-1 --max_steps 1500 --eval_freq 10 --mode BGD
```

To visualize the result, you can open `visualize.ipynb` and run the code block by block.



## Part 3

You can run the following command:

```bash
python train_mlp_numpy.py --dnn_hidden_units 20 --learning_rate 1e-2 --max_steps 1500 --eval_freq 1 --mode SGD
```

To visualize the result, you can open `visualize.ipynb` and run the code block by block.