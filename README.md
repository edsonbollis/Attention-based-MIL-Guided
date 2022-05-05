# Attention-based-MIL-Guided

This project contains the source code described in 'Weakly supervised attention-based models using activation maps for citrus mite and insect pest classification'. This [work](https://doi.org/10.1016/j.compag.2022.106839) was published in [Computers and Electronics in Agriculture](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture) @ 2022.

This code is an example of how to use Two-WAM to produce instances for a multiple instance learning method. It uses the [Citrus Pest Benchmark](https://github.com/edsonbollis/Citrus-Pest-Benchmark) in the training process. The weakly supervised multi-instance learning code presents a way to classify tiny regions of interest (ROIs) through a convolutional neural network, a selection strategy based on saliency maps, a weighted evaluation method, and attention-based models.

![Mite Images](https://github.com/edsonbollis/Weakly-Supervised-Learning-Citrus-Pest-Benchmark/blob/master/pipeline.png)

Our method consists of four steps. In Step 1, we train an attention-based CNN (initially trained on the ImageNet) on the Citrus
Pest Benchmark. In Step 2, we automatically generate multiple patches regarding saliency maps. In Step 3, we train other
attention-based CNN (for the target task) according to a multiple instance learning approach. In Step 4, we apply a weighted
evaluation scheme to predict the image class.

### Tutorial

Use the code `train_attention_based_mil_guided_bag_model.py` to train the Bag Model: python train_attention_based_mil_guided_bag_model.py <dataset_folder>

Cut the instances with: python instance-database-generator.py <dataset_folder> <new_dataset_instance_folder> <weights_folder> 

Use the code `train_attention_based_mil_guided_instance_model.py` to train the Instance Model: python train_attention_based_mil_guided_instance_model.py <dataset_folder> <dataset_instance_folder>

The code `evaluate_attention_based_mil_guided_bag_model.py` evaluates the Bag Models and `evaluate_attention_based_mil_guided_instance_model.py` evaluates the Instance Models.


### Citation
```
@article{bollis2022weakly,
  title={Weakly supervised attention-based models using activation maps for citrus mite and insect pest classification},
  author={Bollis, Edson and Maia, Helena and Pedrini, Helio and Avila, Sandra},
  journal={Computers and Electronics in Agriculture},
  volume={195},
  pages={106839},
  year={2022},
  publisher={Elsevier}
}
```

### Acknowledgments

ATTN: This code is free for academic usage. For other purposes, please contact Edson Bollis (edsonbollis@gmail.com).
