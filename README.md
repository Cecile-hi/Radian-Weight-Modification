For training experiment, run 
```
  python train.sh --yaml
```
An example would be: 

```
  python train.sh clear10/clear10_feature_resnet50_moco.yaml
```
For parsing metric, run
```
python parse_log_to_result.py --split --verbose[to print out the result matrix as well] --move[move to main server to plot] 
```
An example would be: 

```
python parse_log_to_result.py --split clear100_feature_resnet50_moco --verbose 1 --move 1
```
For plotting the result matrix, like one in our paper, first need to specify --move 1 in running parse_log_to_result.py, and then run
```
python get_metric_all.py --plot 1
```


