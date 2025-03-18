1. Data preparation

2. Training for state prediction model:
python Cls_train_State.py -net resnet50 -gpu
python Cls_train_State.py -net resnet50 -gpu -resume

3. testing and generating .txt file.
python Cls_test_Stage.py -net resnet50 -gpu -weights xxx.pth

4. Performance evaluation
python Cls_plot_performance.py

