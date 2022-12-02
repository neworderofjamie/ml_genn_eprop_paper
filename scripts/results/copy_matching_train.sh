#/bin/bash

for TEST in test_output_0_512_100_dvs_gesture_1234_*.csv;
do
    cp ../../classifier_model/train_output_99${TEST#test_output_0} .;
done
