# Real-time-AI-for-solid-state-batteries

## **Related Commands**

[Training]:
python Trainer_SingleBat.py

[Resume the previous training]:
python Trainer_SingleBat.py -resume -Re_time xxx

[continue training with new battery]
python Trainer_SingleBat.py -new_continue -continue_modelPath xxx

------

Please note that full real-time closed-loop operation requires access to actual battery testing hardware.



## **Virtual Instrument interface**

This repository provides a **virtual instrument interface** to simulate communication between the AI workstation and a battery tester.

To launch the simulated environment, run:

`python Exp_Pyctrl_solartron/Exp_ctrl.py`

------

This interface enables users to explore the basic communication and control logic of our platform **without requiring actual hardware**.



