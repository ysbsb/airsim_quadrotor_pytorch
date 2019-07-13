# AirSim Unity Reinforcement Learning Quadrotor Pytorch
Reinforcement learning for AirSim Unity Quadrotor environment and DQN pytorch  



- [Youtube](https://youtu.be/iFZccZm04hQ)



![dqn_quadrotor](https://user-images.githubusercontent.com/37301677/60393570-a33fc600-9b52-11e9-828d-27c35303b021.gif)





<h2>Environment (Setup)</h2>

- [Microsoft Airsim on Unity](<https://github.com/microsoft/AirSim/tree/master/Unity>)
- [Unity-2018.2.7f1 (Linux version)](<https://github.com/microsoft/AirSim/tree/master/Unity#download-and-install-unity-for-linux>)
- Ubuntu 18.04
- CPU Intel Core i7-8750H





<h2>Test</h2>

Simulation test of `agent_v0.py` is completed, but not in `agent.py`.   

`agent_v0.py` is simple version of dqn script to test work well in Airsim Simulator.





<h2>To use</h2>

Put 3 files `env.py `, `agent.py` , `run.py`   in directory  `~/Airsim/PythonClient/multirotor`.  





<h2>To run</h2>

- Launch Unity Editor.

  ```
  cd ~/Unity-2018.2.7f1/Editor
  ./Unity
  ```

- Choose your own environment.

- Default example `UnityDemo` is given in directory `~/Airsim/Unity` .

- Select Play button.

- If simulation is playing, select Drone Mode.

- Now run script.

  ```
  cd ~/Airsim/PythonClient/multirotor
  python run.py
  
  # or run
  python agent_v0.py
  ```





<h2>Author</h2>

- Subin Yang
- subinlab.yang@gmail.com
