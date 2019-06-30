# AirSim Unity Reinforcement Learning Quadrotor Pytorch
Reinforcement learning for AirSim Unity Quadrotor environment and DQN pytorch  



[![Youtube](https://img.youtube.com/vi/iFZccZm04hQ/0.jpg)](https://youtu.be/iFZccZm04hQ)





<h2>Environment</h2>

- [Microsoft Airsim on Unity](<https://github.com/microsoft/AirSim/tree/master/Unity>)
- [Unity-2018.2.7f1 (Linux version)](<https://github.com/microsoft/AirSim/tree/master/Unity#download-and-install-unity-for-linux>)
- Ubuntu 18.04
- CPU Intel Core i7-8750H





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
  ```







<h2>Author</h2>

- Subin Yang
- subinlab.yang@gmail.com