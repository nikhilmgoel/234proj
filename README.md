# CS 234 Project: Navigating Pedestrian Spaces Using Public Intersection Data
#### Cameron Ramos [calramos@stanford.edu] & Nikhil Goel [ngmoel@stanford.edu]

---

#### Getting Started
```bash
# download and check out the latest code (currently unmerged in cam_dev)
$ git clone https://github.com/nikhilmgoel/234proj.git
$ cd 234proj/
$ git checkout cam_dev
$ cd gym-navigate
$ pip install -e .
$ cd ..
# generate episodes (be careful this uses about 12 gigs of data)
$ python read_data.py
# train and evaluate the agent
$ python train_circle_of_death_nature.py
# spin up a monitor in a seperate connection
$ tensoboard --logdir=results
```
