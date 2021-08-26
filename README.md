
# WIP : DL based 5G slicing for Wifi
This project uses ns3 simulation code from [this repo](https://github.com/matteonerini/5g-network-slicing-for-wifi-networks) and [ns3-gym](https://github.com/tkn-tub/ns3-gym).

## Steps to run
1. Install all required dependencies required by ns-3.
```
# minimal requirements for C++:
apt-get install gcc g++ python

see https://www.nsnam.org/wiki/Installation
```
2. Install ZMQ and Protocol Buffers libs:
```
# to install protobuf-3.6 on ubuntu 16.04:
sudo add-apt-repository ppa:maarten-fonville/protobuf
sudo apt-get update

apt-get install libzmq5 libzmq5-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler
```
3. Configure and build ns-3 project (if you are going to use Python virtual environment, please execute these commands inside it):
```
#Clone repo
git clone https://github.com/confifu/wifiSlicing.git

# Opengym Protocol Buffer messages (C++ and Python) are build during configure
cd wifiSlicing
./waf configure
./waf build
```

4. Install ns3gym located in src/opengym/model/ns3gym (Python3 required)
```
pip3 install ./src/opengym/model/ns3gym
```

5. Run wifi slicing simulation:
```
cd ./scratch/wifislice
./wifitest.py
```
