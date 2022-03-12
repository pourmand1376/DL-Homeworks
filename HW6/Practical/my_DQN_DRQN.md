# CE-40719: Deep Learning
## HW6 - Deep Reinforcement Learning

(40 points)

#### Name: 
#### Student No.: 


In this assignment, we are going to design an agent to play Atari game. 
we will use a wrapper to change MDP problem to POMDP, as a result, we are able to investigate the efficiency of using memory to solve problems in a partial observability setting.
For this reseaon, we use a Deep Q-Network model as memory less architecture and a [DRQN](https://arxiv.org/abs/1507.06527) as a memoryful agent to play game Pong(`PongNoFrameskip-v4` environment of [gym](https://gym.openai.com/) library).
In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3). And action space is 6.
We will train the model for 200,000 steps and should take approximately 2 hour.

At the end, you should be able to conclude effectiveness of recurrent memory to cancel out noisy observation.


![Pong](https://cdn-images-1.medium.com/max/800/1*UHYJE7lF8IDZS_U5SsAFUQ.gif)

 ### 1. Setup

if you use google colab to train your network, mount to google drive would be necessary 


```python
from google.colab import drive
drive._mount('/content/drive')
```

First, we need to install `stable-baselines`. This library  is a set of improved implementations of Reinforcement Learning (RL) algorithms based on OpenAI Baselines. We will use some of **wrappers** of this library. Wrappers will allow us to add functionality to environments, such as modifying observations and rewards to be fed to our agent. It is common in reinforcement learning to preprocess observations in order to make them more easy to learn from. 

- For linux based Operating Systems or google colab run cell below:


```python
%%shell

sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev

pip install stable-baselines[mpi]==2.8.0
```

    Get:1 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]
    Get:2 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease [15.9 kB]
    Get:3 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease [3,626 B]
    Hit:4 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Get:5 http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic InRelease [15.9 kB]
    Hit:6 http://archive.ubuntu.com/ubuntu bionic InRelease
    Get:7 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]
    Get:8 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease [21.3 kB]
    Hit:9 http://archive.ubuntu.com/ubuntu bionic-backports InRelease
    Ign:10 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Get:11 http://security.ubuntu.com/ubuntu bionic-security/multiverse amd64 Packages [26.8 kB]
    Get:12 http://security.ubuntu.com/ubuntu bionic-security/restricted amd64 Packages [716 kB]
    Get:13 http://security.ubuntu.com/ubuntu bionic-security/main amd64 Packages [2,489 kB]
    Get:14 http://security.ubuntu.com/ubuntu bionic-security/universe amd64 Packages [1,459 kB]
    Ign:15 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:16 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
    Hit:17 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Get:18 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main Sources [1,822 kB]
    Get:19 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main amd64 Packages [934 kB]
    Get:20 http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic/main amd64 Packages [45.3 kB]
    Get:21 http://archive.ubuntu.com/ubuntu bionic-updates/multiverse amd64 Packages [34.5 kB]
    Get:22 http://archive.ubuntu.com/ubuntu bionic-updates/restricted amd64 Packages [749 kB]
    Get:23 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 Packages [2,238 kB]
    Get:24 http://archive.ubuntu.com/ubuntu bionic-updates/main amd64 Packages [2,929 kB]
    Get:25 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic/main amd64 Packages [44.7 kB]
    Fetched 13.7 MB in 2s (8,936 kB/s)
    Reading package lists... Done
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    zlib1g-dev is already the newest version (1:1.2.11.dfsg-0ubuntu2).
    zlib1g-dev set to manually installed.
    libopenmpi-dev is already the newest version (2.1.1-8).
    cmake is already the newest version (3.10.2-1ubuntu2.18.04.2).
    0 upgraded, 0 newly installed, 0 to remove and 61 not upgraded.
    Collecting stable-baselines[mpi]==2.8.0
      Downloading stable_baselines-2.8.0-py3-none-any.whl (222 kB)
    [K     |████████████████████████████████| 222 kB 12.3 MB/s 
    [?25hRequirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (3.2.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (1.4.1)
    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (1.1.0)
    Requirement already satisfied: opencv-python in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (4.1.2.30)
    Requirement already satisfied: gym[atari,classic_control]>=0.10.9 in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (0.17.3)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (1.19.5)
    Requirement already satisfied: cloudpickle>=0.5.5 in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (1.3.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.8.0) (1.1.5)
    Collecting mpi4py
      Downloading mpi4py-3.1.3.tar.gz (2.5 MB)
    [K     |████████████████████████████████| 2.5 MB 53.5 MB/s 
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
        Preparing wheel metadata ... [?25l[?25hdone
    Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.8.0) (1.5.0)
    Requirement already satisfied: atari-py~=0.2.0 in /usr/local/lib/python3.7/dist-packages (from gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.8.0) (0.2.9)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.8.0) (7.1.2)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from atari-py~=0.2.0->gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.8.0) (1.15.0)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from pyglet<=1.5.0,>=1.4.0->gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.8.0) (0.16.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.8.0) (3.0.6)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.8.0) (2.8.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.8.0) (0.11.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.8.0) (1.3.2)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->stable-baselines[mpi]==2.8.0) (2018.9)
    Building wheels for collected packages: mpi4py
      Building wheel for mpi4py (PEP 517) ... [?25l[?25hdone
      Created wheel for mpi4py: filename=mpi4py-3.1.3-cp37-cp37m-linux_x86_64.whl size=2185303 sha256=2ddd08c46b5a1eca4816159e7b398edc426ae59fe8e8098477de2e87042d4916
      Stored in directory: /root/.cache/pip/wheels/7a/07/14/6a0c63fa2c6e473c6edc40985b7d89f05c61ff25ee7f0ad9ac
    Successfully built mpi4py
    Installing collected packages: stable-baselines, mpi4py
    Successfully installed mpi4py-3.1.3 stable-baselines-2.8.0





    



- For Windows: 
    - First install [MPI for Windows](https://www.microsoft.com/en-us/download/details.aspx?id=57467) (you need to download and install `msmpisetup.exe`)
    - Then run this command in Prompt: `pip install stable-baselines[mpi]==2.8.0`

install ROMs which needed for creating atari env


```python
import urllib.request
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')
!pip install unrar
!unrar x Roms.rar
!mkdir rars
!mv HC\ ROMS.zip   rars
!mv ROMS.zip  rars
!python -m atari_py.import_roms rars
```

    Collecting unrar
      Downloading unrar-0.4-py3-none-any.whl (25 kB)
    Installing collected packages: unrar
    Successfully installed unrar-0.4
    
    UNRAR 5.50 freeware      Copyright (c) 1993-2017 Alexander Roshal
    
    
    Extracting from Roms.rar
    
    Extracting  HC ROMS.zip                                                   36%  OK 
    Extracting  ROMS.zip                                                      74% 99%  OK 
    All OK
    copying adventure.bin from ROMS/Adventure (1980) (Atari, Warren Robinett) (CX2613, CX2613P) (PAL).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/adventure.bin
    copying air_raid.bin from ROMS/Air Raid (Men-A-Vision) (PAL) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/air_raid.bin
    copying alien.bin from ROMS/Alien (1982) (20th Century Fox Video Games, Douglas 'Dallas North' Neubauer) (11006) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/alien.bin
    copying amidar.bin from ROMS/Amidar (1982) (Parker Brothers, Ed Temple) (PB5310) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/amidar.bin
    copying assault.bin from ROMS/Assault (AKA Sky Alien) (1983) (Bomb - Onbase) (CA281).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/assault.bin
    copying asterix.bin from ROMS/Asterix (AKA Taz) (07-27-1983) (Atari, Jerome Domurat, Steve Woita) (CX2696) (Prototype).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/asterix.bin
    copying asteroids.bin from ROMS/Asteroids (1981) (Atari, Brad Stewart - Sears) (CX2649 - 49-75163) [no copyright] ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/asteroids.bin
    copying atlantis.bin from ROMS/Atlantis (Lost City of Atlantis) (1982) (Imagic, Dennis Koble) (720103-1A, 720103-1B, IA3203, IX-010-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/atlantis.bin
    copying bank_heist.bin from ROMS/Bank Heist (Bonnie & Clyde, Cops 'n' Robbers, Hold-Up, Roaring 20's) (1983) (20th Century Fox Video Games, Bill Aspromonte) (11012) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/bank_heist.bin
    copying battle_zone.bin from ROMS/Battlezone (1983) (Atari - GCC, Mike Feinstein, Brad Rice) (CX2681) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/battle_zone.bin
    copying beam_rider.bin from ROMS/Beamrider (1984) (Activision - Cheshire Engineering, David Rolfe, Larry Zwick) (AZ-037-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/beam_rider.bin
    copying berzerk.bin from ROMS/Berzerk (1982) (Atari, Dan Hitchens - Sears) (CX2650 - 49-75168) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/berzerk.bin
    copying bowling.bin from ROMS/Bowling (1979) (Atari, Larry Kaplan - Sears) (CX2628 - 6-99842, 49-75117) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/bowling.bin
    copying boxing.bin from ROMS/Boxing - La Boxe (1980) (Activision, Bob Whitehead) (AG-002, CAG-002, AG-002-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/boxing.bin
    copying breakout.bin from ROMS/Breakout - Breakaway IV (Paddle) (1978) (Atari, Brad Stewart - Sears) (CX2622 - 6-99813, 49-75107) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/breakout.bin
    copying carnival.bin from ROMS/Carnival (1982) (Coleco - Woodside Design Associates, Steve 'Jessica Stevens' Kitchen) (2468) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/carnival.bin
    copying centipede.bin from ROMS/Centipede (1983) (Atari - GCC) (CX2676) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/centipede.bin
    copying chopper_command.bin from ROMS/Chopper Command (1982) (Activision, Bob Whitehead) (AX-015, AX-015-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/chopper_command.bin
    copying crazy_climber.bin from ROMS/Crazy Climber (1983) (Atari - Roklan, Joe Gaucher, Alex Leavens) (CX2683) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/crazy_climber.bin
    copying defender.bin from ROMS/Defender (1982) (Atari, Robert C. Polaro, Alan J. Murphy - Sears) (CX2609 - 49-75186) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/defender.bin
    copying demon_attack.bin from ROMS/Demon Attack (Death from Above) (1982) (Imagic, Rob Fulop) (720000-200, 720101-1B, 720101-1C, IA3200, IA3200C, IX-006-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/demon_attack.bin
    copying donkey_kong.bin from ROMS/Donkey Kong (1982) (Coleco - Woodside Design Associates - Imaginative Systems Software, Garry Kitchen) (2451) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/donkey_kong.bin
    copying double_dunk.bin from ROMS/Double Dunk (Super Basketball) (1989) (Atari, Matthew L. Hubbard) (CX26159) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/double_dunk.bin
    copying elevator_action.bin from ROMS/Elevator Action (1983) (Atari, Dan Hitchens) (CX26126) (Prototype) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/elevator_action.bin
    copying enduro.bin from ROMS/Enduro (1983) (Activision, Larry Miller) (AX-026, AX-026-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/enduro.bin
    copying fishing_derby.bin from ROMS/Fishing Derby (1980) (Activision, David Crane) (AG-004) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/fishing_derby.bin
    copying freeway.bin from ROMS/Freeway (1981) (Activision, David Crane) (AG-009, AG-009-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/freeway.bin
    copying frogger.bin from ROMS/Frogger (1982) (Parker Brothers, Ed English, David Lamkins) (PB5300) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/frogger.bin
    copying frostbite.bin from ROMS/Frostbite (1983) (Activision, Steve Cartwright) (AX-031) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/frostbite.bin
    copying galaxian.bin from ROMS/Galaxian (1983) (Atari - GCC, Mark Ackerman, Tom Calderwood, Glenn Parker) (CX2684) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/galaxian.bin
    copying gopher.bin from ROMS/Gopher (Gopher Attack) (1982) (U.S. Games Corporation - JWDA, Sylvia Day, Todd Marshall, Robin McDaniel, Henry Will IV) (VC2001) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/gopher.bin
    copying gravitar.bin from ROMS/Gravitar (1983) (Atari, Dan Hitchens, Mimi Nyden) (CX2685) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/gravitar.bin
    copying hero.bin from ROMS/H.E.R.O. (1984) (Activision, John Van Ryzin) (AZ-036-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/hero.bin
    copying ice_hockey.bin from ROMS/Ice Hockey - Le Hockey Sur Glace (1981) (Activision, Alan Miller) (AX-012, CAX-012, AX-012-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/ice_hockey.bin
    copying jamesbond.bin from ROMS/James Bond 007 (James Bond Agent 007) (1984) (Parker Brothers - On-Time Software, Joe Gaucher, Louis Marbel) (PB5110) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/jamesbond.bin
    copying journey_escape.bin from ROMS/Journey Escape (1983) (Data Age, J. Ray Dettling) (112-006) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/journey_escape.bin
    copying kaboom.bin from ROMS/Kaboom! (Paddle) (1981) (Activision, Larry Kaplan, David Crane) (AG-010, AG-010-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/kaboom.bin
    copying kangaroo.bin from ROMS/Kangaroo (1983) (Atari - GCC, Kevin Osborn) (CX2689) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/kangaroo.bin
    copying keystone_kapers.bin from ROMS/Keystone Kapers - Raueber und Gendarm (1983) (Activision, Garry Kitchen - Ariola) (EAX-025, EAX-025-04I - 711 025-725) (PAL).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/keystone_kapers.bin
    copying king_kong.bin from ROMS/King Kong (1982) (Tigervision - Software Electronics Corporation, Karl T. Olinger - Teldec) (7-001 - 3.60001 VE) (PAL).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/king_kong.bin
    copying koolaid.bin from ROMS/Kool-Aid Man (Kool Aid Pitcher Man) (1983) (M Network, Stephen Tatsumi, Jane Terjung - Kool Aid) (MT4648) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/koolaid.bin
    copying krull.bin from ROMS/Krull (1983) (Atari, Jerome Domurat, Dave Staugas) (CX2682) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/krull.bin
    copying kung_fu_master.bin from ROMS/Kung-Fu Master (1987) (Activision - Imagineering, Dan Kitchen, Garry Kitchen) (AG-039-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/kung_fu_master.bin
    copying laser_gates.bin from ROMS/Laser Gates (AKA Innerspace) (1983) (Imagic, Dan Oliver) (720118-2A, 13208, EIX-007-04I) (PAL).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/laser_gates.bin
    copying lost_luggage.bin from ROMS/Lost Luggage (Airport Mayhem) (1982) (Apollo - Games by Apollo, Larry Minor, Ernie Runyon, Ed Salvo) (AP-2004) [no opening scene] ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/lost_luggage.bin
    copying montezuma_revenge.bin from ROMS/Montezuma's Revenge - Featuring Panama Joe (1984) (Parker Brothers - JWDA, Henry Will IV) (PB5760) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/montezuma_revenge.bin
    copying mr_do.bin from ROMS/Mr. Do! (1983) (CBS Electronics, Ed English) (4L4478) (PAL).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/mr_do.bin
    copying ms_pacman.bin from ROMS/Ms. Pac-Man (1983) (Atari - GCC, Mark Ackerman, Glenn Parker) (CX2675) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/ms_pacman.bin
    copying name_this_game.bin from ROMS/Name This Game (Guardians of Treasure) (1983) (U.S. Games Corporation - JWDA, Roger Booth, Sylvia Day, Ron Dubren, Todd Marshall, Robin McDaniel, Wes Trager, Henry Will IV) (VC1007) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/name_this_game.bin
    copying pacman.bin from ROMS/Pac-Man (1982) (Atari, Tod Frye) (CX2646) (PAL).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/pacman.bin
    copying phoenix.bin from ROMS/Phoenix (1983) (Atari - GCC, Mike Feinstein, John Mracek) (CX2673) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/phoenix.bin
    copying video_pinball.bin from ROMS/Pinball (AKA Video Pinball) (Zellers).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/video_pinball.bin
    copying pitfall.bin from ROMS/Pitfall! - Pitfall Harry's Jungle Adventure (Jungle Runner) (1982) (Activision, David Crane) (AX-018, AX-018-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/pitfall.bin
    copying pooyan.bin from ROMS/Pooyan (1983) (Konami) (RC 100-X 02) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/pooyan.bin
    copying private_eye.bin from ROMS/Private Eye (1984) (Activision, Bob Whitehead) (AG-034-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/private_eye.bin
    copying qbert.bin from ROMS/Q-bert (1983) (Parker Brothers - Western Technologies, Dave Hampton, Tom Sloper) (PB5360) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/qbert.bin
    copying riverraid.bin from ROMS/River Raid (1982) (Activision, Carol Shaw) (AX-020, AX-020-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/riverraid.bin
    copying road_runner.bin from patched version of ROMS/Road Runner (1989) (Atari - Bobco, Robert C. Polaro) (CX2663) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/road_runner.bin
    copying robotank.bin from ROMS/Robot Tank (Robotank) (1983) (Activision, Alan Miller) (AZ-028, AG-028-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/robotank.bin
    copying seaquest.bin from ROMS/Seaquest (1983) (Activision, Steve Cartwright) (AX-022) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/seaquest.bin
    copying sir_lancelot.bin from ROMS/Sir Lancelot (1983) (Xonox - K-Tel Software - Product Guild, Anthony R. Henderson) (99006, 6220) (PAL).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/sir_lancelot.bin
    copying skiing.bin from ROMS/Skiing - Le Ski (1980) (Activision, Bob Whitehead) (AG-005, CAG-005, AG-005-04) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/skiing.bin
    copying solaris.bin from ROMS/Solaris (The Last Starfighter, Star Raiders II, Universe) (1986) (Atari, Douglas Neubauer, Mimi Nyden) (CX26136) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/solaris.bin
    copying space_invaders.bin from ROMS/Space Invaders (1980) (Atari, Richard Maurer - Sears) (CX2632 - 49-75153) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/space_invaders.bin
    copying star_gunner.bin from ROMS/Stargunner (1983) (Telesys, Alex Leavens) (1005) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/star_gunner.bin
    copying surround.bin from ROMS/Surround (32 in 1) (Bit Corporation) (R320).bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/surround.bin
    copying tennis.bin from ROMS/Tennis - Le Tennis (1981) (Activision, Alan Miller) (AG-007, CAG-007) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/tennis.bin
    copying time_pilot.bin from ROMS/Time Pilot (1983) (Coleco - Woodside Design Associates, Harley H. Puthuff Jr.) (2663) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/time_pilot.bin
    copying trondead.bin from ROMS/TRON - Deadly Discs (TRON Joystick) (1983) (M Network - INTV - APh Technological Consulting, Jeff Ronne, Brett Stutz) (MT5662) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/trondead.bin
    copying tutankham.bin from ROMS/Tutankham (1983) (Parker Brothers, Dave Engman, Dawn Stockbridge) (PB5340) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/tutankham.bin
    copying up_n_down.bin from ROMS/Up 'n Down (1984) (SEGA - Beck-Tech, Steve Beck, Phat Ho) (009-01) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/up_n_down.bin
    copying venture.bin from ROMS/Venture (1982) (Coleco, Joseph Biel) (2457) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/venture.bin
    copying pong.bin from ROMS/Video Olympics - Pong Sports (Paddle) (1977) (Atari, Joe Decuir - Sears) (CX2621 - 99806, 6-99806, 49-75104) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/pong.bin
    copying wizard_of_wor.bin from ROMS/Wizard of Wor (1982) (CBS Electronics - Roklan, Joe Hellesen, Joe Wagner) (M8774, M8794) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/wizard_of_wor.bin
    copying yars_revenge.bin from ROMS/Yars' Revenge (Time Freeze) (1982) (Atari, Howard Scott Warshaw - Sears) (CX2655 - 49-75167) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/yars_revenge.bin
    copying zaxxon.bin from ROMS/Zaxxon (1983) (Coleco) (2454) ~.bin to /usr/local/lib/python3.7/dist-packages/atari_py/atari_roms/zaxxon.bin


### 2. Import Libraries:


```python
import random, os.path, math, glob, csv, os
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta

import matplotlib
%matplotlib inline
from IPython.display import clear_output
from plot import plot_all_data 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ######################## #
# USE ONLY IN GOOGLE COLAB #
%tensorflow_version 1.x 
# ######################## #

import gym
from gym.spaces.box import Box
from stable_baselines import bench
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
```

    TensorFlow 1.x selected.
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    


### 2. Hyperparameters


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Epsilon variables for epsilon-greedy:
epsilon_start    = 1.0
epsilon_final    = 0.01
epsilon_decay    = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay) 

# Misc agent variables
GAMMA = 0.99 
LR    = 1e-4 

# Memory
TARGET_NET_UPDATE_FREQ = 1000 
EXP_REPLAY_SIZE        = 100000 
BATCH_SIZE             = 32 

# Learning control variables
LEARN_START = 10000
MAX_FRAMES  = 200000 # Probably takes about an hour training. You can increase it if you have time!
UPDATE_FREQ = 1 

# Data logging parameters
ACTION_SELECTION_COUNT_FREQUENCY = 1000 

#DRQN Parameters
SEQUENCE_LENGTH = 8
```

### 3. Wrapper


```python
class WrapPOMDP(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPOMDP, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        # this method change MDP into POMDP
        pomdp = np.random.uniform()
        if pomdp >= 0.5:
          return observation.transpose(2, 0, 1)
        else:
          return observation.transpose(2, 0, 1) * 0.0
```

### 4. Abstract Agent Class (3 Points)


```python
class BaseAgent():
    def __init__(self, model, target_model, log_dir, env):
        self.device = device
        self.gamma = GAMMA
        self.lr = LR
        self.target_net_update_freq = TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = EXP_REPLAY_SIZE
        self.batch_size = BATCH_SIZE
        self.learn_start = LEARN_START
        self.update_freq = UPDATE_FREQ
        self.update_count = 0
        self.nstep_buffer = []
        self.rewards = []
        self.model = model
        self.target_model = target_model
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # compelete this method
        # * save log_dir
        # * save env
        # * make a list of action selections
        # * using load_state_dict, share learnable parameters (i.e. weights and biases) of
        #   self.model with self.target_model 
        # * move both model to correct device
        # * use Adam optimizer
        # * set both model to train mode
        #################################################################################
        self.log_dir = None
        self.env = None
        self.action_selections = None
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = None
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def update_target_model(self):
        # update target model:
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_ += torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            if count > 0:
                with open(os.path.join(self.log_dir, 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_ / count))

    def save_td(self, td, tstep):
        with open(os.path.join(self.log_dir, 'td.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, td))

    def save_reward(self, reward):
        self.rewards.append(reward)

    def save_action(self, action, tstep):
        self.action_selections[int(action)] += 1.0 / self.action_log_frequency
        if (tstep + 1) % self.action_log_frequency == 0:
            with open(os.path.join(self.log_dir, 'action_log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list([tstep] + self.action_selections))
            self.action_selections = [0 for _ in range(len(self.action_selections))]

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)
```

# 1. Simple DQN

We will implement a DQN model with experience replay. We implement a class for experience replay  `ExperienceReplayMemory`, for extarct features of observed picture of game we implement a class `DQN` that uses a CNN and inheritance from `nn.Module`. We implemented a wrapper class `WrapPyTorch` that you will use it in training loop.

## 1.1. Experience Replay (3 Points)


```python
#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# complete push and sample methods.
#################################################################################
class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        pass

    def sample(self, batch_size):
        return None

    def __len__(self):
        return len(self.memory)
#################################################################################
#                                   THE END                                     #
#################################################################################  
```

## 1.2. Network Declaration (4 Points)


```python
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # Initialize CNN Model :
        # conv1: out_channels:32, kernel_size=8, stride=4
        # conv2: out_channels:64, kernel_size=4, stride=2
        # conv3: out_channels:64, kernel_size=3, stride=1
        # fc1(512)
        # fc2(512)
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def forward(self, x):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete forward pass using initialized CNN Model. use Relu activation function 
        # for conv1, conv2, conv3, and fc1.  
        #################################################################################
        pass

        return x
        #################################################################################
        #                                   THE END                                     #
        #################################################################################   


```

 ## 1.3. Agent (6 Points)


```python
class Model():
    def __init__(self, env=None, log_dir):
        self.device = device

        self.gamma = GAMMA 
        self.lr = LR
        self.target_net_update_freq = TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = EXP_REPLAY_SIZE
        self.batch_size = BATCH_SIZE
        self.learn_start = LEARN_START
        self.update_freq = UPDATE_FREQ
        self.log_dir = log_dir
        self.rewards = []
        self.action_log_frequency = ACTION_SELECTION_COUNT_FREQUENCY
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # do followings line by line:
        # * make a list of action selections
        # * use shape of observation space to save number of features
        # * save naumber of actions
        # * use DQN class to declare model and target model (the game is 2-player game, so
        #   we declare 2 model)
        # * using load_state_dict, share learnable parameters (i.e. weights and biases) of
        #   self.model with self.target_model 
        # * use Adam optimizer
        #################################################################################
        self.num_feats = None
        self.num_actions = None
        self.model = None
        self.target_model = None
        super(Model, self).__init__()
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
        

    def prep_minibatch(self):
      
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # random transition batch is taken from experience replay memory
        # do followings line by line:
        # * sample from self.memory with batch size, and save result in transitions
        # * use transitions to save batch_state, batch_action, batch_reward, 
        #   batch_next_state as tensors
        # * save non_final_mask,  non_final_next_states as tensors, note that sometimes 
        #   all next states are false
        #################################################################################
        transitions = None
        
        batch_state, batch_action, batch_reward, batch_next_state = None

  

        batch_state = None
        batch_action = None
        batch_reward = None
        
        non_final_mask = None

        try: 
            non_final_next_states = None
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement calculation of loss (you should use "with torch.no_grad():" for target_model)
        #################################################################################

        loss = None
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return loss

    def update(self, s, a, r, s_, frame=0):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement update method to optimize model 
        # * push state, action, reward and new state to memoty
        # * note that if frame is lower than self.learn_start or frame % self.update_freq != 0
        #   return None
        # * take a random transition batch and compute loss
        # * optimize the model
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

        self.update_target_model()       
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0.1):

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################  
         # implement get_action method (epsilon greedy)
         # you should use "with torch.no_grad():"
         ################################################################################# 
        pass

        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def update_target_model(self):
      # update target model:
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())


```

 ## 1.3. Training Loop (4 Points)


```python
start=timer()

log_dir = "/tmp/gym/dqn"
try:
    os.makedirs(log_dir)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv')) \
        + glob.glob(os.path.join(log_dir, '*td.csv')) \
        + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv')) \
        + glob.glob(os.path.join(log_dir, '*action_log.csv'))
    for f in files:
        os.remove(f)

#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# do followings line by line:
# * use "make_atari" wrapper and save "PongNoFrameskip-v4" game to env
# * use "bench.Monitor" wrapper to know the episode reward, length, time and other data.
# * use "wrap_deepmind" wrapper to configure environment for DeepMind-style Atari.
# * use *WrapPyTorch*
# * save model
# * implement training loop
#################################################################################
env_id = "PongNoFrameskip-v4"
env =  None

model  = None

episode_reward = 0
observation = env.reset()
for frame_idx in range(1, MAX_FRAMES + 1):
    epsilon = None

    action = None

    model.save_action(action, frame_idx) #log action selection

    prev_observation = None
    observation, reward, done, _ = None

    ....

    if done:
        pass
        
#################################################################################
#                                   THE END                                     #
################################################################################# 
    if frame_idx % 1000 == 0:
        try:
            clear_output(True)
            plot_all_data(log_dir, env_id, 'DQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=True)
        except IOError:
            pass
 
env.close()
plot_all_data(log_dir, env_id, 'DQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=True)
```

# 2. DRQN

## 2.1. Experience Replay (4 Points)


```python
class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, transition):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete push method.
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################  

    def sample(self, batch_size):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete sample method.
        # notice that you should take these tips into consideration
        # * sample here will be trajectory not transition
        # * should use padding if trajectories aren't in same len
        #################################################################################
        samples = []
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################  
        return samples

    def __len__(self):
        return len(self.memory)
```

## 2.2. Network Declaration (4 Points)


```python
class RecurrentDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(RecurrentDQN,self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # Initialize CNN Model :
        # conv1: out_channels:32, kernel_size=8, stride=4
        # conv2: out_channels:64, kernel_size=4, stride=2
        # conv3: out_channels:64, kernel_size=3, stride=1
        # fc2(256)
        # GRU:  input_size: 256  hidden_size=256
        # fc2(256)
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def forward(self, x, bsize, time_step, hidden_state, cell_state):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete forward pass using initialized CNN and GRU Model. use Relu activation
        # function for conv1, conv2, conv3 and fc1 . 
        #################################################################################
        pass  
        return x, hidden
      
      def init_hidden_states(self,bsize):
        h = None
        pass
        return h
        #################################################################################
        #                                   THE END                                     #
        #################################################################################   


```

 ## 2.3. Agent (6 Points)


```python
class RecurrentModel():
    def __init__(self, env=None, log_dir):
        self.sequence_length = SEQUENCE_LENGTH
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # do followings line by line:
        # * use shape of observation space to save number of features
        # * save naumber of actions
        # * use DRQN class to declare model and target model 
        # * call parent class constructor
        # * declare memory
        # * reset hidden state
        #################################################################################
        self.num_feats = None
        self.num_actions = None
        model = None
        target_model = None
        super(RecurrentModel, self).__init__()
        self.memory = None
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        

    def prep_minibatch(self):
      
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # random transition batch is taken from experience replay memory
        # do followings line by line:
        # * sample from self.memory with batch size, and save result in transitions
        # * use transitions to save batch_state, batch_action, batch_reward, 
        #   batch_next_state as tensors
        # * reshape batch_state, batch_action, batch_reward, batch_next_state into 
        #   (batch_size, sequence_length, feat_size)
        # * get set of next states for end of each sequence
        # * save non_final_mask,  non_final_next_states as tensors, note that sometimes 
        #   all next states are false
        #################################################################################
        try: 
            non_final_next_states = None
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement calculation of loss (you should use "with torch.no_grad():" for target_model)
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return loss

    def update(self, s, a, r, s_, frame=0):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement update method to optimize model 
        # * push state, action, reward and new state to memoty
        # * note that if frame is lower than self.learn_start or frame % self.update_freq != 0
        #   return None
        # * take a random transition batch and compute loss
        # * clamp grad between -1 and 1
        # * optimize the model 
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

        self.update_target_model()       
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0.1):

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################  
         # implement get_action method (epsilon greedy)
         # you should use "with torch.no_grad():"
         ################################################################################# 
        pass

        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def reset_hx(self):
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))
```


      File "<ipython-input-1-865c4d6d8781>", line 185
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]
           ^
    IndentationError: expected an indented block



 ## 2.4. Training Loop (2 Points)


```python
start = timer()

log_dir = "/tmp/gym/drqn"
try:
    os.makedirs(log_dir)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv')) \
            + glob.glob(os.path.join(log_dir, '*td.csv')) \
            + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv')) \
            + glob.glob(os.path.join(log_dir, '*action_log.csv'))
    for f in files:
        os.remove(f)
#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# do followings line by line:
# * use "make_atari" wrapper and save "PongNoFrameskip-v4" game to env
# * use "bench.Monitor" wrapper to know the episode reward, length, time and other data.
# * use "wrap_deepmind" wrapper to configure environment for DeepMind-style Atari.
# * use *WrapPyTorch*
# * save model
# * implement training loop
#################################################################################
env_id = "PongNoFrameskip-v4"
env =  None

model  = None

episode_reward = 0
observation = env.reset()

for frame_idx in range(1, MAX_FRAMES + 1):
    epsilon = None

    action = None

    model.save_action(action, frame_idx) #log action selection

    prev_observation = None
    observation, reward, done, _ = None

    ....

    if done:
        model.finish_nstep()
        model.reset_hx()
        pass
#################################################################################
#                                   THE END                                     #
################################################################################# 
    if frame_idx % 10000 == 0:
        try:
            clear_output(True)
            plot_all_data(log_dir, env_id, 'DRQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1,
                          time=timedelta(seconds=int(timer() - start)), ipynb=True)
        except IOError:
            pass

env.close()
plot_all_data(log_dir, env_id, 'DRQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1,
              time=timedelta(seconds=int(timer() - start)), ipynb=True)
```

 ## 2.5. Interpret Results (4 Points)

Explain what you have seen. Is using recurrent memory improve performance? support your answer
