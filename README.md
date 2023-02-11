<h1>Particle Tracking</h1>

<p>Package that performs tracking of colloidal particles in **2D**. Based on conventional tracking methods used in [this paper](https://crocker.seas.upenn.edu/CrockerGrier1996b.pdf). This is version based on a tracking code written in R (see [Repo](https://github.com/merrygoat/2d-particle-tracking)). NOTE: other available packages, such as ``Trackpy``, allow particle tracking in 2D & 3D.</p>

<h2>Installation</h2>

<p>Use

```
pip install particle-tracking
```

to simply install the code.
</p>

<h2>Using the code</h2>

<p>
the simples way to run the code is busing a `jupyer notebook`. See templates inside `particle_tracking/example/`. Start with
</p>

```
import particle_tracking as pt
```

<p>
The code has three main class objects `Tracker`, `Linker` & `Filtering'`, which allow to detect particles, reconstruct individual trajectories (using `Trackpy`'s linking method, and filter trajectories respectively. Simply use</p>

```
pt.Tracker()
pt.Linker()
pt.Filtering
```

<p>
to access the main functions. NOTE: all of the class obejct above require `params`, a dictionary containing parameters to exceute the tracking methods. See the next section to learn how to generate `params`.
</p>

<h3>Interactive Menu</h3>

<p>
The code includes an interactive widget that allows to select and modify quickly the input parameters. Import the interactive menu using</p>

```
from particle_tracking.src  import menus
menu = menus.MainMenu()
menu.options
```

![Interactive menu](/docs/interactive_menu.gif "Sample gif") 

<p>
Once the paramerts are selected use
</p>

```
params = menus.return_params(menu.options.children)
```
