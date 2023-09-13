# ShipLoading-Optimization
A multi-objective analysis and optimization of a container ship loading problem with Pymoo library.

## Problem description
(Credits to MODA teaching team in LIACS, Leiden University)
The container ship is traveling on a route involving five harbors in northern Europe: Rotterdam, Hamburg, Kiel, Aarhus, and Copenhagen. You have been asked to help the Captain with the loading and unloading plan (for the destination harbors) that produces:
• Even and well-balanced solution
• A solution that is easy to unload (so that containers that are earlier unloaded are not beneath containers that are later unloaded)
• Solutions that can integrate as many containers as possible

The following details about the problem are provided:
1. The ship is now docked in Rotterdam. After visiting Rotterdam, the ship will be traveling to Hamburg, Aarhus, and Copenhagen, in that order. The containers destined for Rotterdam have been unloaded and the containers destined for the remaining three harbors have to be loaded on the ship.
2. It is a rather small container ship. It has eight bays, three tiers and four rows. The layout of the ship is shown in the figure below.
3. Containers should be loaded such that a container that has to be unloaded earlier should be placed in a higher position.
4. Each cell in the above figure is able to hold a forty-foot container. That is, the ship has room for 8 × 3 × 4 = 96 forty-foot containers.
5. A forty-foot container can be placed on-top of another forty-foot container, but not on top of an empty cell. We assume that only forty-foot containers are loaded on the ship. Each container has a destination port.
6. Each container i has a certain weight wi. If a container is much heavier than another i container j, say wi − wj > δw, then it is not allowed to place wi on top of wj.
7. The containers should be balanced in a way that is evenly distributed. The longitudinal center of gravity should be as close as possible to the middle of the container section of the ship. Secondly, the latitudinal center of gravity should be as much to the middle as possible. Note that the center of gravity can be computed as the weighted sum of the centroids of the containers, where the weights are the total weight of the containers.
