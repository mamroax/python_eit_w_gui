import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import functions as f

""" 0. construct mesh """
n_el = 16  # nb of electrodes
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = f.create_mesh(n_el, h0=0.1, fd=f.thorax) # creating an empty mash
else:
    mesh_obj = f.create_mesh(n_el, h0=0.1)

n_el = 16
mesh_obj = f.create_mesh(n_el, h0=0.05)
protocol_obj = f.create_protocol(n_el, dist_exc=1, step_meas=1, parser_meas="std")
jac = f.JAC(mesh_obj, protocol_obj)

breath_file = open('experimental.txt')
breath_matrix = [line.split("	") for line in breath_file]

# def jt_solve(self, v1: np.ndarray, v0: np.ndarray, normalize: bool = True) -> np.ndarray:
# нужно вставить два списка
# jac.jt_solve()

print(breath_matrix[0])

v1 = np.array([float(i) for i in breath_matrix[339]])
v0 = np.array([float(i) for i in breath_matrix[11]])


# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element

""" 1. problem setup """
# this step is not needed, actually
# mesh_0 = mesh.set_perm(mesh_obj, background=1.0)

# test function for altering the 'permittivity' in mesh
anomaly = [
    f.PyEITAnomaly_Circle(center=[0.4, 0], r=0, perm=10.0),
    f.PyEITAnomaly_Circle(center=[-0.4, 0], r=0, perm=10.0),
    f.PyEITAnomaly_Circle(center=[0, 0.5], r=0, perm=0.1),
    f.PyEITAnomaly_Circle(center=[0, -0.5], r=0, perm=0.1),
]
#mesh_new = f.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
#delta_perm = np.real(mesh_new.perm - mesh_obj.perm)

""" 2. FEM forward simulations """
# setup EIT scan conditions
protocol_obj = f.create_protocol(n_el, dist_exc=1, step_meas=1, parser_meas="std")

# calculate simulated data
fwd = f.EITForward(mesh_obj, protocol_obj)
v0 = v0
v1 = v1

""" 3. Construct using GREIT """
eit = f.GREIT(mesh_obj, protocol_obj)
eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
ds = eit.solve(v1, v0, normalize=True)
x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

# show alpha
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))

# ax = axes[0]
# im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")
# ax.axis("equal")
# ax.set_xlim([-1.2, 1.2])
# ax.set_ylim([-1.2, 1.2])
# ax.set_title(r"$\Delta$ Conductivity")
# fig.set_size_inches(6, 4)

# plot
"""
imshow will automatically set NaN (bad values) to 'w',
if you want to manually do so

import matplotlib.cm as cm
cmap = cm.gray
cmap.set_bad('w', 1.)
plt.imshow(np.real(ds), interpolation='nearest', cmap=cmap)
"""
ax = axes[0]
im = ax.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)
ax.axis("equal")

fig.colorbar(im, ax=axes.ravel().tolist())
# fig.savefig('../doc/images/demo_greit.png', dpi=96)
plt.show()


############################################################################################################33333##3#3#
root = tk.Tk()

x, breath_points, ind_max, ind_min = f.breath('experimental.txt', 1)

figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
# ax3.scatter(x, breath_points, color='b')
ax3.plot(x, breath_points, color='b')
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['average voltage difference'])
ax3.set_xlabel('measurement points')
ax3.set_ylabel('voltage(V)')
ax3.set_title('Breating chart')

x, breath_points, ind_max, ind_min = f.breath('experimental.txt', 3)

figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.plot(x, breath_points, color='b', lw = 2)
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['average voltage difference'])
ax3.set_xlabel('measurement points')
ax3.set_ylabel('voltage(V)')
ax3.set_title('Breating chart')

x, breath_points, ind_max, ind_min = f.breath('experimental.txt', 5)
figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.plot(x, breath_points, color='b', lw = 2)
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['average voltage difference'])
ax3.set_xlabel('measurement points')
ax3.set_ylabel('voltage(V)')
ax3.set_title('Breating chart')

root.mainloop()