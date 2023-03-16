import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import functions as f

""" 0. construct mesh """
n_el = 16  # nb of electrodes
# Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
mesh_obj = f.create_mesh(n_el, h0=0.05) # creating an empty mash(complete)
breath_file = open('experimental.txt') # opening a file w measurements
breath_matrix = [line.split("	") for line in breath_file] # creating a matrix out of a file
# print(breath_matrix[0]) # printing a matrix to assume it's working correctly

v1 = np.array([float(i) for i in breath_matrix[339]]) # first EIT frame
v0 = np.array([float(i) for i in breath_matrix[11]]) # second EIT frame with maximum voltage difference

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element
x, y = pts[:, 0], pts[:, 1]

""" 2. FEM simulation """
# setup EIT scan conditions
protocol_obj = f.create_protocol(n_el, dist_exc=8, step_meas=1, parser_meas="std")

# calculate simulated data
fwd = f.EITForward(mesh_obj, protocol_obj)
v0 = v0
v1 = v1

""" 3. JAC solver """
# Note: if the jac and the real-problem are generated using the same mesh,
# then, data normalization in solve are not needed.
# However, when you generate jac from a known mesh, but in real-problem
# (mostly) the shape and the electrode positions are not exactly the same
# as in mesh generating the jac, then data must be normalized.
jac = f.JAC(mesh_obj, protocol_obj)
eit = jac
eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
ds = eit.solve(v1, v0, normalize=True)
ds_n = f.sim2pts(pts, tri, np.real(ds))

# plot ground truth
fig, axes = plt.subplots(1, 2, constrained_layout=True)
fig.set_size_inches(9, 4)

ax = axes[0]

im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
ax.set_aspect("equal")

# plot EIT reconstruction
ax = axes[1]
im = ax.tripcolor(x, y, tri, ds_n, shading="flat")
for i, e in enumerate(mesh_obj.el_pos):
    ax.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
ax.set_aspect("equal")

fig.colorbar(im, ax=axes.ravel().tolist())
# plt.savefig('../doc/images/demo_jac.png', dpi=96)
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