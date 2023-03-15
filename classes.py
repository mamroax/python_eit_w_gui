import numpy as np
from numpy import sqrt
from dataclasses import dataclass
from typing import Union
from itertools import combinations
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
import scipy.linalg as la
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from typing import Callable


@dataclass
class PyEITMesh:
    """
    Pyeit buid-in mesh object

    Parameters
    ----------
    node : np.ndarray
        node of the mesh of shape (n_nodes, 2), (n_nodes, 3)
    element : np.ndarray
        elements of the mesh of shape (n_elem, 3) for 2D mesh, (n_elem, 4) for 3D mesh
    perm : Union[int, float, np.ndarray], optional
        permittivity on elements; shape (n_elems,), by default `None`.
        If `None`, a uniform permittivity on elements with a value 1 will be generated.
        If perm is int or float, uniform permittivity on elements with value of perm will be generated.
    el_pos : np.ndarray
        node corresponding to each electrodes of shape (n_el, 1)
    ref_node : int
        reference node. ref_node should not be on electrodes, default 0.
    """

    node: np.ndarray
    element: np.ndarray
    perm: Union[int, float, np.ndarray] = None
    el_pos: np.ndarray = np.arange(16)
    ref_node: int = 0

    def __post_init__(self) -> None:
        """Checking of the inputs"""
        self.element = self._check_element(self.element)
        self.node = self._check_node(self.node)
        self.perm = self.get_valid_perm(self.perm)
        self.ref_node = self._check_ref_node(self.ref_node)

    def print_stats(self):
        """
        Print mesh or tetrahedral status

        Parameters
        ----------
        p: array_like
            coordinates of nodes (x, y) in 2D, (x, y, z) in 3D
        t: array_like
            connectives forming elements

        Notes
        -----
        a simple function for illustration purpose only.
        print the status (size) of nodes and elements
        """
        text_2D_3D = "3D" if self.is_3D else "2D"
        print(f"{text_2D_3D} mesh status:")
        print(f"{self.n_nodes} nodes, {self.n_elems} elements")

    def _check_element(self, element: np.ndarray) -> np.ndarray:
        """
        Check nodes element
        return nodes [x,y,z]

        Parameters
        ----------
        node : np.ndarray, optional
            nodes [x,y] ; shape (n_elem,3)
            nodes [x,y,z] ; shape (n_nodes,4)

        Returns
        -------
        np.ndarray
            nodes [x,y,z] ; shape (n_nodes,3)

        Raises
        ------
        TypeError
            raised if perm is not ndarray and of shape (n_tri,)
        """
        if not isinstance(element, np.ndarray):
            raise TypeError(f"Wrong type of {element=}, expected an ndarray")
        if element.ndim != 2:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray with 2 dimensions"
            )
        if element.shape[1] not in [3, 4]:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )

        return element

    def _check_node(self, node: np.ndarray) -> np.ndarray:
        """
        Check nodes shape
        return nodes [x,y,z]

        Parameters
        ----------
        node : np.ndarray, optional
            nodes [x,y] ; shape (n_nodes,2) (in that case z will be set 0)
            nodes [x,y,z] ; shape (n_nodes,3)

        Returns
        -------
        np.ndarray
            nodes [x,y,z] ; shape (n_nodes,3)

        Raises
        ------
        TypeError
            raised if perm is not ndarray and of shape (n_tri,)
        """
        if not isinstance(node, np.ndarray):
            raise TypeError(f"Wrong type of {node=}, expected an ndarray")
        if node.ndim != 2:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray with 2 dimensions"
            )
        if node.shape[1] not in [2, 3]:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )
        # convert nodes [x,y] to nodes [x,y,0]
        if node.shape[1] == 2:
            node = np.hstack((node, np.zeros((node.shape[0], 1))))

        return node

    def get_valid_perm(self, perm: Union[int, float, np.ndarray] = None) -> np.ndarray:
        """
        Return a valid permittivity on element

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            Permittivity on elements ; shape (n_elems,), by default `None`.
            If `None`, a uniform permittivity on elements with a value 1 will be used.
            If perm is int or float, uniform permittivity on elements will be used.

        Returns
        -------
        np.ndarray
            permittivity on elements ; shape (n_elems,)

        Raises
        ------
        TypeError
            check if perm passed as ndarray has a shape (n_elems,)
        """

        if perm is None:
            return np.ones(self.n_elems, dtype=float)
        elif isinstance(perm, (int, float)):
            return np.ones(self.n_elems, dtype=float) * perm

        if not isinstance(perm, np.ndarray) or perm.shape != (self.n_elems,):
            raise TypeError(
                f"Wrong type/shape of {perm=}, expected an ndarray; shape ({self.n_elems}, )"
            )
        return perm

    def _check_ref_node(self, ref: int = 0) -> int:
        """
        Return a valid reference electrode node

        Parameters
        ----------
        ref : int, optional
            node number of reference node, by default 0
            If the choosen node is on electrode node, a node-list in
            np.arange(0, len(el_pos)+1) will be checked iteratively until
            a non-electrode node is selected.

        returns
        -------
        int
            valid reference electrode node
        """
        default_ref = np.setdiff1d(np.arange(len(self.el_pos) + 1), self.el_pos)[0]
        return ref if ref not in self.el_pos else default_ref
        # assert ref < self.n_nodes

    def set_ref_node(self, ref: int = 0) -> None:
        """
        Set reference electrode node

        Parameters
        ----------
        ref : int, optional
            node number of reference electrode
        """
        self.ref_node = self._check_ref_node(ref)

    @property
    def n_nodes(self) -> int:
        """
        Returns
        -------
        int
            number of nodes contained in the mesh
        """
        return self.node.shape[0]

    @property
    def n_elems(self) -> int:
        """
        Returns
        -------
        int
            number of elements contained in the mesh
        """
        return self.element.shape[0]

    @property
    def n_vertices(self) -> int:
        """
        Returns
        -------
        int
            number of vertices of the elements contained in the mesh
        """
        return self.element.shape[1]

    @property
    def n_el(self) -> int:
        """
        Returns
        -------
        int
            number of electrodes
        """
        return self.el_pos.shape[0]

    @property
    def elem_centers(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            center of the nodes [x,y,z]; shape (n_elems,3)
        """
        return np.mean(self.node[self.element], axis=1)

    @property
    def is_3D(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            True if the mesh is a 3D mesh (use elements with 4 vertices)
        """
        return self.n_vertices == 4

    @property
    def is_2D(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            True if the mesh is a 2D mesh (use elements with 3 vertices)
        """
        return self.n_vertices == 3


class DISTMESH:
    """class for distmesh"""

    def __init__(
        self,
        fd,
        fh,
        h0=0.1,
        p_fix=None,
        bbox=None,
        density_ctrl_freq=30,
        deltat=0.1,
        dptol=0.001,
        ttol=0.1,
        Fscale=1.2,
        verbose=False,
    ):
        """initial distmesh class

        Parameters
        ----------
        fd : str
            function handle for distance of boundary
        fh : str
            function handle for distance distributions
        h0 : float, optional
            Distance between points in the initial distribution p0,
            default=0.1 For uniform meshes, h(x,y) = constant,
            the element size in the final mesh will usually be
            a little larger than this input.
        p_fix : array_like, optional
            fixed points, default=[]
        bbox : array_like, optional
            bounding box for region, bbox=[xmin, ymin, xmax, ymax].
            default=[-1, -1, 1, 1]
        density_ctrl_freq : int, optional
            cycles of iterations of density control, default=20
        deltat : float, optional
            mapping forces to distances, default=0.2
        dptol : float, optional
            exit criterion for minimal distance all points moved, default=0.01
        ttol : float, optional
            enter criterion for re-delaunay the lattices, default=0.1
        Fscale : float, optional
            rescaled string forces, default=1.2
            if set too small, points near boundary will be pushed back
            if set too large, points will be pushed towards boundary

        Notes
        -----
        """
        # shape description
        self.fd = fd
        self.fh = fh
        self.h0 = h0

        # a small gap, allow points who are slightly outside of the region
        self.deps = np.sqrt(np.finfo(np.double).eps) * h0
        self.geps = 1e-1 * h0

        # control the distmesh computation flow
        self.densityctrlfreq = density_ctrl_freq
        self.dptol = dptol
        self.ttol = ttol
        self.Fscale = Fscale
        self.deltat = deltat

        # default bbox is 2D
        if bbox is None:
            bbox = [[-1, -1], [1, 1]]
        # p : coordinates (x,y) or (x,y,z) of meshes
        self.n_dim = np.shape(bbox)[1]
        p = bbox2d_init(h0, bbox)


        # control debug messages
        self.verbose = verbose
        self.num_triangulate = 0
        self.num_density = 0
        self.num_move = 0

        # keep points inside (minus distance) with a small gap (geps)
        p = p[fd(p) < self.geps]  # pylint: disable=E1136

        # rejection points by sampling on fh
        r0 = 1.0 / fh(p) ** self.n_dim
        selection = np.random.rand(p.shape[0]) < (r0 / np.max(r0))
        p = p[selection]

        # specify fixed points
        if p_fix is None:
            p_fix = []
        self.pfix = p_fix
        self.nfix = len(p_fix)

        # remove duplicated points of p and p_fix
        # avoid overlapping of mesh points
        if self.nfix > 0:
            p = remove_duplicate_nodes(p, p_fix, self.geps)
            p = np.vstack([p_fix, p])

        # store p and N
        self.N = p.shape[0]
        self.p = p

        # initialize pold with inf: it will be re-triangulate at start
        self.pold = np.inf * np.ones((self.N, self.n_dim))

        # build edges list for triangle or tetrahedral. i.e., in 2D triangle
        # edge_combinations is [[0, 1], [1, 2], [2, 0]]
        self.edge_combinations = list(combinations(range(self.n_dim + 1), 2))

        # triangulate, generate simplices and bars
        self.triangulate()

    def is_retriangulate(self):
        """test whether re-triangulate is needed"""
        return np.max(dist(self.p - self.pold)) > (self.h0 * self.ttol)

    def triangulate(self):
        """retriangle by delaunay"""
        self.debug("enter triangulate = ", self.num_triangulate)
        self.num_triangulate += 1
        # pnew[:] = pold[:] makes a new copy, not reference
        self.pold = self.p.copy()

        # triangles where the points are arranged counterclockwise
        # QJ parameter so tuples don't exceed boundary
        tri = Delaunay(self.p, qhull_options="QJ").simplices
        pmid = np.mean(self.p[tri], axis=1)

        # keeps only interior points
        t = tri[self.fd(pmid) < -self.geps]
        # extract edges (bars)
        bars = t[:, self.edge_combinations].reshape((-1, 2))
        # sort and remove duplicated edges, eg (1,2) and (2,1)
        # note : for all edges, non-duplicated edge is boundary edge
        bars = np.sort(bars, axis=1)
        bars_tuple = bars.view([("", bars.dtype)] * bars.shape[1])

        self.bars = np.unique(bars_tuple).view(bars.dtype).reshape((-1, 2))
        self.t = t

    def bar_length(self):
        """the forces of bars (python is by-default row-wise operation)"""
        # two node of a bar
        bars_a, bars_b = self.p[self.bars[:, 0]], self.p[self.bars[:, 1]]
        barvec = bars_a - bars_b

        # L : length of bars, must be column ndarray (2D)
        L = dist(barvec).reshape((-1, 1))
        # density control on bars
        hbars = self.fh((bars_a + bars_b) / 2.0).reshape((-1, 1))
        # L0 : desired lengths (Fscale matters!)
        L0 = hbars * self.Fscale * sqrt(np.sum(L**2) / np.sum(hbars**2))

        return L, L0, barvec

    def bar_force(self, L, L0, barvec):
        """forces on bars"""
        # abs(forces)
        F = np.maximum(L0 - L, 0)
        # normalized and vectorized forces
        Fvec = F * (barvec / L)
        # now, we get forces and sum them up on nodes
        # using sparse matrix to perform automatic summation
        # rows : left, left, right, right (2D)
        # : left, left, left, right, right, right (3D)
        # cols : x, y, x, y (2D)
        # : x, y, z, x, y, z (3D)
        data = np.hstack([Fvec, -Fvec])
        if self.n_dim == 2:
            rows = self.bars[:, [0, 0, 1, 1]]
            cols = np.dot(np.ones(np.shape(F)), np.array([[0, 1, 0, 1]]))
        else:
            rows = self.bars[:, [0, 0, 0, 1, 1, 1]]
            cols = np.dot(np.ones(np.shape(F)), np.array([[0, 1, 2, 0, 1, 2]]))
        # sum nodes at duplicated locations using sparse matrices
        Ftot = csr_matrix(
            (data.reshape(-1), [rows.reshape(-1), cols.reshape(-1)]),
            shape=(self.N, self.n_dim),
        )
        Ftot = Ftot.toarray()
        # zero out forces at fixed points, as they do not move
        Ftot[0 : len(self.pfix)] = 0

        return Ftot

    def density_control(self, L, L0, dscale=3.0):
        """
        Density control - remove points that are too close
        L0 : Kx1, L : Kx1, bars : Kx2
        bars[L0 > 2*L] only returns bar[:, 0] where L0 > 2L
        """
        self.debug("enter density control = ", self.num_density)
        self.num_density += 1
        # print(self.num_density, self.p.shape)
        # quality control
        ixout = (L0 > dscale * L).ravel()
        ixdel = np.setdiff1d(self.bars[ixout, :].reshape(-1), np.arange(self.nfix))
        self.p = self.p[np.setdiff1d(np.arange(self.N), ixdel)]
        # Nold = N
        self.N = self.p.shape[0]
        self.pold = np.inf * np.ones((self.N, self.n_dim))
        # print('density control ratio : %f' % (float(N)/Nold))

    def move_p(self, Ftot):
        """update p"""
        self.debug("  number of moves = ", self.num_move)
        self.num_move += 1
        # move p along forces
        self.p += self.deltat * Ftot

        # if there is any point ends up outside
        # move it back to the closest point on the boundary
        # using the numerical gradient of distance function
        d = self.fd(self.p)
        ix = d > 0
        if ix.any():
            self.p[ix] = edge_project(self.p[ix], self.fd, self.geps)

        # check whether convergence : no big movements
        delta_move = self.deltat * np.max(dist(Ftot[d < -self.geps]))
        self.debug("  delta_move = ", delta_move)
        score = delta_move < self.dptol * self.h0

        return score

    def debug(self, *args):
        """print debug messages"""
        if self.verbose:
            print(*args)

def tet_volume(xyz):
    """calculate the volume of tetrahedron"""
    s = xyz[[2, 3, 0]] - xyz[[1, 2, 3]]
    v_tot = (1.0 / 6.0) * la.det(s)
    return v_tot

def edge_grad(p, fd, h0=1.0):
    """
    project points back on the boundary (where fd=0) using numerical gradient
    3D, ND compatible

    Parameters
    ----------
    p : array_like
        points on 2D, 3D
    fd : str
        function handler of distances
    h0 : float
        minimal distance

    Returns
    -------
    array_like
        gradients of points on the boundary

    Note
    ----
        numerical gradient:
        f'_x = (f(p+delta_x) - f(p)) / delta_x
        f'_y = (f(p+delta_y) - f(p)) / delta_y
        f'_z = (f(p+delta_z) - f(p)) / delta_z

        you should specify h0 according to your actual mesh size
    """
    # d_eps = np.sqrt(np.finfo(float).eps)*h0
    # d_eps = np.sqrt(np.finfo(float).eps)
    d_eps = 1e-8 * h0

    # get dimensions
    if np.ndim(p) == 1:
        p = p[:, np.newaxis]

    # distance
    d = fd(p)

    # calculate the gradient of each axis
    ndim = p.shape[1]
    pts_xyz = np.repeat(p, ndim, axis=0)
    delta_xyz = np.repeat([np.eye(ndim)], p.shape[0], axis=0).reshape(-1, ndim)
    deps_xyz = d_eps * delta_xyz
    g_xyz = (fd(pts_xyz + deps_xyz) - np.repeat(d, ndim, axis=0)) / d_eps

    # normalize gradient, avoid divide by zero
    g = g_xyz.reshape(-1, ndim)
    g2 = np.sum(g**2, axis=1)

    # move unit
    g_num = g / g2[:, np.newaxis] * d[:, np.newaxis]

    return g_num

def edge_project(pts, fd, h0=1.0):
    """project points back on edge"""
    g_vec = edge_grad(pts, fd, h0)
    return pts - g_vec

def remove_duplicate_nodes(p, pfix, geps):
    """remove duplicate points in p who are closed to pfix. 3D, ND compatible

    Parameters
    ----------
    p : array_like
        points in 2D, 3D, ND
    pfix : array_like
        points that are fixed (can not be moved in distmesh)
    geps : float, optional (default=0.01*h0)
        minimal distance that two points are assumed to be identical

    Returns
    -------
    array_like
        non-duplicated points
    """
    for row in pfix:
        pdist = dist(p - row)
        # extract non-duplicated row slices
        p = p[pdist > geps]
    return p

def bbox2d_init(h0, bbox):
    """
    generate points in 2D bbox (not including the ending point of bbox)

    Parameters
    ----------
    h0 : float
        minimal distance of points
    bbox : array_like
        [[x0, y0],
         [x1, y1]]

    Returns
    -------
    array_like
        points in bbox
    """
    x, y = np.meshgrid(
        np.arange(bbox[0][0], bbox[1][0], h0),
        np.arange(bbox[0][1], bbox[1][1], h0 * sqrt(3) / 2.0),
        indexing="xy",
    )
    # shift even rows of x
    # shift every second row h0/2 to the right, therefore,
    # all points will be a distance h0 from their closest neighbors
    x[1::2, :] += h0 / 2.0
    # p : Nx2 ndarray
    p = np.array([x.ravel(), y.ravel()]).T
    return p


def breath(file_path, num_of_frames):
    try:
        breath_file = open(file_path)
        breath_matrix = [line.split("	") for line in breath_file]
        breath_points = []
        x = [i for i in range(0, 450, num_of_frames)]
        min_value_index = 0
        max_value_index = 0
        min_value = 1
        max_value = 0
        for i in range(0, len(breath_matrix), num_of_frames):
            summa = 0
            for k in range(num_of_frames):
                for j in breath_matrix[i+k]:
                    summa += float(j)
            average = summa/208/num_of_frames
            if average < min_value:
                min_value = average
                min_value_index = i
            if average > max_value:
                max_value = average
                max_value_index = i
            breath_points.append(average)
    except Exception:
        print('Error')
    finally:
        breath_file.close()
    return [x, breath_points, min_value_index, max_value_index]

def area_uniform(p):
    """uniform mesh distribution

    Parameters
    ----------
    p : array_like
        points coordinates

    Returns
    -------
    array_like
        ones

    """
    return np.ones(p.shape[0])

def thorax(pts):
    """
    thorax polygon signed distance function

    Thorax contour points coordinates are taken from
    a thorax simulation based on EIDORS
    """
    poly = [
        (0.0487, 0.6543),
        (0.1564, 0.6571),
        (0.2636, 0.6697),
        (0.3714, 0.6755),
        (0.479, 0.6686),
        (0.5814, 0.6353),
        (0.6757, 0.5831),
        (0.7582, 0.5137),
        (0.8298, 0.433),
        (0.8894, 0.3431),
        (0.9347, 0.2452),
        (0.9698, 0.1431),
        (0.9938, 0.0379),
        (1.0028, -0.0696),
        (0.9914, -0.1767),
        (0.9637, -0.281),
        (0.9156, -0.3771),
        (0.8359, -0.449),
        (0.7402, -0.499),
        (0.6432, -0.5463),
        (0.5419, -0.5833),
        (0.4371, -0.6094),
        (0.3308, -0.6279),
        (0.2243, -0.6456),
        (0.1168, -0.6508),
        (0.0096, -0.6387),
        (-0.098, -0.6463),
        (-0.2058, -0.6433),
        (-0.313, -0.6312),
        (-0.4181, -0.6074),
        (-0.5164, -0.5629),
        (-0.6166, -0.5232),
        (-0.7207, -0.4946),
        (-0.813, -0.4398),
        (-0.8869, -0.3614),
        (-0.933, -0.2647),
        (-0.9451, -0.1576),
        (-0.9425, -0.0498),
        (-0.9147, 0.0543),
        (-0.8863, 0.1585),
        (-0.8517, 0.2606),
        (-0.8022, 0.3565),
        (-0.7413, 0.4455),
        (-0.6664, 0.5231),
        (-0.5791, 0.5864),
        (-0.4838, 0.6369),
        (-0.3804, 0.667),
        (-0.2732, 0.6799),
        (-0.1653, 0.6819),
        (-0.0581, 0.6699),
    ]
    poly_obj = Polygon(poly)
    return fd_polygon(poly_obj, pts)

def fd_polygon(poly, pts):
    """return signed distance of polygon"""
    pts_ = [Point(p) for p in pts]
    # calculate signed distance
    dist = [poly.exterior.distance(p) for p in pts_]
    sign = np.sign([-int(poly.contains(p)) + 0.5 for p in pts_])

    return sign * dist

def circle(pts, pc=None, r=1.0):
    """
    Distance function for the circle centered at pc = [xc, yc]

    Parameters
    ----------
    pts : array_like
        points on 2D
    pc : array_like, optional
        center of points
    r : float, optional
        radius

    Returns
    -------
    array_like
        distance of (points - pc) - r

    Note
    ----
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    if pc is None:
        pc = [0, 0]
    return dist(pts - pc) - r

def dist(p):
    """distances to origin of nodes. '3D', 'ND' compatible
        distances of points to origin
    """
    if p.ndim == 1:
        d = np.sqrt(np.sum(p**2))
    else:
        d = np.sqrt(np.sum(p**2, axis=1))
    return d

def create_mesh(
    n_el: int = 16,
    fd: Callable = None,
    fh: Callable = area_uniform,
    h0: float = 0.1,
    p_fix: np.ndarray = None,
    bbox: np.ndarray = None,
) -> PyEITMesh:
    """
    Generating 2D/3D meshes using distmesh (pyEIT built-in)

    Parameters
    ----------
    n_el: int
        number of electrodes (point-type electrode)
    fd: function
        distance function (circle in 2D, ball in 3D)
    fh: function
        mesh size quality control function
    p_fix: NDArray
        fixed points
    bbox: NDArray
        bounding box
    h0: float
        initial mesh size, default=0.1

    Returns
    -------
    PyEITMesh
        mesh object
    """

    # test conditions if fd or/and bbox are none
    bbox = np.array([[-1, -1], [1, 1]]) # here i may have some problems////////////////////////////////////////////
    # list is converted to Numpy array so we can use it then (calling shape method..)
    bbox = np.array(bbox)
    n_dim = bbox.shape[1]  # bring dimension
    fd = circle

    if n_dim not in [2, 3]:
        raise TypeError("distmesh only supports 2D or 3D")
    if bbox.shape[0] != 2:
        raise TypeError("please specify lower and upper bound of bbox")

    p_fix = np.array(
        [
            (-0.098, -0.6463),
            (-0.4181, -0.6074),
            (-0.7207, -0.4946),
            (-0.933, -0.2647),
            (-0.9147, 0.0543),
            (-0.8022, 0.3565),
            (-0.5791, 0.5864),
            (-0.1653, 0.6819),
            (0.1564, 0.6571),
            (0.5814, 0.6353),
            (0.8298, 0.433),
            (0.9698, 0.1431),
            (0.9914, -0.1767),
            (0.8359, -0.449),
            (0.5419, -0.5833),
            (0.2243, -0.6456),
        ]
    )

    # 1. build mesh
    p, t = build(fd, fh, pfix=p_fix, bbox=bbox, h0=h0)
    # 2. check whether t is counter-clock-wise, otherwise reshape it
    t = check_order(p, t)
    # 3. generate electrodes, the same as p_fix (top n_el)
    el_pos = np.arange(n_el)
    return PyEITMesh(element=t, node=p, perm=None, el_pos=el_pos, ref_node=0)


def build(
    fd, fh, pfix=None, bbox=None, h0=0.1, densityctrlfreq=10, maxiter=500, verbose=False
):
    """main function for distmesh

    See Also
    --------
    DISTMESH : main class for distmesh

    Parameters
    ----------
    fd, fh : function
        signed distance and quality control function
    pfix : ndarray
        fixed points on the boundary, usually used for electrodes positions
    maxiter : int, optional
        maximum iteration numbers, default=1000

    Returns
    -------
    p : array_like
        points on 2D bbox
    t : array_like
        triangles describe the mesh structure

    Notes
    -----
    there are many python or hybrid python + C implementations in github,
    this implementation is merely implemented from scratch
    using PER-OLOF PERSSON's Ph.D thesis and SIAM paper.

    .. [1] P.-O. Persson, G. Strang, "A Simple Mesh Generator in MATLAB".
       SIAM Review, Volume 46 (2), pp. 329-345, June 2004

    Also, the user should be aware that, equal-edged tetrahedron cannot fill
    space without gaps. So, in 3D, you can lower dptol, or limit the maximum
    iteration steps.
    """
    # parsing arguments
    # make sure : g_Fscale < 1.5
    mode_3D = False
    if bbox is None:
        g_dptol, g_ttol, g_Fscale, g_deltat = 0.001, 0.1, 1.2, 0.2
    else:
        # perform error check on bbox
        bbox = np.array(bbox)
        if (bbox.ndim == 1) or (bbox.shape[1] not in [2, 3]):
            raise TypeError("only 2D, 3D are supported, bbox = ", bbox)
        if bbox.shape[0] != 2:
            raise TypeError("please specify lower and upper bound of bbox")
        # default parameters for 2D
        g_dptol, g_ttol, g_Fscale, g_deltat = 0.001, 0.1, 1.3, 0.2

    # initialize distmesh
    dm = DISTMESH(
        fd,
        fh,
        h0=h0,
        p_fix=pfix,
        bbox=bbox,
        density_ctrl_freq=densityctrlfreq,
        deltat=g_deltat,
        dptol=g_dptol,
        ttol=g_ttol,
        Fscale=g_Fscale,
        verbose=verbose,
    )

    # now iterate to push to equilibrium
    for i in range(maxiter):
        if dm.is_retriangulate():
            # print("triangulate = %d" % dm.num_triangulate)
            dm.triangulate()

        # calculate bar forces
        L, L0, barvec = dm.bar_length()

        # density control
        if (i % densityctrlfreq) == 0 and (L0 > 2 * L).any():
            dm.density_control(L, L0)
            # continue to triangulate
            continue

        # calculate bar forces
        Ftot = dm.bar_force(L, L0, barvec)

        # update p
        converge = dm.move_p(Ftot)
        # the stopping ctriterion (movements interior are small)
        if converge:
            break

    # at the end of iteration, (p - pold) is small, so we recreate delaunay
    dm.triangulate()

    # you should remove duplicate nodes and triangles
    return dm.p, dm.t

def check_order(no2xy, el2no):
    """
    loop over all elements, calculate the Area of Elements (aoe)
    if AOE > 0, then the order of element is correct
    if AOE < 0, reorder the element

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity

    Returns
    -------
    NDArray
        ae, area of each element

    Notes
    -----
    tetrahedron should be parsed that the sign of volume is [1, -1, 1, -1]
    """
    el_num, n_vertices = np.shape(el2no)
    # select ae function
    if n_vertices == 3:
        _fn = tri_area
    elif n_vertices == 4:
        _fn = tet_volume
    # calculate ae and re-order tri if necessary
    for ei in range(el_num):
        no = el2no[ei, :]
        xy = no2xy[no, :]
        v = _fn(xy)
        # if CCW, area should be positive, otherwise,
        if v < 0:
            el2no[ei, [1, 2]] = el2no[ei, [2, 1]]

    return el2no

def tri_area(xy):
    """
    return area of a triangle, given its tri-coordinates xy

    Parameters
    ----------
    xy : NDArray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    float
        area of this element
    """
    s = xy[[2, 0]] - xy[[1, 2]]
    a_tot = 0.50 * la.det(s)
    # (should be positive if tri-points are counter-clockwise)
    return a_tot